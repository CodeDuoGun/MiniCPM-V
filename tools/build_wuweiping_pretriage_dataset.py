#!/usr/bin/env python3
"""Build a privacy-reduced multimodal pre-triage SFT dataset for MiniCPM-V.

The source records contain direct identifiers, diagnoses, and prescriptions. This
builder deliberately emits none of those fields. Image-linked findings are weak
labels derived from the same encounter's chief complaint and tongue description;
they must be clinically reviewed before production use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from image_integrity import is_complete_image


DEFAULT_RECORDS = Path(
    "/Users/tangxueduo/Projects/LLaMA-Factory/medical/data/zongyuan_wuweiping/"
    "wuweiping_record_20260525.json"
)
DEFAULT_DIALOGUES = Path(
    "/Users/tangxueduo/Projects/LLaMA-Factory/medical/processed_data/wuweiping/"
    "wuweiping_postasr_speaker_content.json"
)
SYSTEM_PROMPT = (
    "你是线上预问诊助手，只做病史采集、图片客观描述、风险提示和就医建议。"
    "你不能做最终诊断，不能替代线下医生，不能开具处方或给出具体用药方案。"
    "图片只能用于记录舌面和患处的可见表现，不能据此确诊。"
    "回答应简洁、温和，先追问关键病史，再给风险提示。"
    "遇到胸痛、呼吸困难、意识障碍、高热不退、严重过敏等危险信号，建议立即线下就医或急诊。"
    "儿童、孕妇或哺乳期，以及肝肾功能异常者，必须先询问年龄体重、孕哺情况、过敏史、"
    "基础病、肝肾功能和正在使用的药物等禁忌信息。"
)
DIRECT_IDENTIFIER_RE = re.compile(
    r"(?<!\d)(?:1\d{10}|\d{17}[\dXx]|\d{15})(?!\d)|"
    r"(?:姓名|身份证(?:号)?|手机号|电话|住址|地址)\s*[:：]?\s*[^，。；\n]{2,30}"
)
INTENT_RE = re.compile(r"\s*\[意图\s*:\s*[^\]]+\]\s*")
TONGUE_RE = re.compile(
    r"(舌(?:质)?(?:淡红|淡白|淡|红|暗红|紫暗|胖大|有齿痕|瘦薄|裂纹){1,3}"
    r"(?:[，,、 ]*苔(?:薄白|白|黄|黄腻|白腻|少|剥脱|厚腻|薄黄|薄))?)"
)
DANGER_RE = re.compile(
    r"胸痛|胸闷伴(?:呼吸困难|大汗)|呼吸困难|喘不过气|意识(?:不清|障碍)|昏迷|"
    r"高热不退|严重过敏|喉头水肿|口唇发紫|抽搐|大出血"
)
LIVER_KIDNEY_RE = re.compile(
    r"肝功能异常|肾功能异常|肝损伤|肾损伤|肝炎|肾炎|肝硬化|肾衰|"
    r"谷丙|谷草|肌酐|尿素氮|胆红素[^，。；\n]{0,10}(?:高|↑)"
)
PREGNANCY_RE = re.compile(r"孕妇|怀孕|妊娠|孕期|哺乳|产后")
SYMPTOMS = (
    "潮红",
    "红斑",
    "丘疹",
    "脓疱",
    "脓包",
    "结节",
    "瘙痒",
    "发痒",
    "发烫",
    "灼热",
    "疼痛",
    "肿胀",
    "脱屑",
    "干燥",
    "渗液",
    "破溃",
    "色沉",
    "色斑",
    "痘印",
    "痘坑",
    "脱发",
)


def text(value: Any) -> str:
    return str(value or "").strip()


def clean_short(value: Any, limit: int = 120) -> str:
    value = INTENT_RE.sub("", text(value))
    value = DIRECT_IDENTIFIER_RE.sub("[已脱敏]", value)
    value = re.sub(r"\s+", " ", value).strip(" ，,。；;")
    return value[:limit]


def safe_age(value: Any) -> int | None:
    match = re.search(r"\d{1,3}", text(value))
    if not match:
        return None
    age = int(match.group())
    return age if 0 <= age <= 120 else None


def find_tongue_description(*values: Any) -> str:
    matches: list[str] = []
    for value in values:
        matches.extend(TONGUE_RE.findall(text(value)))
    return matches[-1] if matches else ""


def symptom_summary(value: Any) -> str:
    """Keep symptoms and duration while excluding disease/diagnosis labels."""
    source = clean_short(value, 120)
    found = []
    for symptom in SYMPTOMS:
        normalized = "脓疱" if symptom == "脓包" else symptom
        if symptom in source and normalized not in found:
            found.append(normalized)
    duration = re.search(r"(?:反复(?:发作)?|持续)?\s*\d+(?:个)?(?:天|周|月|年)(?:余|多|左右)?", source)
    result = "、".join(found[:6])
    if duration:
        result += ("，" if result else "") + duration.group().strip()
    return result


def dialogue_signals(dialogue: list[dict[str, Any]]) -> set[str]:
    doctor = " ".join(
        clean_short(turn.get("content"), 300)
        for turn in dialogue
        if turn.get("speaker") == "医生"
    )
    signals = set()
    for label, pattern in {
        "trigger": r"诱因|加重|遇热|日晒|饮食|情绪",
        "sensation": r"痒|痛|烫|灼|渗|肿",
        "medicine": r"用药|内服|外用|异常反应|过敏",
        "daily": r"吃饭|食纳|大小便|睡眠",
        "liver_kidney": r"肝功|肾功|谷丙|谷草|肌酐|胆红素",
    }.items():
        if re.search(pattern, doctor):
            signals.add(label)
    return signals


def choose_images(record: dict[str, Any], per_source: int) -> list[str]:
    urls: list[str] = []
    for field in ("tongue_face_img", "admin_face_img"):
        added = 0
        for item in record.get(field) or []:
            url = item.get("img") if isinstance(item, dict) else item
            if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in urls:
                urls.append(url)
                added += 1
                if added >= per_source:
                    break
    return urls


def image_name(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    return hashlib.sha256(url.encode()).hexdigest()[:24] + suffix


def is_valid_local_image(path: Path) -> bool:
    return is_complete_image(path)


def patient_split(patient_key: str, seed: str) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{patient_key}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 85:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def patient_components(
    best_dialogue: dict[str, dict[str, Any]],
    record_by_id: dict[str, dict[str, Any]],
    per_source: int,
) -> dict[str, str]:
    """Group patients connected by a reused image to prevent cross-split leakage."""
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owner: dict[str, str] = {}
    for rid in best_dialogue:
        record = record_by_id.get(rid)
        if not record:
            continue
        patient = str(record.get("patient_id") or record.get("user_info_id") or rid)
        find(patient)
        for url in choose_images(record, per_source):
            if url in owner:
                union(patient, owner[url])
            else:
                owner[url] = patient
    groups: dict[str, list[str]] = {}
    for patient in parent:
        groups.setdefault(find(patient), []).append(patient)
    component_key = {member: min(members) for members in groups.values() for member in members}
    return component_key


def build_answer(record: dict[str, Any], record_text: str, signals: set[str]) -> str:
    age = safe_age(record.get("patient_age"))
    sex = clean_short(record.get("patient_sex"), 4)
    appeal = symptom_summary(record.get("doc_ass_stu_appeal") or record.get("patient_appeal"))
    history_blob = " ".join(
        text(record.get(key))
        for key in (
            "new_medical_history",
            "old_medical_history",
            "allergic_history",
            "special_history",
        )
    )
    tongue = find_tongue_description(record.get("new_medical_history"), record_text)
    observed = []
    if tongue:
        observed.append(f"舌面可先记录为“{tongue}”")
    if appeal:
        observed.append(f"患处结合主诉重点记录“{appeal}”")
    if not observed:
        observed.append("请在自然光下补拍清晰的舌面、患处全景和近景")

    high_risk = bool(DANGER_RE.search(history_blob + " " + appeal))
    contraindication = []
    if age is not None and age < 18:
        contraindication.append("这是儿童，请监护人先补充准确年龄、体重、过敏史、基础病和目前用药")
    if PREGNANCY_RE.search(history_blob) or (sex == "女" and age is not None and 12 <= age <= 55):
        contraindication.append("请先确认是否怀孕、备孕或哺乳")
    if LIVER_KIDNEY_RE.search(history_blob) or "liver_kidney" in signals:
        contraindication.append("请先提供近期肝肾功能结果、异常指标及目前全部用药")

    questions = ["症状何时开始，范围和程度是在加重、减轻还是反复"]
    if "trigger" in signals or appeal:
        questions.append("是否受日晒、冷热、饮食、护肤品或情绪影响")
    questions.append("是否伴瘙痒、疼痛、灼热、肿胀、渗液或破溃")
    if "medicine" in signals:
        questions.append("近期用过哪些口服或外用药，效果如何，有无不适或过敏")
    else:
        questions.append("请补充药物/食物过敏史、基础病和目前正在使用的药物")
    if "daily" in signals:
        questions.append("食欲、睡眠及大小便近期是否有明显变化")

    lines = [
        "我先做预问诊记录：" + "；".join(observed) + "。图片受光线、角度和美颜影响，不能仅凭图片判断病因或确诊。",
    ]
    if contraindication:
        lines.append("先确认禁忌信息：" + "；".join(contraindication) + "。")
    if high_risk:
        lines.append("你提供的信息可能包含危险信号，请不要继续等待线上回复，立即前往急诊或呼叫急救。")
    else:
        lines.append("请补充：" + "；".join(questions[:4]) + "。")
        lines.append(
            "若有胸痛、呼吸困难、意识异常、高热不退、严重过敏或症状快速加重，请立即线下就医或急诊。"
            "其余情况可携带既往检查结果预约皮肤科/相应专科面诊；我不提供最终诊断或处方。"
        )
    return "\n".join(lines)


def add_red_flag_augmentations(train_samples: list[dict[str, Any]]) -> int:
    """Add image-present cases where a reported emergency must override image analysis."""
    scenarios = (
        ("我现在突发持续胸痛，还在出冷汗。", "持续胸痛伴冷汗"),
        ("我现在呼吸困难，感觉喘不过气。", "呼吸困难"),
        ("家人说我刚才出现意识模糊、叫不醒。", "意识异常"),
        ("我已经持续高热不退，精神很差。", "高热不退"),
        ("接触过敏原后嘴唇和舌头迅速肿胀。", "严重过敏并可能影响气道"),
    )
    bases = list(train_samples[: min(10, len(train_samples))])
    for index, base in enumerate(bases):
        user = base["conversations"][0]["content"] + "\n补充情况：" + scenarios[index % len(scenarios)][0]
        signal = scenarios[index % len(scenarios)][1]
        answer = (
            f"你补充的“{signal}”属于需要立即处理的危险信号。请停止等待线上回复，立即前往急诊或呼叫120，"
            "不要自行驾车；若身边有人，请让其陪同并携带现有用药和过敏信息。此时不应因继续拍摄或分析舌面、"
            "患处图片而延误急救，我也不能在线确诊或开处方。"
        )
        train_samples.append(
            {
                "id": hashlib.sha256(f"{base['id']}:redflag:{index}".encode()).hexdigest()[:16],
                "image": base["image"].copy() if isinstance(base["image"], dict) else base["image"],
                "conversations": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return len(bases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--dialogues", type=Path, default=DEFAULT_DIALOGUES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images-per-source", type=int, default=2)
    parser.add_argument("--seed", default="20260715")
    parser.add_argument("--require-local-images", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    records = json.loads(args.records.read_text(encoding="utf-8"))
    dialogues = json.loads(args.dialogues.read_text(encoding="utf-8"))
    record_by_id = {str(item["id"]): item for item in records}

    # Multiple ASR outputs may exist for one encounter. Keep the richest transcript.
    best_dialogue: dict[str, dict[str, Any]] = {}
    for item in dialogues:
        rid = str(item.get("record_id", ""))
        turns = item.get("cleared_data", {}).get("dialogue", [])
        if rid not in best_dialogue or len(turns) > len(
            best_dialogue[rid].get("cleared_data", {}).get("dialogue", [])
        ):
            best_dialogue[rid] = item

    component_key = patient_components(
        best_dialogue, record_by_id, max(1, args.images_per_source)
    )

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    manifest: dict[str, dict[str, str]] = {}
    dropped = Counter()
    for rid in sorted(best_dialogue, key=lambda value: int(value) if value.isdigit() else value):
        record = record_by_id.get(rid)
        if not record:
            dropped["unmatched_record"] += 1
            continue
        urls = choose_images(record, max(1, args.images_per_source))
        if not urls:
            dropped["no_clinical_image"] += 1
            continue
        local_images = []
        for url in urls:
            rel = f"images/{image_name(url)}"
            manifest[url] = {"url": url, "path": rel}
            if args.require_local_images and not is_valid_local_image(args.output_dir / rel):
                continue
            local_images.append(rel)
        if not local_images:
            dropped["no_local_image"] += 1
            continue

        dialogue_item = best_dialogue[rid]
        turns = dialogue_item.get("cleared_data", {}).get("dialogue", [])
        age = safe_age(record.get("patient_age"))
        sex = clean_short(record.get("patient_sex"), 4) or "未说明"
        appeal = symptom_summary(record.get("doc_ass_stu_appeal") or record.get("patient_appeal"))
        age_label = f"{age}岁" if age is not None else "年龄未说明"
        profile = f"患者：{sex}，{age_label}，{clean_short(record.get('is_first'), 4) or '问诊'}。"
        absolute_images = [str((args.output_dir / rel).resolve()) for rel in local_images]
        if len(absolute_images) == 1:
            image_value: str | dict[str, str] = absolute_images[0]
            image_placeholders = "<image>"
        else:
            image_value = {
                f"<image_{index:02d}>": path for index, path in enumerate(absolute_images)
            }
            image_placeholders = "\n".join(image_value)
        user_content = (
            image_placeholders
            + "\n任务要求："
            + SYSTEM_PROMPT
            + "\n请实时观察这些舌面或患处图片，并进行线上预问诊信息收集。"
            + f"\n{profile}"
            + (f"\n主诉：{appeal}。" if appeal else "")
        )
        patient_key = str(record.get("patient_id") or record.get("user_info_id") or rid)
        split = patient_split(component_key.get(patient_key, patient_key), args.seed)
        sample_id = hashlib.sha256(f"wuweiping:{rid}".encode()).hexdigest()[:16]
        splits[split].append(
            {
                "id": sample_id,
                "image": image_value,
                "conversations": [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": build_answer(
                            record,
                            text(dialogue_item.get("record_text")),
                            dialogue_signals(turns),
                        ),
                    },
                ],
            }
        )

    red_flag_augmentations = add_red_flag_augmentations(splits["train"])
    for split, samples in splits.items():
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    with (args.output_dir / "image_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in sorted(manifest.values(), key=lambda value: value["path"]):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output_dir / "download_images.curl.conf").open("w", encoding="utf-8") as handle:
        handle.write(
            "create-dirs\nparallel\nparallel-max = 32\nno-clobber\n"
            "retry = 3\nconnect-timeout = 20\n"
        )
        for item in sorted(manifest.values(), key=lambda value: value["path"]):
            handle.write(f'url = "{item["url"]}"\noutput = "{item["path"]}"\n')

    stats = {
        "source_records": len(records),
        "source_dialogue_rows": len(dialogues),
        "unique_dialogue_records": len(best_dialogue),
        "samples": {key: len(value) for key, value in splits.items()},
        "patients": len(
            {
                str(record_by_id[rid].get("patient_id") or record_by_id[rid].get("user_info_id") or rid)
                for rid in best_dialogue
                if rid in record_by_id and choose_images(record_by_id[rid], max(1, args.images_per_source))
            }
        ),
        "unique_images": len(manifest),
        "red_flag_augmentations": red_flag_augmentations,
        "dropped": dict(dropped),
        "split_policy": "patient-and-shared-image component deterministic 85/10/5",
        "weak_label_warning": "Image findings are encounter-linked weak labels and require clinical review.",
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
