#!/usr/bin/env python3
"""Build MiniCPM-o 4.5 LLaMA-Factory SFT data for TCM dermatology consultation.

The builder joins cleaned doctor-patient transcripts with the original encounter
records. Uploaded tongue/face/lesion images are placed in the first user turn so
the assistant learns to keep visual observations in the consultation context.
Direct identifiers are removed and obviously prescriptive doctor tails can be
trimmed to keep the dataset focused on observation and slot-filling inquiry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVER_PROJECT_ROOT = Path("/data/home/opsadmin/txd/MiniCPM-V")
DEFAULT_DIALOGUES = PROJECT_ROOT / "outputs/medical_sft_minicpmo/cleaned.json"
DEFAULT_RECORDS = Path(
    "/Users/tangxueduo/Projects/LLaMA-Factory/medical/data/zongyuan_wuweiping/"
    "wuweiping_record_20260525.json"
)
DEFAULT_MANIFESTS = (
    PROJECT_ROOT / "data/wuweiping_vlm_pretriage/image_manifest.private.json",
)

SYSTEM_PROMPT = (
    "你是中医皮肤科线上问诊助手，服务于医生的预问诊和复诊随访。"
    "你需要先客观观察患者上传的舌面、面部或患处图片，把可见表现保留为本轮对话上下文，"
    "再按槽位填充原则收集患者信息。图片观察只能描述可见现象，不能仅凭图片确诊。"
    "初诊重点建立主诉、起病时间、诱因、部位、症状性质和程度、变化趋势、伴随症状、"
    "既往诊疗、过敏史、既往史、当前用药、家族史、生活作息及女性孕哺/月经等信息。"
    "复诊重点比较上次后变化、治疗执行情况、疗效、不良反应、新症状、复查指标、"
    "过敏/孕哺/基础病和当前用药变化，不要机械重复已明确且无变化的信息。"
    "遇到胸痛、呼吸困难、意识异常、高热不退、严重过敏、症状快速加重等危险信号，"
    "应建议及时线下就医或急诊。不要输出具体处方、剂量或替代医生的最终诊断。"
)

FIRST_VISIT_POLICY = (
    "初诊槽位：主诉与持续时间、起病诱因、患处部位与范围、红斑/丘疹/脓疱/结节/瘙痒/"
    "疼痛/发烫/渗液等症状、加重缓解因素、既往诊疗与效果、过敏史、既往史、当前用药、"
    "家族史、饮食睡眠二便、女性月经/孕哺情况、儿童年龄体重。"
)
FOLLOWUP_VISIT_POLICY = (
    "复诊槽位：上次后皮损变化、红痒痛烫等症状变化、是否按医嘱执行、疗效、"
    "不良反应、是否新增症状或危险信号、复查报告/舌面/患处变化、当前用药、过敏/"
    "孕哺/基础病变化。"
)

DIRECT_ID_RE = re.compile(
    r"(?<!\d)(?:1\d{10}|\d{17}[\dXx]|\d{15})(?!\d)|"
    r"(?:姓名|身份证(?:号)?|手机号|电话|住址|地址|就诊号|住院号|订单号|患者号)"
    r"\s*[:：]?\s*[^，。；\n]{2,40}"
)
INTENT_TAG_RE = re.compile(r"(?:\[\s*意图\s*[:：]?[^\]\n]*\]|【\s*意图[^】\n]*】)")
PRESCRIPTIVE_RE = re.compile(
    r"给你开|我先给.*开|处方|(?:每日|每天|一天|一日)[一二两三四五六七八九十\d]+次|"
    r"每次\s*\d|(?:服用|口服).{0,12}(?:毫升|mg|毫克|克|片|粒|袋|丸)|"
    r"(?:外用|涂).{0,12}(?:每日|每天|一天|一日|早晚)|治疗方案|疗程"
)
TREATMENT_PLAN_RE = re.compile(
    r"方案|建议用|湿敷|外用药|口服药|用药|停用|暂停|继续使用|不能吃|不要吃|"
    r"饮食要注意|治疗|疗效|起效|复诊通过|按时复诊|联系医助"
)
DIAGNOSIS_RE = re.compile(r"(?:确诊为|诊断为|辨证为|考虑为)\s*[^，。；\n]{1,30}")
QUESTION_RE = re.compile(r"[？?]|吗|呢|是否|有没有|有无|多久|什么时候|什么|哪里|哪边|怎么样|如何")
TONGUE_RE = re.compile(
    r"(舌(?:质)?(?:淡红|淡白|淡|红|暗红|紫暗|胖大|有齿痕|齿痕|瘦薄|裂纹){1,4}"
    r"(?:[，,、 ]*苔(?:薄白|白|黄|黄腻|白腻|少|剥脱|厚腻|薄黄|薄))?)"
)
SYMPTOMS = (
    "潮红", "红斑", "丘疹", "脓疱", "脓包", "结节", "囊肿", "瘙痒", "发痒", "发烫",
    "灼热", "疼痛", "肿胀", "脱屑", "干燥", "渗液", "破溃", "色沉", "色斑", "痘印", "痘坑", "脱发",
)


def read_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(value: Any, limit: int = 800) -> str:
    result = INTENT_TAG_RE.sub("", str(value or ""))
    result = DIRECT_ID_RE.sub("[已脱敏]", result)
    result = DIAGNOSIS_RE.sub("", result)
    result = re.sub(r"\s+", " ", result).strip(" ，,。；;")
    return result[:limit]


def clean_profile_text(value: Any, limit: int = 160) -> str:
    result = clean_text(value, limit)
    if result in {"", "否", "无", "无。", "无明显", "暂无", "未填写", "不详", "未知"}:
        return ""
    return result


def safe_age(value: Any) -> int | None:
    match = re.search(r"\d{1,3}", str(value or ""))
    if not match:
        return None
    age = int(match.group())
    return age if 0 <= age <= 120 else None


def normalize_visit_type(value: Any) -> str:
    return "复诊" if str(value or "").strip() in {"复诊", "复查", "随访"} else "初诊"


def symptom_summary(*values: Any) -> str:
    source = clean_text(" ".join(str(value or "") for value in values), 240)
    found: list[str] = []
    for symptom in SYMPTOMS:
        normalized = "脓疱" if symptom == "脓包" else symptom
        if symptom in source and normalized not in found:
            found.append(normalized)
    duration = re.search(r"(?:反复(?:发作)?|持续)?\s*\d+(?:个)?(?:天|周|月|年)(?:余|多|左右)?", source)
    result = "、".join(found[:8])
    if duration:
        result += ("，" if result else "") + duration.group().strip()
    return result


def tongue_summary(*values: Any) -> str:
    matches: list[str] = []
    for value in values:
        matches.extend(TONGUE_RE.findall(str(value or "")))
    return clean_text(matches[-1], 80) if matches else ""


def load_manifest(paths: Iterable[Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in paths:
        path = path.expanduser()
        if not path.is_file():
            continue
        root = path.parent
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                url = item.get("url")
                normalized = item.get("normalized_path")
                local = normalized if normalized and (root / str(normalized)).is_file() else item.get("local_path")
                if url and local:
                    mapping[str(url)] = (root / str(local)).resolve()
        elif isinstance(data, dict):
            for url, value in data.items():
                if isinstance(value, dict):
                    local = value.get("redacted_path") or value.get("local_path") or value.get("path")
                else:
                    local = value
                if local:
                    mapping[str(url)] = (root / str(local)).resolve()
    return mapping


def iter_urls(record: dict[str, Any], fields: list[str], per_field: int) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for field in fields:
        added = 0
        raw = record.get(field) or []
        if isinstance(raw, str):
            raw = [{"img": raw}]
        for item in raw:
            url = item.get("img") if isinstance(item, dict) else item
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            if url in seen:
                continue
            seen.add(url)
            result.append((field, url))
            added += 1
            if added >= per_field:
                break
    return result


def json_image_path(path: Path, local_root: Path, json_root: Path | None) -> str:
    resolved = path.resolve()
    if json_root is None:
        return str(resolved)
    try:
        rel = resolved.relative_to(local_root.resolve())
    except ValueError:
        return str(resolved)
    return str((json_root / rel).as_posix())


def patient_split(patient_key: str, seed: str) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{patient_key}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 85:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def history_item(record: dict[str, Any], label: str, value_key: str, flag_key: str | None = None) -> str:
    value = clean_profile_text(record.get(value_key))
    flag = clean_text(record.get(flag_key), 20) if flag_key else ""
    if not value and flag in {"否", "无"}:
        value = "无"
    elif not value:
        value = "未说明"
    return f"{label}：{value}。"


def build_visual_context(record: dict[str, Any], dialogue_item: dict[str, Any], image_fields: list[str]) -> str:
    visit_type = normalize_visit_type(record.get("is_first") or dialogue_item.get("is_first"))
    sex = clean_text(record.get("patient_sex"), 8) or "性别未说明"
    age = safe_age(record.get("patient_age"))
    age_text = f"{age}岁" if age is not None else "年龄未说明"
    appeal = symptom_summary(record.get("doc_ass_stu_appeal"), record.get("patient_appeal"))
    tongue = tongue_summary(
        record.get("new_medical_history"),
        record.get("old_medical_history"),
        dialogue_item.get("record_text"),
    )
    image_kind = "、".join(image_fields)
    if visit_type == "复诊":
        parts = [
            f"患者基础信息：{sex}，{age_text}。",
            history_item(record, "既往史", "old_medical_history", "is_old_medical_history"),
            history_item(record, "家族史", "family_history", "is_family_history"),
            history_item(record, "个人史", "personal_history", "is_personal_history"),
            history_item(record, "过敏史", "allergic_history", "is_allergic_history"),
        ]
    else:
        parts = [f"患者基础信息：{sex}，{age_text}。"]
    parts.append(f"就诊类型：{visit_type}。")
    parts.append(f"本轮已上传图片字段：{image_kind}。")
    if appeal:
        parts.append(f"主诉/患处线索：{appeal}。")
    if tongue:
        parts.append(f"既有舌象文本线索：{tongue}。")
    policy = FOLLOWUP_VISIT_POLICY if visit_type == "复诊" else FIRST_VISIT_POLICY
    parts.append(f"问诊策略：{policy}")
    return "\n".join(parts)


def visual_summary(record: dict[str, Any], dialogue_item: dict[str, Any], image_count: int) -> str:
    appeal = symptom_summary(record.get("doc_ass_stu_appeal"), record.get("patient_appeal"))
    tongue = tongue_summary(
        record.get("new_medical_history"),
        record.get("old_medical_history"),
        dialogue_item.get("record_text"),
    )
    observations = []
    if image_count:
        observations.append(f"我先把你上传的{image_count}张图片作为本轮问诊参考，记录舌面、面部或患处的可见表现")
    if appeal:
        observations.append(f"患处结合主诉先记录为“{appeal}”")
    if tongue:
        observations.append(f"舌面线索先记录为“{tongue}”")
    if not observations:
        observations.append("我先把图片质量、舌面和患处可见表现记录下来")
    return "；".join(observations) + "。这些只作为问诊上下文，不能单凭图片确诊。"


def convert_turns(raw_turns: list[dict[str, Any]], max_turns: int, stop_at_treatment_plan: bool) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for raw in raw_turns:
        speaker = raw.get("speaker")
        role = "assistant" if speaker == "医生" else "user" if speaker in {"患者", "病人", "用户", "家属"} else ""
        content = clean_text(raw.get("content"), 900)
        if not role or not content:
            continue
        if role == "assistant" and stop_at_treatment_plan:
            if PRESCRIPTIVE_RE.search(content) or TREATMENT_PLAN_RE.search(content):
                break
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})
        if len(messages) >= max_turns:
            break
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages


def build_sample(
    dialogue_item: dict[str, Any],
    record: dict[str, Any],
    image_paths: list[str],
    image_fields: list[str],
    max_turns: int,
    stop_at_treatment_plan: bool,
) -> dict[str, Any] | None:
    raw_turns = dialogue_item.get("cleared_data", {}).get("dialogue", [])
    if not isinstance(raw_turns, list) or not raw_turns:
        return None
    turns = convert_turns(raw_turns, max_turns, stop_at_treatment_plan)
    if not turns:
        return None
    first_user = {
        "role": "user",
        "content": "\n".join(["<image>"] * len(image_paths) + [build_visual_context(record, dialogue_item, image_fields), "请先完成图片观察记录，并开始本次问诊。"]),
    }
    first_assistant_content = visual_summary(record, dialogue_item, len(image_paths))
    if turns[0]["role"] == "assistant":
        first_assistant_content += "\n" + turns[0]["content"]
        tail = turns[1:]
    else:
        tail = turns
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        first_user,
        {"role": "assistant", "content": first_assistant_content},
    ]
    messages.extend(tail)
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if len(messages) < 3:
        return None
    rid = str(dialogue_item.get("record_id") or record.get("id"))
    sample_id = hashlib.sha256(f"minicpmo45:tcm:{rid}".encode()).hexdigest()[:20]
    return {
        "id": sample_id,
        "messages": messages,
        "images": image_paths,
        "metadata": {
            "record_id": rid,
            "visit_type": normalize_visit_type(record.get("is_first") or dialogue_item.get("is_first")),
            "image_fields": image_fields,
            "source": "outputs/medical_sft_minicpmo/cleaned.json + wuweiping_record_20260525.json",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", type=Path, default=DEFAULT_DIALOGUES)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "web_demos/minicpm-o4.5/data")
    parser.add_argument("--manifest", type=Path, action="append", default=list(DEFAULT_MANIFESTS))
    parser.add_argument("--image-field", action="append")
    parser.add_argument("--images-per-field", type=int, default=2)
    parser.add_argument("--json-path-root", type=Path, default=SERVER_PROJECT_ROOT)
    parser.add_argument("--max-dialogue-turns", type=int, default=64)
    parser.add_argument("--seed", default="20260719")
    parser.add_argument("--keep-treatment-plan", action="store_true")
    args = parser.parse_args()
    if not args.image_field:
        args.image_field = ["admin_face_img", "tongue_face_img"]

    dialogues = read_json(args.dialogues)
    records = read_json(args.records)
    record_by_id = {str(item.get("id")): item for item in records if isinstance(item, dict)}
    manifest = load_manifest(args.manifest)

    best_dialogue: dict[str, dict[str, Any]] = {}
    for item in dialogues:
        rid = str(item.get("record_id") or "")
        turns = item.get("cleared_data", {}).get("dialogue", [])
        if not rid:
            continue
        if rid not in best_dialogue or len(turns) > len(best_dialogue[rid].get("cleared_data", {}).get("dialogue", [])):
            best_dialogue[rid] = item

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    counts: Counter[str] = Counter()
    missing_images: dict[str, dict[str, str]] = {}
    for rid in sorted(best_dialogue, key=lambda value: int(value) if value.isdigit() else value):
        dialogue_item = best_dialogue[rid]
        record = record_by_id.get(rid)
        if not record:
            counts["unmatched_record"] += 1
            continue
        selected_urls = iter_urls(record, args.image_field, max(1, args.images_per_field))
        if not selected_urls:
            counts["no_source_image"] += 1
            continue
        image_paths: list[str] = []
        image_fields: list[str] = []
        for field, url in selected_urls:
            local = manifest.get(url)
            if not local or not local.is_file():
                counts[f"missing_local_{field}"] += 1
                missing_images[url] = {"url": url, "field": field}
                continue
            image_paths.append(json_image_path(local, PROJECT_ROOT, args.json_path_root))
            if field not in image_fields:
                image_fields.append(field)
        if not image_paths:
            counts["no_local_image"] += 1
            continue
        sample = build_sample(
            dialogue_item,
            record,
            image_paths,
            image_fields,
            max(4, args.max_dialogue_turns),
            not args.keep_treatment_plan,
        )
        if not sample:
            counts["invalid_dialogue"] += 1
            continue
        patient_key = str(record.get("patient_id") or record.get("user_info_id") or rid)
        splits[patient_split(patient_key, args.seed)].append(sample)
        counts["accepted"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in splits.items():
        write_json(args.output_dir / f"tcm_o45_{split}.json", samples)
    report = {
        "dialogues": str(args.dialogues),
        "records": str(args.records),
        "output_dir": str(args.output_dir.resolve()),
        "image_fields": args.image_field,
        "images_per_field": args.images_per_field,
        "json_path_root": str(args.json_path_root) if args.json_path_root else "",
        "max_dialogue_turns": args.max_dialogue_turns,
        "stop_at_treatment_plan": not args.keep_treatment_plan,
        "splits": {key: len(value) for key, value in splits.items()},
        "counts": dict(counts),
        "missing_unique_images": len(missing_images),
        "warning": "Image observations are weak encounter-linked supervision and should be reviewed by clinicians before production fine-tuning.",
    }
    write_json(args.output_dir / "tcm_o45_build_report.json", report)
    missing_manifest = args.output_dir / "missing_images.jsonl"
    with missing_manifest.open("w", encoding="utf-8") as handle:
        for item in sorted(missing_images.values(), key=lambda value: value["url"]):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
