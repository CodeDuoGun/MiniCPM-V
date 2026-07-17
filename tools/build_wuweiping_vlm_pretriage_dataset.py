#!/usr/bin/env python3
"""Build a MiniCPM multimodal pre-triage dataset with VLM-assisted labeling.

Pipeline:
  prepare  -> private image manifest + curl download configuration
  analyze  -> VLM classification for tongue/lesion images and OCR for reports
  redact   -> cover VLM-detected PII regions in report images
  verify   -> second VLM pass that rejects report images with visible PII
  build    -> separate realtime-clinical and manual-report-upload datasets

All API results are cached as JSONL and every stage is resumable. Report images
fail closed: production build excludes them unless redaction verification says
that no direct identifier remains (or an explicit development override is used).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from image_integrity import is_complete_image


DEFAULT_RECORDS = Path(
    "/Users/tangxueduo/Projects/LLaMA-Factory/medical/data/zongyuan_wuweiping/"
    "wuweiping_record_20260525.json"
)
DEFAULT_DIALOGUES = Path(
    __file__
).resolve().parents[1] / "outputs/medical_sft_minicpmo/tcm_consult_minicpmo.json"
DEFAULT_MEDICAL_ROOT = Path("/Users/tangxueduo/Projects/LLaMA-Factory")
DEFAULT_OUTPUT = Path("data/wuweiping_vlm_pretriage")

REALTIME_SYSTEM_POLICY = (
    "你是实时音视频预问诊助手，只做病史采集、舌象/面象/患处的客观描述、风险提示和就医建议。"
    "不能做最终诊断，不能替代线下医生，不能开具处方或给出具体用药方案。"
    "实时视频严禁读取、转写、概括或解释任何检查报告、处方、病历、证件和其他文档内容；"
    "看到疑似文档时，只能请用户停止展示，并通过检查报告手动上传入口提交，由独立VLM接口分析。"
    "实时图片只能用于记录舌面、面部和患处的可见信息，不能据此确诊。"
    "回答简洁、温和，先追问关键病史，再给风险提示。"
    "遇到胸痛、呼吸困难、意识障碍、高热不退、严重过敏等危险信号，建议立即就医或急诊。"
    "儿童、孕妇或哺乳期，以及肝肾功能异常者，必须先询问年龄体重、孕哺情况、过敏史、"
    "基础病、肝肾功能和正在使用的药物等禁忌信息。"
)
REPORT_UPLOAD_SYSTEM_POLICY = (
    "你是检查报告手动上传分析助手。当前图片必须来自用户主动选择文件后的独立上传入口，"
    "不得把实时视频帧当作报告输入。你只能读取报告中真实可见的项目、数值、单位、参考范围和异常标记，"
    "必须提示OCR可能有误并要求医生复核；不能确诊、不能开处方或给出具体用药方案。"
)
DIRECT_ID_RE = re.compile(
    r"(?<!\d)(?:1\d{10}|\d{17}[\dXx]|\d{15})(?!\d)|"
    r"(?:姓名|身份证(?:号)?|手机号|电话|住址|地址|就诊号|住院号)\s*[:：]?\s*[^，。；\n]{2,40}"
)
DIAGNOSIS_PHRASE_RE = re.compile(
    r"(?:考虑|诊断为|确诊为|提示为|符合)\s*[^，。；\n]{1,40}"
)
DIRECT_NUMBER_RE = re.compile(r"(?<!\d)(?:1\d{10}|\d{17}[\dXx]|\d{15})(?!\d)")
INTENT_TAG_RE = re.compile(
    r"(?:\[\s*意图\s*[:：]?[^\]\n]*\]|【\s*意图[^】\n]*】|(?<!\[)意图\s*[:：][^\]\n]*\])"
)
NON_REPORT_DOCUMENT_RE = re.compile(
    r"处方|诊断|主诉|现病史|体格检查|病历(?:号)?|就诊号|住院号|处方号|"
    r"药品(?:名称)?|用法|用量|嘱托|治疗(?:方案|记录)?|科室"
)
UNSAFE_DOCTOR_TURN_RE = re.compile(
    r"确诊|诊断为|考虑为|辨证为|处方|开药|服用|口服|外用|每日\d|一天\d|"
    r"治疗方案|疗程|加减方|剂量|mg|毫克|克/次"
)
QUESTION_LIKE_RE = re.compile(
    r"[？?]|吗|呢|么|是否|有没有|有无|多久|什么时候|从什么时候|什么原因|"
    r"哪里|哪边|哪个部位|怎样|怎么样|如何|几岁|多大|体重|严重|程度|"
    r"加重|减轻|缓解|反复|复发|变化|伴随|痒|疼|痛|烫|渗|肿|破溃|"
    r"过敏|既往|家族|用药|药物|效果|不适|睡眠|饮食|食欲|大便|小便|"
    r"月经|怀孕|备孕|哺乳|检查|报告|看下|看一下|伸舌|拍.*(?:舌|患处|皮损)"
)

FIRST_VISIT_POLICY = (
    "这是初诊。先建立主诉，再按起病时间与诱因、部位、性质和严重度、变化趋势、伴随症状、"
    "既往诊疗与效果、过敏史、既往史、当前用药、家族史和必要的生活/生育信息逐步追问；"
    "优先排除危险信号，未知信息要继续确认，信息足够后总结并请患者核对。"
)
FOLLOWUP_VISIT_POLICY = (
    "这是复诊。先比较上次就诊后的症状变化，再询问治疗执行情况、疗效和不良反应、是否出现新症状或危险信号、"
    "复查指标以及用药/过敏/孕哺等关键信息变化；不要机械重复已经明确且没有变化的完整既往史，"
    "信息足够后总结本次变化并请患者核对。"
)
SYMPTOMS = (
    "潮红", "红斑", "丘疹", "脓疱", "脓包", "结节", "瘙痒", "发痒", "发烫", "灼热",
    "疼痛", "肿胀", "脱屑", "干燥", "渗液", "破溃", "色沉", "色斑", "痘印", "痘坑", "脱发",
)
CLINICAL_TYPES = {"tongue", "lesion", "face_overview"}

CLINICAL_PROMPT = """你是医学图片标注助手，只描述真实可见内容，禁止诊断、辨证、推测病因或推荐药物。
图片来自一个同时存放舌面和患处照片的字段。请先分类，再结构化描述。看不清时使用“未知”。
只返回 JSON 对象：
{
  "image_type": "tongue|lesion|face_overview|invalid",
  "quality": "usable|limited|unusable",
  "description": "一句客观中文描述，不得出现疾病诊断",
  "tongue": {
    "body_color": "淡红|红|暗红|淡白|紫暗|未知",
    "shape": "正常|胖大|瘦薄|齿痕|裂纹|未知",
    "coating_color": "白苔|黄苔|灰黑苔|少苔|无苔|未知",
    "coating_texture": "薄苔|厚苔|腻苔|腐苔|剥脱苔|未知",
    "moisture": "润|燥|滑|未知"
  },
  "lesion": {
    "primary": ["丘疹|脓疱|结节|囊肿|红斑|斑疹|水疱|风团|鳞屑|无明显皮损|未知"],
    "secondary": ["色素沉着|色素减退|瘢痕|抓痕|结痂|糜烂|渗液|无|未知"],
    "color": "肤色|淡红|鲜红|暗红|紫红|紫暗|黄|褐|黑|未知",
    "distribution": ["额头|双颊|左颊|右颊|鼻部|鼻翼|口周|下巴|下颌|耳周|颈部|胸部|背部|四肢|手部|腿部|未知"],
    "count": "极少|少量|中等|较多|大量|未知"
  }
}
不适用的对象仍保留但填“未知”或空数组。仅输出 JSON。"""

REPORT_PROMPT = """你是医学检查报告 OCR 与隐私标注助手。只能读取图片中可见文字，不得诊断、推测病因或推荐药物。
必须定位所有直接身份信息：姓名、身份证号、手机号、地址、就诊号、住院号、条码、二维码及其他编号。
bbox 使用 [x1,y1,x2,y2]，坐标归一化到 0-1000，覆盖完整敏感区域。若看不清是否含隐私，pii_status=uncertain。
不要在任何文本字段中复述姓名、证件号、手机号、地址或就诊编号。只返回 JSON：
{
  "image_type": "report|invalid",
  "quality": "usable|limited|unusable",
  "report_type": "血常规/尿常规/肝功能/肾功能/血糖/血脂/甲状腺功能/影像检查/病理/其他/未知",
  "report_date": "YYYY-MM-DD或未知",
  "items": [
    {"name":"项目名","value":"结果","unit":"单位或无","reference":"参考范围或无","flag":"高|低|正常|异常|未知"}
  ],
  "summary": "仅概括报告类型、可见项目和异常箭头，不得含身份信息或疾病诊断",
  "pii_status": "detected|clear|uncertain",
  "pii_regions": [
    {"kind":"name|id_number|phone|address|visit_id|barcode|qrcode|other","bbox":[0,0,1000,1000]}
  ]
}
最多返回30个最清晰项目。仅输出 JSON。"""

VERIFY_PROMPT = """检查这张已经遮挡处理的医学报告图片，判断是否仍能看到任何直接身份信息，包括姓名、身份证号、
手机号、地址、就诊号、住院号、条码、二维码或可关联患者的编号。不要复述看到的内容。
只返回 JSON：{"pii_clear":true,"visible_pii_types":[],"reason":"简短说明"}。
只要疑似仍可识别，pii_clear 必须为 false。"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(value: Any, limit: int = 240) -> str:
    result = INTENT_TAG_RE.sub("", str(value or ""))
    result = DIRECT_ID_RE.sub("[已脱敏]", result)
    result = DIAGNOSIS_PHRASE_RE.sub("", result)
    result = re.sub(r"\s+", " ", result).strip(" ，,。；;")
    return result[:limit]


def dialogue_pairs(raw_dialogue: Any) -> list[dict[str, str]]:
    """Extract safe doctor-question/patient-answer pairs from an encounter."""

    if not isinstance(raw_dialogue, list):
        return []
    pairs: list[dict[str, str]] = []
    for index, turn in enumerate(raw_dialogue[:-1]):
        if not isinstance(turn, dict) or turn.get("speaker") != "医生":
            continue
        next_turn = raw_dialogue[index + 1]
        if not isinstance(next_turn, dict) or next_turn.get("speaker") not in {"患者", "病人", "用户", "家属"}:
            continue
        question = clean_text(turn.get("content"), 240)
        answer = clean_text(next_turn.get("content"), 320)
        if not question or not answer:
            continue
        if UNSAFE_DOCTOR_TURN_RE.search(question) or not QUESTION_LIKE_RE.search(question):
            continue
        pairs.append({"question": question, "answer": answer})
    return pairs


def minicpmo_dialogue_pairs(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Extract next-question supervision from the designated MiniCPM dataset."""

    conversations = sample.get("conversations") or []
    pairs: list[dict[str, str]] = []
    for index, message in enumerate(conversations):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        question = clean_text(message.get("content"), 240)
        if not question or INTENT_TAG_RE.search(question):
            continue
        if UNSAFE_DOCTOR_TURN_RE.search(question) or not QUESTION_LIKE_RE.search(question):
            continue
        answer = ""
        if index + 1 < len(conversations) and conversations[index + 1].get("role") == "user":
            answer = clean_text(conversations[index + 1].get("content"), 320)
        # A final unanswered doctor question is still a valid next-question target.
        if answer or index == len(conversations) - 1:
            pairs.append({"question": question, "answer": answer})
    return pairs


def minicpmo_visit_type(sample: dict[str, Any]) -> str:
    conversations = sample.get("conversations") or []
    first_content = conversations[0].get("content", "") if conversations else ""
    match = re.search(r"就诊类型\s*[:：]\s*(初诊|复诊)", first_content)
    return match.group(1) if match else ""


def visit_policy(record: dict[str, Any]) -> str:
    return FOLLOWUP_VISIT_POLICY if record.get("is_first") == "复诊" else FIRST_VISIT_POLICY


def symptom_summary(value: Any) -> str:
    source = clean_text(value, 160)
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


def safe_age(value: Any) -> int | None:
    match = re.search(r"\d{1,3}", str(value or ""))
    if not match:
        return None
    age = int(match.group())
    return age if 0 <= age <= 120 else None


def hash_id(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def image_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"


def load_jsonl_index(path: Path, key: str = "image_id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get(key):
            result[str(item[key])] = item
    return result


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def extract_json_payload(response_text: str) -> dict[str, Any]:
    text = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("VLM response does not contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("VLM response JSON must be an object")
    return value


def build_image_url(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://", "data:image/")):
        return path_or_url
    path = Path(path_or_url)
    if not path.is_file():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def make_client(args: argparse.Namespace) -> tuple[Any, str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 包，请在 medical 环境中安装/启用 openai Python SDK") from exc

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    base_url = args.base_url or os.getenv("DASHSCOPE_BASE_URL")
    model = args.model
    if not api_key or not base_url or not model:
        medical_root = Path(args.medical_root).resolve()
        sys.path.insert(0, str(medical_root))
        previous_cwd = Path.cwd()
        try:
            # Dynaconf resolves ``medical/config.yaml`` against the process CWD.
            # Load it from the LLaMA-Factory root, then immediately restore CWD.
            os.chdir(medical_root)
            config_module = importlib.import_module("medical.config")
            config = config_module.config
            # Dynaconf is lazy; resolve values before restoring the original CWD.
            config_api_key = getattr(config, "DASHSCOPE_API_KEY", "")
            config_base_url = getattr(config, "DASHSCOPE_BASE_URL", "")
            config_model = getattr(config, "QWEN_IMAGE_DESCRIBE_MODEL", "")
        except Exception as exc:
            raise RuntimeError(f"无法加载 {medical_root}/medical/config.py") from exc
        finally:
            os.chdir(previous_cwd)
        api_key = api_key or config_api_key
        base_url = base_url or config_base_url
        model = model or config_model
    if not api_key or not base_url or not model:
        raise RuntimeError("DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL 或 VLM model 未配置")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=(10.0, args.timeout)), str(model)


def vlm_call(client: Any, model: str, image: str, prompt: str, attempts: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": build_image_url(image)}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return extract_json_payload(response.choices[0].message.content or "{}")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(30.0, 2.0 ** attempt + random.random()))
    raise RuntimeError(f"VLM call failed after {attempts} attempts: {last_error}")


def normalize_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [max(0, min(1000, int(float(part)))) for part in value]
    except (TypeError, ValueError):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def normalize_analysis(raw: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_id": entry["image_id"],
        "source_field": entry["source_field"],
        "model_output_version": 1,
    }
    if entry["source_field"] == "tongue_face_img":
        image_type = raw.get("image_type") if raw.get("image_type") in CLINICAL_TYPES | {"invalid"} else "invalid"
        description = clean_text(raw.get("description") or "图像特征不清，无法可靠描述。", 180)
        if re.search(r"诊断|考虑|疑似|符合|癌|痤疮|湿疹|皮炎|银屑病", description):
            description = "仅记录到可见的舌面或皮肤外观特征，具体细节需人工复核。"
        result.update(
            {
                "image_type": image_type,
                "quality": raw.get("quality") if raw.get("quality") in {"usable", "limited", "unusable"} else "limited",
                "description": description,
                "tongue": raw.get("tongue") if isinstance(raw.get("tongue"), dict) else {},
                "lesion": raw.get("lesion") if isinstance(raw.get("lesion"), dict) else {},
            }
        )
        return result

    regions = []
    for region in raw.get("pii_regions") or []:
        if not isinstance(region, dict):
            continue
        bbox = normalize_bbox(region.get("bbox"))
        if bbox:
            regions.append({"kind": clean_text(region.get("kind") or "other", 30), "bbox": bbox})
    items = []
    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_name = clean_text(item.get("name"), 60)
        if re.search(r"姓名|患者|身份证|电话|手机|地址|就诊|住院|条码|二维码", item_name):
            continue
        items.append(
            {
                "name": item_name,
                "value": clean_text(item.get("value"), 40),
                "unit": clean_text(item.get("unit") or "无", 30),
                "reference": clean_text(item.get("reference") or "无", 50),
                "flag": item.get("flag") if item.get("flag") in {"高", "低", "正常", "异常", "未知"} else "未知",
            }
        )
    report_type = clean_text(raw.get("report_type") or "未知", 50)
    abnormal_names = [item["name"] for item in items if item["flag"] in {"高", "低", "异常"}]
    safe_summary = f"{report_type}报告，可读取{len(items)}个项目"
    if abnormal_names:
        safe_summary += "；带异常标记的项目包括" + "、".join(abnormal_names[:8])
    result.update(
        {
            "image_type": "report" if raw.get("image_type") == "report" else "invalid",
            "quality": raw.get("quality") if raw.get("quality") in {"usable", "limited", "unusable"} else "limited",
            "report_type": report_type,
            "report_date": clean_text(raw.get("report_date") or "未知", 20),
            "items": items[:30],
            "summary": safe_summary,
            "pii_status": raw.get("pii_status") if raw.get("pii_status") in {"detected", "clear", "uncertain"} else "uncertain",
            "pii_regions": regions,
        }
    )
    return result


def command_prepare(args: argparse.Namespace) -> None:
    records = read_json(args.records)
    dialogues = read_json(args.dialogues)
    best_dialogues: dict[str, dict[str, Any]] = {}
    for item in dialogues:
        rid = str(item.get("id"))
        pairs = minicpmo_dialogue_pairs(item)
        current_pairs = best_dialogues.get(rid, {}).get("pairs", [])
        if len(pairs) > len(current_pairs):
            best_dialogues[rid] = {
                "pairs": pairs,
                "visit_type": minicpmo_visit_type(item),
            }
    matched_ids = set(best_dialogues)
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    record_rows = []
    for record in records:
        rid = str(record.get("id"))
        if rid not in matched_ids:
            continue
        record_rows.append(
            {
                "record_id": rid,
                "patient_key": str(record.get("patient_id") or record.get("user_info_id") or rid),
                "sex": clean_text(record.get("patient_sex"), 4),
                "age": safe_age(record.get("patient_age")),
                "is_first": best_dialogues.get(rid, {}).get("visit_type") or clean_text(record.get("is_first"), 8),
                "symptoms": symptom_summary(record.get("doc_ass_stu_appeal") or record.get("patient_appeal")),
                "dialogue_pairs": best_dialogues.get(rid, {}).get("pairs", []),
                "dialogue_source": str(args.dialogues.resolve()),
            }
        )
        for field in ("tongue_face_img", "admin_report_img"):
            for order, image in enumerate(record.get(field) or []):
                url = image.get("img") if isinstance(image, dict) else image
                if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    continue
                key = (url, field)
                if key not in entries:
                    image_id = hash_id(f"{field}:{url}")
                    subdir = "clinical_original" if field == "tongue_face_img" else "report_original"
                    entries[key] = {
                        "image_id": image_id,
                        "source_field": field,
                        "url": url,
                        "local_path": f"{subdir}/{image_id}{image_suffix(url)}",
                        "redacted_path": f"report_redacted/{image_id}.png" if field == "admin_report_img" else None,
                        "occurrences": [],
                    }
                entries[key]["occurrences"].append(
                    {"record_id": rid, "order": order, "time": clean_text(image.get("time") if isinstance(image, dict) else "", 30)}
                )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = sorted(entries.values(), key=lambda item: (item["source_field"], item["image_id"]))
    write_json(output / "image_manifest.private.json", manifest)
    write_json(output / "record_index.private.json", record_rows)
    with (output / "download_images.curl.conf").open("w", encoding="utf-8") as handle:
        handle.write("create-dirs\nparallel\nparallel-max = 16\nno-clobber\nretry = 4\nconnect-timeout = 20\n")
        for item in manifest:
            handle.write(f'url = "{item["url"]}"\noutput = "{item["local_path"]}"\n')
    stats = Counter(item["source_field"] for item in manifest)
    print(json.dumps({
        "records": len(record_rows),
        "dialogue_source": str(args.dialogues),
        "records_with_dialogue_pairs": sum(bool(item["dialogue_pairs"]) for item in record_rows),
        "dialogue_pairs": sum(len(item["dialogue_pairs"]) for item in record_rows),
        "unique_images": len(manifest),
        "by_field": stats,
    }, ensure_ascii=False, default=dict, indent=2))


def command_download(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    config_path = output / "download_images.curl.conf"
    if not config_path.is_file():
        raise SystemExit("请先运行 prepare")
    if args.max_images or args.source_field:
        manifest = read_json(output / "image_manifest.private.json")
        selected = [item for item in manifest if not args.source_field or item["source_field"] == args.source_field]
        if args.max_images:
            selected = selected[: args.max_images]
        for item in selected:
            target = output / item["local_path"]
            if is_complete_image(target):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "20", "--output", str(target), item["url"]],
                check=True,
            )
        return
    subprocess.run(["curl", "-L", "--fail", "--config", str(config_path)], cwd=output, check=True)


def run_concurrent(
    entries: list[dict[str, Any]], workers: int, worker: Callable[[dict[str, Any]], dict[str, Any]], sink: Path
) -> tuple[int, int]:
    succeeded = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(worker, item): item for item in entries}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                append_jsonl(sink, future.result())
                succeeded += 1
            except Exception as exc:
                append_jsonl(
                    sink.with_name(sink.stem + ".errors.jsonl"),
                    {"image_id": item["image_id"], "error": str(exc), "timestamp": int(time.time())},
                )
                failed += 1
            if (succeeded + failed) % 20 == 0:
                print(f"processed={succeeded + failed} succeeded={succeeded} failed={failed}", file=sys.stderr)
    return succeeded, failed


def command_analyze(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    manifest = read_json(output / "image_manifest.private.json")
    cache_path = output / "vlm_analysis.private.jsonl"
    done = load_jsonl_index(cache_path)
    pending = [item for item in manifest if item["image_id"] not in done]
    if args.source_field:
        pending = [item for item in pending if item["source_field"] == args.source_field]
    if args.max_images:
        pending = pending[: args.max_images]
    client, model = make_client(args)

    def worker(entry: dict[str, Any]) -> dict[str, Any]:
        prompt = CLINICAL_PROMPT if entry["source_field"] == "tongue_face_img" else REPORT_PROMPT
        raw = vlm_call(client, model, entry["url"], prompt, args.attempts)
        result = normalize_analysis(raw, entry)
        result["model"] = model
        return result

    succeeded, failed = run_concurrent(pending, args.workers, worker, cache_path)
    print(json.dumps({"cached": len(done), "attempted": len(pending), "succeeded": succeeded, "failed": failed}, ensure_ascii=False))


def command_redact(args: argparse.Namespace) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("redact 阶段需要 Pillow") from exc
    output = Path(args.output_dir)
    manifest = {item["image_id"]: item for item in read_json(output / "image_manifest.private.json")}
    analyses = load_jsonl_index(output / "vlm_analysis.private.jsonl")
    statuses = []
    for image_id, analysis in analyses.items():
        if analysis.get("source_field") != "admin_report_img" or analysis.get("image_type") != "report":
            continue
        entry = manifest[image_id]
        source = output / entry["local_path"]
        target = output / entry["redacted_path"]
        status = {"image_id": image_id, "redacted_path": entry["redacted_path"], "ready": False, "reason": ""}
        if not is_complete_image(source):
            status["reason"] = "missing_or_incomplete_source"
            statuses.append(status)
            continue
        pii_status = analysis.get("pii_status")
        regions = analysis.get("pii_regions") or []
        if pii_status in {"detected", "uncertain"} and not regions:
            status["reason"] = "pii_not_localized_fail_closed"
            statuses.append(status)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            # Medical reports commonly place name/IDs/barcodes in the header.
            # VLM boxes are approximate, so cover the full header as a robust
            # first layer, then cover every detected region anywhere else.
            if args.header_redact_ratio > 0:
                draw.rectangle(
                    [0, 0, width, min(height, int(height * args.header_redact_ratio))],
                    fill="black",
                )
            for region in regions:
                x1, y1, x2, y2 = region["bbox"]
                px = [int(x1 * width / 1000), int(y1 * height / 1000), int(x2 * width / 1000), int(y2 * height / 1000)]
                pad_x, pad_y = max(6, int(width * 0.01)), max(6, int(height * 0.01))
                px = [max(0, px[0] - pad_x), max(0, px[1] - pad_y), min(width, px[2] + pad_x), min(height, px[3] + pad_y)]
                draw.rectangle(px, fill="black")
            image.save(target, format="PNG", optimize=True)
        status.update(
            {
                "ready": is_complete_image(target),
                "reason": "header_and_regions_redacted" if regions else "header_redacted_vlm_reported_clear",
                "header_redact_ratio": args.header_redact_ratio,
            }
        )
        statuses.append(status)
    write_json(output / "redaction_status.private.json", statuses)
    print(json.dumps(dict(Counter(item["reason"] for item in statuses)), ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    statuses = {item["image_id"]: item for item in read_json(output / "redaction_status.private.json")}
    cache_path = output / "redaction_verification.private.jsonl"
    done = load_jsonl_index(cache_path)
    pending = [item for item in statuses.values() if item.get("ready") and item["image_id"] not in done]
    if args.max_images:
        pending = pending[: args.max_images]
    client, model = make_client(args)

    def worker(status: dict[str, Any]) -> dict[str, Any]:
        raw = vlm_call(client, model, str(output / status["redacted_path"]), VERIFY_PROMPT, args.attempts)
        return {
            "image_id": status["image_id"],
            "pii_clear": raw.get("pii_clear") is True,
            "visible_pii_types": [clean_text(v, 30) for v in (raw.get("visible_pii_types") or [])],
            "reason": clean_text(raw.get("reason"), 120),
            "model": model,
        }

    succeeded, failed = run_concurrent(pending, args.workers, worker, cache_path)
    print(json.dumps({"cached": len(done), "attempted": len(pending), "succeeded": succeeded, "failed": failed}, ensure_ascii=False))


def patient_components(records: dict[str, dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for record in records.values():
        find(record["patient_key"])
    for entry in manifest:
        patients = [records[o["record_id"]]["patient_key"] for o in entry["occurrences"] if o["record_id"] in records]
        for patient in patients[1:]:
            union(patients[0], patient)
    groups: dict[str, list[str]] = defaultdict(list)
    for patient in parent:
        groups[find(patient)].append(patient)
    return {member: min(members) for members in groups.values() for member in members}


def split_name(component: str, seed: str) -> str:
    bucket = int(hash_id(f"{seed}:{component}", 8), 16) % 100
    return "train" if bucket < 85 else "validation" if bucket < 95 else "test"


def native_image_value(paths: list[Path]) -> tuple[str | dict[str, str], str]:
    absolute = [str(path.resolve()) for path in paths]
    if len(absolute) == 1:
        return absolute[0], "<image>"
    value = {f"<image_{index:02d}>": path for index, path in enumerate(absolute)}
    return value, "\n".join(value)


def profile_text(record: dict[str, Any]) -> str:
    age = f"{record['age']}岁" if record.get("age") is not None else "年龄未说明"
    profile = f"患者：{record.get('sex') or '性别未说明'}，{age}，{record.get('is_first') or '问诊'}。"
    if record.get("symptoms"):
        profile += f"\n主诉：{record['symptoms']}。"
    return profile


def contraindication_questions(record: dict[str, Any]) -> list[str]:
    questions = []
    age, sex = record.get("age"), record.get("sex")
    if age is not None and age < 18:
        questions.append("这是儿童，请监护人先补充准确年龄、体重、过敏史、基础病和目前用药")
    if sex == "女" and age is not None and 12 <= age <= 55:
        questions.append("请先确认是否怀孕、备孕或哺乳")
    return questions


def clinical_answer(analyses: list[dict[str, Any]], record: dict[str, Any]) -> str:
    descriptions = [f"图{index + 1}：{clean_text(item.get('description'), 140)}" for index, item in enumerate(analyses)]
    lines = ["我先做客观记录：" + "；".join(descriptions) + "。图片受光线、角度和清晰度影响，不能据此确诊。"]
    contraindications = contraindication_questions(record)
    if contraindications:
        lines.append("先确认禁忌信息：" + "；".join(contraindications) + "。")
    lines.append("请补充：症状何时开始及变化趋势；是否瘙痒、疼痛、灼热、肿胀、渗液或破溃；是否受日晒、冷热、饮食或护肤品影响；既往病史、过敏史及目前用药。")
    lines.append("若有胸痛、呼吸困难、意识异常、高热不退、严重过敏或症状快速加重，请立即线下就医或急诊。我不提供最终诊断或处方。")
    return "\n".join(lines)


def report_answer(analyses: list[dict[str, Any]], record: dict[str, Any]) -> str:
    page_lines = []
    has_liver_kidney = False
    for index, item in enumerate(analyses):
        visible = []
        for lab in item.get("items") or []:
            if not lab.get("name"):
                continue
            flag = f"（{lab['flag']}）" if lab.get("flag") in {"高", "低", "异常"} else ""
            visible.append(f"{lab['name']} {lab.get('value') or '未读清'} {lab.get('unit') or ''}{flag}".strip())
            if re.search(r"肝|肾|谷丙|谷草|肌酐|尿素|胆红素", lab["name"]):
                has_liver_kidney = True
        content = "、".join(visible[:8]) or clean_text(item.get("summary"), 160)
        page_lines.append(f"报告图{index + 1}（{item.get('report_type') or '类型未知'}）：{content}")
    lines = ["我先按图片读取：" + "；".join(page_lines) + "。OCR可能受清晰度影响，请以报告原件和医生复核为准；这些结果不能单独用于确诊。"]
    questions = []
    if has_liver_kidney:
        questions.append("报告涉及肝肾相关指标，请先说明检查日期、既往异常、基础病及目前全部用药")
    questions.extend(["请补充这次检查的原因和当前症状", "是否有药物或食物过敏", "女性请说明是否孕期、备孕或哺乳期"])
    lines.append("请补充：" + "；".join(questions) + "。")
    lines.append("若有胸痛、呼吸困难、意识异常、高热不退或严重过敏，请立即就医或急诊；我不提供最终诊断或处方。")
    return "\n".join(lines)


def is_supported_uploaded_report(analysis: dict[str, Any]) -> bool:
    """Exclude prescriptions, clinical notes and pages carrying direct IDs."""

    for item in analysis.get("items") or []:
        name = str(item.get("name") or "")
        value = str(item.get("value") or "")
        if NON_REPORT_DOCUMENT_RE.search(name) or DIRECT_NUMBER_RE.search(value):
            return False
    return True


def clinical_observation(analyses: list[dict[str, Any]]) -> str:
    descriptions = [
        f"图{index + 1}：{clean_text(item.get('description'), 140)}"
        for index, item in enumerate(analyses)
    ]
    return (
        "我先做客观记录："
        + "；".join(descriptions)
        + "。图片受光线、角度和清晰度影响，不能据此确诊。"
    )


def realtime_multiturn_conversations(
    user_prompt: str,
    analyses: list[dict[str, Any]],
    record: dict[str, Any],
    max_dialogue_pairs: int,
) -> tuple[list[dict[str, str]], int]:
    pairs = list(record.get("dialogue_pairs") or [])[:max_dialogue_pairs]
    if not pairs:
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": clinical_answer(analyses, record)},
        ], 0

    conversations = [
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": clinical_observation(analyses) + "\n" + pairs[0]["question"],
        },
    ]
    # End on a doctor question so every patient answer supervises the next ask.
    for index in range(len(pairs) - 1):
        conversations.append({"role": "user", "content": pairs[index]["answer"]})
        conversations.append({"role": "assistant", "content": pairs[index + 1]["question"]})
    return conversations, len(pairs)


def text_dialogue_sample(record: dict[str, Any], max_dialogue_pairs: int) -> dict[str, Any] | None:
    pairs = list(record.get("dialogue_pairs") or [])[:max_dialogue_pairs]
    if not pairs:
        return None
    user_prompt = (
        f"任务要求：{REALTIME_SYSTEM_POLICY}\n"
        f"问诊策略：{visit_policy(record)}\n"
        "当前没有可用的舌象或患处图片，请仅根据患者已经回答的内容逐步追问；一次优先询问1至3个关键问题。\n"
        f"{profile_text(record)}"
    )
    conversations = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": pairs[0]["question"]},
    ]
    for index in range(len(pairs) - 1):
        conversations.append({"role": "user", "content": pairs[index]["answer"]})
        conversations.append({"role": "assistant", "content": pairs[index + 1]["question"]})
    visit_type = record.get("is_first") or "未知"
    return {
        "id": hash_id(f"{record['record_id']}:text-dialogue:{visit_type}", 16),
        "conversations": conversations,
        "metadata": {
            "visit_type": visit_type,
            "dialogue_pairs": len(pairs),
            "task": "realtime_text_consultation",
            "dialogue_source": record.get("dialogue_source"),
        },
        "_category": "dialogue_followup" if visit_type == "复诊" else "dialogue_initial",
        "_patient_key": record["patient_key"],
    }


def make_sample(
    record: dict[str, Any],
    entries: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
    paths: list[Path],
    category: str,
    max_dialogue_pairs: int,
) -> dict[str, Any]:
    image, placeholders = native_image_value(paths)
    if category.startswith("clinical"):
        user = (
            f"{placeholders}\n任务要求：{REALTIME_SYSTEM_POLICY}\n"
            f"问诊策略：{visit_policy(record)}\n"
            f"以上为实时音视频会话采集的舌面/面部/患处画面。请逐图客观记录，并继续收集预问诊信息。\n"
            f"{profile_text(record)}"
        )
        conversations, used_dialogue_pairs = realtime_multiturn_conversations(
            user, analyses, record, max_dialogue_pairs
        )
    else:
        user = (
            f"{placeholders}\n任务要求：{REPORT_UPLOAD_SYSTEM_POLICY}\n"
            f"以上检查报告由用户通过独立的检查报告手动上传入口提交，不是实时视频帧。"
            f"请读取可见项目并继续收集必要病史。\n{profile_text(record)}"
        )
        conversations = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": report_answer(analyses, record)},
        ]
        used_dialogue_pairs = 0
    signature = ":".join(entry["image_id"] for entry in entries)
    return {
        "id": hash_id(f"{record['record_id']}:{category}:{signature}", 16),
        "image": image,
        "conversations": conversations,
        "metadata": {
            "visit_type": record.get("is_first") or "未知",
            "dialogue_pairs": used_dialogue_pairs,
            "task": "realtime_visual_consultation" if category.startswith("clinical") else "manual_report_upload",
            "dialogue_source": record.get("dialogue_source") if category.startswith("clinical") else None,
        },
        "_category": category,
        "_patient_key": record["patient_key"],
    }


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def shuffled(samples: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    result = list(samples)
    random.Random(seed).shuffle(result)
    return result


def add_red_flag_samples(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return samples
    scenarios = (
        ("我现在突发持续胸痛并伴冷汗。", "持续胸痛伴冷汗"),
        ("我现在呼吸困难，感觉喘不过气。", "呼吸困难"),
        ("家人说我刚才意识模糊、很难叫醒。", "意识异常"),
        ("我已经持续高热不退，精神很差。", "高热不退"),
        ("接触过敏原后嘴唇和舌头迅速肿胀。", "严重过敏并可能影响气道"),
    )
    bases = [sample for sample in samples if sample["_category"].startswith("clinical")]
    additions = []
    for index, base in enumerate(bases[:limit]):
        scenario, signal = scenarios[index % len(scenarios)]
        copy = dict(base)
        copy["id"] = hash_id(f"{base['id']}:redflag:{index}", 16)
        copy["_category"] = "safety_red_flag"
        copy["conversations"] = [dict(turn) for turn in base["conversations"][:2]]
        copy["conversations"][0]["content"] += "\n补充情况：" + scenario
        copy["conversations"][1]["content"] = (
            f"你补充的“{signal}”属于需要立即处理的危险信号。请停止等待线上回复，立即前往急诊或呼叫120，"
            "不要自行驾车；若身边有人，请让其陪同并携带现有用药和过敏信息。此时不应因继续拍摄或分析图片"
            "而延误急救，我也不能在线确诊或开处方。"
        )
        copy["metadata"] = dict(base.get("metadata") or {})
        copy["metadata"]["dialogue_pairs"] = 0
        copy["metadata"]["safety_augmentation"] = "red_flag"
        additions.append(copy)
    return samples + additions


def command_build(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    manifest = read_json(output / "image_manifest.private.json")
    entries = {item["image_id"]: item for item in manifest}
    records = {item["record_id"]: item for item in read_json(output / "record_index.private.json")}
    analyses = load_jsonl_index(output / "vlm_analysis.private.jsonl")
    verification = load_jsonl_index(output / "redaction_verification.private.jsonl")
    components = patient_components(records, manifest)

    clinical_by_record: dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]] = defaultdict(list)
    report_by_record: dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]] = defaultdict(list)
    for image_id, analysis in analyses.items():
        entry = entries.get(image_id)
        if not entry or analysis.get("quality") == "unusable":
            continue
        if entry["source_field"] == "tongue_face_img":
            path = output / entry["local_path"]
            if analysis.get("image_type") not in CLINICAL_TYPES or not is_complete_image(path):
                continue
            target = clinical_by_record
        else:
            path = output / entry["redacted_path"]
            verified = verification.get(image_id, {}).get("pii_clear") is True
            if analysis.get("image_type") != "report" or not is_complete_image(path):
                continue
            if not is_supported_uploaded_report(analysis):
                continue
            if not verified and not args.allow_unverified_reports:
                continue
            target = report_by_record
        for occurrence in entry["occurrences"]:
            if occurrence["record_id"] in records:
                target[occurrence["record_id"]].append((entry, analysis, path))

    realtime_candidates: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    report_upload_candidates: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for rid, record in records.items():
        clinical = sorted(clinical_by_record.get(rid, []), key=lambda item: item[0]["image_id"])
        reports = sorted(report_by_record.get(rid, []), key=lambda item: item[0]["image_id"])
        component = components.get(record["patient_key"], record["patient_key"])
        split = split_name(component, args.seed)
        dialogue_sample = text_dialogue_sample(record, args.max_dialogue_pairs)
        if dialogue_sample is not None:
            realtime_candidates[split].append(dialogue_sample)
        for item in clinical:
            realtime_candidates[split].append(make_sample(record, [item[0]], [item[1]], [item[2]], "clinical_single", args.max_dialogue_pairs))
        for group in chunks(clinical, args.max_clinical_group):
            if len(group) > 1:
                realtime_candidates[split].append(make_sample(record, [x[0] for x in group], [x[1] for x in group], [x[2] for x in group], "clinical_group", args.max_dialogue_pairs))
        for item in reports:
            report_upload_candidates[split].append(make_sample(record, [item[0]], [item[1]], [item[2]], "report_upload_single", args.max_dialogue_pairs))
        for group in chunks(reports, args.max_report_group):
            if len(group) > 1:
                report_upload_candidates[split].append(make_sample(record, [x[0] for x in group], [x[1] for x in group], [x[2] for x in group], "report_upload_group", args.max_dialogue_pairs))

    stats: dict[str, Any] = {
        "manifest_images": len(manifest),
        "analyzed_images": sum(image_id in analyses for image_id in entries),
        "orphaned_cached_analyses": sum(image_id not in entries for image_id in analyses),
        "verified_report_images": sum(item.get("pii_clear") is True for item in verification.values()),
        "build_is_partial": any(image_id not in analyses for image_id in entries),
        "allow_unverified_reports": args.allow_unverified_reports,
        "dataset_policy": {
            "realtime": "clinical images only; report/document OCR is prohibited",
            "report_upload": "verified report images from the explicit manual-upload path only",
            "combined_samples": 0,
        },
        "realtime_splits": {},
        "report_upload_splits": {},
    }
    for split, values in realtime_candidates.items():
        selected = shuffled(values, f"{args.seed}:realtime:{split}")
        if split == "train":
            selected = add_red_flag_samples(selected, args.red_flag_samples)
        category_counts = Counter(item["_category"] for item in selected)
        public = []
        for item in selected:
            public.append({key: value for key, value in item.items() if not key.startswith("_")})
        write_json(output / f"{split}.json", public)
        stats["realtime_splits"][split] = {
            "candidate_count": len(values),
            "selected_count": len(public),
            "categories": dict(category_counts),
            "visit_types": dict(Counter(item.get("metadata", {}).get("visit_type", "未知") for item in public)),
            "multiturn_samples": sum(len(item.get("conversations", [])) > 2 for item in public),
            "dialogue_pairs": sum(item.get("metadata", {}).get("dialogue_pairs", 0) for item in public),
        }
    for split, values in report_upload_candidates.items():
        selected = shuffled(values, f"{args.seed}:report-upload:{split}")
        category_counts = Counter(item["_category"] for item in selected)
        public = [{key: value for key, value in item.items() if not key.startswith("_")} for item in selected]
        write_json(output / f"report_upload_{split}.json", public)
        stats["report_upload_splits"][split] = {
            "candidate_count": len(values),
            "selected_count": len(public),
            "categories": dict(category_counts),
        }
    write_json(output / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def add_common_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--medical-root", type=Path, default=DEFAULT_MEDICAL_ROOT)
    parser.add_argument("--api-key", default=None, help="默认从环境变量或 medical.config 读取")
    parser.add_argument("--base-url", default=None, help="默认从环境变量或 medical.config 读取")
    parser.add_argument("--model", default=None, help="默认使用 config.QWEN_IMAGE_DESCRIBE_MODEL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-images", type=int, default=0, help="0 表示全部，调试时可限制数量")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    prepare.add_argument("--dialogues", type=Path, default=DEFAULT_DIALOGUES)

    download = subparsers.add_parser("download")
    download.add_argument("--source-field", choices=["tongue_face_img", "admin_report_img"], default=None)
    download.add_argument("--max-images", type=int, default=0)
    analyze = subparsers.add_parser("analyze")
    add_common_api_args(analyze)
    analyze.add_argument("--source-field", choices=["tongue_face_img", "admin_report_img"], default=None)
    redact = subparsers.add_parser("redact")
    redact.add_argument("--header-redact-ratio", type=float, default=0.24)
    verify = subparsers.add_parser("verify")
    add_common_api_args(verify)
    build = subparsers.add_parser("build")
    build.add_argument("--seed", default="20260715")
    build.add_argument("--max-clinical-group", type=int, default=4)
    build.add_argument("--max-report-group", type=int, default=4)
    build.add_argument("--max-dialogue-pairs", type=int, default=12, help="每个实时样本最多保留多少组医生问题/患者回答")
    build.add_argument("--allow-unverified-reports", action="store_true", help="仅限开发；生产数据不应使用")
    build.add_argument("--red-flag-samples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = {
        "prepare": command_prepare,
        "download": command_download,
        "analyze": command_analyze,
        "redact": command_redact,
        "verify": command_verify,
        "build": command_build,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
