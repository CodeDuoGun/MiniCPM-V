from __future__ import annotations

import base64
import io
import json
import re
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from .persistence import utc_now
from .settings import Settings


SCENES = {"tongue", "lesion", "report"}
SOURCES = {"assistant_requested", "manual_upload"}
MIME_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
DIRECT_IDENTIFIER_RE = re.compile(r"(?<!\d)(?:1\d{10}|\d{17}[\dXx]|\d{15})(?!\d)")
PRIVATE_FIELD_RE = re.compile(r"姓名|身份证|手机号|电话|地址|就诊号|住院号|病历号|条码|二维码")

QUALITY_PROMPTS = {
    "tongue": "检查图片是否能用于舌象观察：舌体需完整、对焦清晰、自然伸舌、无遮挡、不过曝，光线不能明显偏色。",
    "lesion": "检查图片是否能用于患处观察：患处需清晰、曝光正常、主体完整，不能被衣物、手指或滤镜明显遮挡。",
    "report": "检查图片是否能用于检查报告识别：页面需方向正确、文字清晰、主要项目完整、没有严重反光或遮挡。",
}

ANALYSIS_PROMPTS = {
    "tongue": """客观描述舌面，不诊断、不开方。仅返回JSON：
{"summary":"简短摘要","findings":{"tongue_body_color":"","tongue_shape":"","coating_color":"","coating_thickness":"","moisture":"","other":""},"limitations":[],"red_flags":[],"next_questions":[]}""",
    "lesion": """客观描述患处，不推断疾病、不诊断、不开方。仅返回JSON：
{"summary":"简短摘要","findings":{"location":"","extent":"","color":"","morphology":"","border":"","surface":"","exudate":"","other":""},"limitations":[],"red_flags":[],"next_questions":[]}""",
    "report": """图片是用户通过显式上传入口提交的检查报告。仅提取真实可见的检查项目、结果、单位、参考范围和异常标记；不要输出姓名、证件号、手机号、地址、就诊号、住院号、条码或二维码；不要诊断、推测病因或推荐药物。仅返回JSON：
{"summary":"不含身份信息的客观摘要","findings":{"report_type":"未知","report_date":"未知","items":[{"name":"","value":"","unit":"","reference":"","flag":"未知"}]},"limitations":["OCR可能有误，请以报告原件和医生复核为准"],"red_flags":[],"next_questions":[]}""",
}


def _extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`").removeprefix("json").strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM response does not contain a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("VLM response must be a JSON object")
    return parsed


def decode_image(image_data: str, mime_type: str) -> tuple[bytes, Image.Image]:
    if mime_type not in MIME_EXTENSIONS:
        raise ValueError("unsupported image mime type")
    try:
        raw = base64.b64decode(image_data, validate=True)
        if not raw or len(raw) > 15 * 1024 * 1024:
            raise ValueError("image must be between 1 byte and 15 MB")
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            converted = image.convert("RGB")
        if converted.width < 224 or converted.height < 224:
            raise ValueError("image resolution must be at least 224x224")
        return raw, converted
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid image_data") from exc


class QiniuImageStorage:
    def __init__(self, config: Settings):
        self.config = config
        self.manager: Any = None
        if all((config.qiniu_access_key, config.qiniu_secret_key, config.qiniu_bucket_name, config.qiniu_bucket_domain)):
            from utils.qiniu_lib import QiniuManager

            self.manager = QiniuManager(
                config.qiniu_access_key,
                config.qiniu_secret_key,
                config.qiniu_bucket_name,
                config.qiniu_bucket_domain,
            )

    @property
    def configured(self) -> bool:
        return self.manager is not None

    def upload(self, raw: bytes, consultation_id: str, scene: str, mime_type: str) -> dict[str, Any]:
        if not self.manager:
            raise RuntimeError("Qiniu image storage is not configured")
        image_id = str(uuid.uuid4())
        suffix = MIME_EXTENSIONS[mime_type]
        key = f"{self.config.qiniu_key_prefix.strip('/')}/{consultation_id}/{scene}/{image_id}{suffix}"
        with tempfile.TemporaryDirectory(prefix="minicpmo45-image-") as temp_dir:
            local_path = Path(temp_dir) / f"image{suffix}"
            local_path.write_bytes(raw)
            uploaded = self.manager.upload(str(local_path), key=key, overwrite=False)
        return {"image_id": image_id, "image_key": uploaded["key"], "image_url": uploaded["url"]}


class ImageAnalyzer:
    def __init__(self, config: Settings):
        self.config = config

    @property
    def configured(self) -> bool:
        return bool(self.config.vision_vlm_base_url and self.config.vision_vlm_api_key and self.config.vision_vlm_model)

    def _url(self) -> str:
        base = self.config.vision_vlm_base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def _call(self, image_data: str, mime_type: str, prompt: str) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError("vision VLM is not configured")
        payload = {
            "model": self.config.vision_vlm_model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                {"type": "text", "text": prompt},
            ]}],
        }
        request = urllib.request.Request(
            self._url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.vision_vlm_api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        return _extract_json_object(result["choices"][0]["message"]["content"])

    def quality_check(self, scene: str, image_data: str, mime_type: str) -> dict[str, Any]:
        prompt = QUALITY_PROMPTS[scene] + """不要分析病情。仅返回JSON：
{"passed":true,"quality":"usable|limited|unusable","issues":[],"retake_instruction":"通过时为空，否则给出简短重拍指导"}"""
        raw = self._call(image_data, mime_type, prompt)
        quality = raw.get("quality") if raw.get("quality") in {"usable", "limited", "unusable"} else "unusable"
        passed = bool(raw.get("passed")) and quality == "usable"
        return {
            "passed": passed,
            "quality": quality,
            "issues": [str(item)[:120] for item in (raw.get("issues") or [])[:8]],
            "retake_instruction": str(raw.get("retake_instruction") or "")[:300],
        }

    def analyze(self, scene: str, image_data: str, mime_type: str) -> dict[str, Any]:
        raw = self._call(image_data, mime_type, ANALYSIS_PROMPTS[scene])
        summary = DIRECT_IDENTIFIER_RE.sub("[已脱敏]", str(raw.get("summary") or "")[:800])
        if scene == "report" and PRIVATE_FIELD_RE.search(summary):
            summary = "检查报告已完成结构化提取；身份字段已隐藏，请以报告原件和医生复核为准。"
        findings = raw.get("findings") if isinstance(raw.get("findings"), dict) else {}
        if scene == "report":
            findings = self._sanitize_report_findings(findings)
        return {
            "summary": summary,
            "findings": findings,
            "limitations": [str(item)[:200] for item in (raw.get("limitations") or [])[:8]],
            "red_flags": [str(item)[:160] for item in (raw.get("red_flags") or [])[:8]],
            "next_questions": [str(item)[:160] for item in (raw.get("next_questions") or [])[:8]],
        }

    @staticmethod
    def _sanitize_report_findings(findings: dict[str, Any]) -> dict[str, Any]:
        items = []
        for raw in findings.get("items") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "")[:80]
            value = DIRECT_IDENTIFIER_RE.sub("[已脱敏]", str(raw.get("value") or "")[:120])
            if PRIVATE_FIELD_RE.search(name):
                continue
            items.append({
                "name": name,
                "value": value,
                "unit": str(raw.get("unit") or "")[:40],
                "reference": str(raw.get("reference") or "")[:80],
                "flag": str(raw.get("flag") or "未知")[:10],
            })
            if len(items) >= 30:
                break
        return {
            "report_type": str(findings.get("report_type") or "未知")[:80],
            "report_date": str(findings.get("report_date") or "未知")[:20],
            "items": items,
        }


def make_image_record(
    *,
    consultation_id: str,
    scene: str,
    source: str,
    upload: dict[str, Any],
    quality: dict[str, Any],
    analysis: dict[str, Any] | None,
    model: str,
) -> dict[str, Any]:
    return {
        "observation_id": str(uuid.uuid4()),
        "consultation_id": consultation_id,
        "scene": scene,
        "source": source,
        **upload,
        "status": "analyzed" if analysis is not None else "quality_rejected",
        "quality_llm_result": quality,
        "analysis_llm_result": analysis,
        "model": model,
        "created_at": utc_now(),
    }
