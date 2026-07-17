"""Input policy for realtime medical video frames.

This module deliberately performs only document-likeness detection. It does
not run OCR and never extracts text from a camera frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


REALTIME_DOCUMENT_NOTICE = (
    "检测到疑似检查报告、处方、病历或其他文档画面。实时视频通道禁止读取或解释文档内容，"
    "请停止在镜头前展示，并通过页面上的检查报告手动上传入口提交，由独立VLM接口分析。"
)


def document_likeness(image: Image.Image) -> tuple[bool, dict[str, float]]:
    """Conservatively detect light document/screen frames without OCR.

    The guard combines a large low-saturation light background with a
    plausible amount of dark ink and high-frequency edges. It is intentionally
    conservative: false positives cause an upload prompt, while the system
    prompt remains the second line of defence for missed documents.
    """

    rgb = image.convert("RGB")
    rgb.thumbnail((384, 384))
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or min(array.shape[:2]) < 32:
        return False, {"reason": 0.0}

    channel_range = array.max(axis=2) - array.min(axis=2)
    gray = array.mean(axis=2)
    light_neutral = (gray >= 178) & (channel_range <= 42)
    dark_ink = gray <= 118

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_fraction = float(((dx > 24).mean() + (dy > 24).mean()) / 2)
    light_fraction = float(light_neutral.mean())
    dark_fraction = float(dark_ink.mean())

    is_document = (
        light_fraction >= 0.48
        and 0.012 <= dark_fraction <= 0.38
        and edge_fraction >= 0.035
    )
    return is_document, {
        "light_neutral_fraction": round(light_fraction, 4),
        "dark_ink_fraction": round(dark_fraction, 4),
        "edge_fraction": round(edge_fraction, 4),
    }


def is_manual_report_upload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("source") == "manual_upload"
