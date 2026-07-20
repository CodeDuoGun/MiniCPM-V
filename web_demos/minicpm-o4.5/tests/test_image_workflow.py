from __future__ import annotations

import os
import sys
import base64
import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

os.environ["LOAD_MODEL"] = "false"
DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from app.context import ConsultationContext  # noqa: E402
from app.image_analysis import ImageAnalyzer, make_image_record  # noqa: E402
from app.persistence import ConsultationStore  # noqa: E402
from app.settings import Settings  # noqa: E402
from app import server  # noqa: E402


def test_quality_gate_only_accepts_usable(monkeypatch):
    analyzer = ImageAnalyzer(Settings(
        load_model=False,
        vision_vlm_base_url="https://vlm.example/v1",
        vision_vlm_api_key="test",
        vision_vlm_model="vlm",
    ))
    monkeypatch.setattr(analyzer, "_call", lambda *_: {
        "passed": True,
        "quality": "limited",
        "issues": ["舌体未完整进入画面"],
        "retake_instruction": "请重新拍摄",
    })
    result = analyzer.quality_check("tongue", "unused", "image/jpeg")
    assert result["passed"] is False
    assert result["quality"] == "limited"


def test_visual_analysis_updates_condition_and_next_request():
    context = ConsultationContext(Settings(load_model=False))
    context.reset({"visit_type": "初诊", "patient_age": 30})
    record = make_image_record(
        consultation_id="consultation-1",
        scene="tongue",
        source="assistant_requested",
        upload={"image_id": "image-1", "image_key": "key", "image_url": "https://cdn/tongue.jpg"},
        quality={"passed": True, "quality": "usable", "issues": [], "retake_instruction": ""},
        analysis={"summary": "舌体淡红，舌苔薄白。", "findings": {}, "limitations": []},
        model="vlm",
    )
    context.record_visual_observation(record)
    snapshot = context.snapshot()
    filled = {item["name"]: item.get("value") for item in snapshot["slots"] if item["status"] == "filled"}
    assert filled["tongue"] == "https://cdn/tongue.jpg"
    assert filled["tongue_analysis"] == "舌体淡红，舌苔薄白。"
    assert snapshot["next_media_request"] == "lesion"
    assert snapshot["history"][-1]["role"] == "tool"
    assert "舌体淡红" in context.build_prompt()


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, _ttl, value):
        self.values[key] = value

    def set(self, key, value):
        self.values[key] = value


def test_redis_document_keeps_images_when_context_is_refreshed():
    store = ConsultationStore(Settings(load_model=False))
    store.client = FakeRedis()
    snapshot = {
        "visit_type": "初诊",
        "turn_id": 1,
        "signals": {},
        "slots": [],
        "visual_observations": [],
        "history": [{"role": "user", "content": "失眠"}],
    }
    store.save_context("consultation-1", "uid-1", {"patient_age": 30}, snapshot)
    store.append_image("consultation-1", "uid-1", {"image_id": "image-1", "status": "quality_rejected"})
    store.save_context("consultation-1", "uid-1", {"patient_age": 30}, snapshot)
    restored = store.get("consultation-1")
    assert restored["patient"]["patient_age"] == 30
    assert restored["conversation"][0]["content"] == "失眠"
    assert restored["images"][0]["status"] == "quality_rejected"


class FakeImageStorage:
    configured = True

    def upload(self, _raw, consultation_id, scene, _mime_type):
        return {
            "image_id": "image-1",
            "image_key": f"{consultation_id}/{scene}/image-1.jpg",
            "image_url": f"https://cdn/{consultation_id}/{scene}/image-1.jpg",
        }


class FakeAnalyzer:
    configured = True

    def quality_check(self, *_args):
        return {"passed": True, "quality": "usable", "issues": [], "retake_instruction": ""}

    def analyze(self, scene, *_args):
        return {"summary": f"{scene}客观分析", "findings": {}, "limitations": [], "red_flags": [], "next_questions": []}


def image_b64():
    output = io.BytesIO()
    Image.new("RGB", (224, 224), "white").save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode()


def test_image_endpoint_runs_quality_analysis_and_persists(monkeypatch):
    store = ConsultationStore(Settings(load_model=False))
    store.client = FakeRedis()
    monkeypatch.setattr(server, "consultation_store", store)
    monkeypatch.setattr(server, "image_storage", FakeImageStorage())
    monkeypatch.setattr(server, "image_analyzer", FakeAnalyzer())
    monkeypatch.setattr(server, "registry", server.SessionRegistry())
    monkeypatch.setattr(server.settings, "vision_vlm_model", "test-vlm")
    with TestClient(server.app) as client:
        response = client.post("/api/v1/images/analyze", headers={"uid": "uid-1"}, json={
            "consultation_id": "consultation-1",
            "scene": "tongue",
            "source": "assistant_requested",
            "mime_type": "image/jpeg",
            "image_data": image_b64(),
        })
    assert response.status_code == 200
    assert response.json()["observation"]["status"] == "analyzed"
    assert response.json()["next_media_request"] == "lesion"
    document = store.get("consultation-1")
    assert document["images"][0]["image_url"].startswith("https://cdn/")
    assert document["condition"]["visual_observations"][0]["summary"] == "tongue客观分析"
