from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


os.environ["LOAD_MODEL"] = "false"
DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from app import server  # noqa: E402


def test_legacy_stop_endpoint_resets_pending_turn(monkeypatch):
    monkeypatch.setattr(server, "registry", server.SessionRegistry())
    with TestClient(server.app) as client:
        response = client.post("/api/v1/stop?type=0", headers={"uid": "uid-1"})

    assert response.status_code == 200
    assert response.json() == {
        "id": "uid-1",
        "choices": {
            "role": "assistant",
            "content": "success",
            "finish_reason": "stop",
        },
    }
