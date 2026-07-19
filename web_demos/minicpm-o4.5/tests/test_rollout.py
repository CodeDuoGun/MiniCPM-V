from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ["LOAD_MODEL"] = "false"
DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from app.server import rollout_enabled  # noqa: E402
from app.settings import settings  # noqa: E402


def test_rollout_is_deterministic():
    settings.rollout_percent = 50
    assert rollout_enabled("patient-1") == rollout_enabled("patient-1")


def test_rollout_bounds():
    settings.rollout_percent = 0
    assert rollout_enabled("patient-1") is False
    settings.rollout_percent = 100
    assert rollout_enabled("patient-1") is True

