from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from app.runtime import MiniCPMO45Runtime  # noqa: E402
from app.settings import Settings  # noqa: E402


class FakeDuplexModel:
    def __init__(self):
        self.prepare_paths = []
        self.reset_count = 0
        self.token2wav_initialized = False
        self.flow_cache_base = None
        self.hift_cache_base = None
        self.pre_lookahead = 0

    def prepare(self, **kwargs):
        path = kwargs.get("prompt_wav_path")
        self.prepare_paths.append(path)
        self.token2wav_initialized = False
        self.flow_cache_base = None
        self.hift_cache_base = None
        self.pre_lookahead = 0
        if path:
            self.token2wav_initialized = True
            self.flow_cache_base = {"flow": "base"}
            self.hift_cache_base = {"hift": "base"}
            self.pre_lookahead = 3

    def _reset_token2wav_for_new_turn(self):
        self.reset_count += 1


def test_prepare_reuses_token2wav_base_cache():
    runtime = MiniCPMO45Runtime(Settings(load_model=False, model_mode="duplex"))
    runtime.model = FakeDuplexModel()
    runtime.ref_audio = np.zeros(16000, dtype=np.float32)

    runtime.prepare("uid-1", "first prompt")
    runtime.prepare("uid-2", "second prompt")

    assert runtime.model.prepare_paths[0]
    assert runtime.model.prepare_paths[1] is None
    assert runtime.model.flow_cache_base == {"flow": "base"}
    assert runtime.model.hift_cache_base == {"hift": "base"}
    assert runtime.model.token2wav_initialized is True
    assert runtime.model.reset_count == 1
