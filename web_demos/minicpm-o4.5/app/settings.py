from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal


DEMO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEMO_ROOT.parents[1]


@dataclass
class Settings:
    model_id: str = "openbmb/MiniCPM-o-4_5"
    device: str = "cuda:0"
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: Literal["sdpa", "flash_attention_2"] = "sdpa"
    load_model: bool = True
    model_mode: Literal["duplex", "simplex"] = "duplex"
    port: int = 32560
    log_level: str = "INFO"
    max_active_sessions: int = 1
    session_idle_seconds: int = 900
    max_turn_audio_seconds: int = 120

    lora_adapter: str = ""
    merge_lora: bool = True
    ref_audio_path: str = "assets/ref_audios/Wuweiping_test3_ref_16k_mono.wav"

    slot_config: str = "data/slots/doctor_wuweiping.json"
    slot_history_turns: int = 12
    slot_llm_base_url: str = ""
    slot_llm_api_key: str = ""
    slot_llm_model: str = ""
    slot_asr_api_key: str = ""
    slot_asr_model: str = "paraformer-realtime-v2"
    slot_asr_ws_url: str = ""
    slot_asr_vocabulary_id: str = ""

    rollout_percent: int = 0
    rollout_salt: str = "change-me-before-production"
    media_retention_hours: int = 0
    log_root: str = "web_demos/minicpm-o4.5/runtime_logs"

    @classmethod
    def from_env(cls) -> "Settings":
        values = {}
        for item in fields(cls):
            raw = os.getenv(item.name.upper())
            if raw is None:
                continue
            default = item.default
            if isinstance(default, bool):
                values[item.name] = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(default, int):
                values[item.name] = int(raw)
            else:
                values[item.name] = raw
        return cls(**values)

    def project_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def slot_config_path(self) -> Path:
        return self.project_path(self.slot_config)

    @property
    def reference_audio_path(self) -> Path:
        return self.project_path(self.ref_audio_path)

    @property
    def log_root_path(self) -> Path:
        return self.project_path(self.log_root)


settings = Settings.from_env()
