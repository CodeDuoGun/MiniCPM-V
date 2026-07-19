#!/usr/bin/env python3
"""Load the unmodified model and validate the official 4.5 duplex surface."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from app.runtime import MiniCPMO45Runtime  # noqa: E402
from app.settings import Settings  # noqa: E402


def main() -> None:
    config = Settings(load_model=True, lora_adapter="")
    runtime = MiniCPMO45Runtime(config)
    runtime.prepare("smoke", "你是谨慎的中医预问诊助手，只做信息采集。")
    capabilities = runtime.capabilities()
    required = {"audio_waveform", "frame_list"}
    params = set(inspect.signature(runtime.model.streaming_prefill).parameters)
    if not required.issubset(params):
        raise RuntimeError(f"unexpected duplex streaming_prefill signature: {params}")
    print(capabilities)
    print("MiniCPM-o 4.5 load and duplex API smoke test passed")


if __name__ == "__main__":
    main()

