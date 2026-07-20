#!/usr/bin/env python3
"""Minimal recorded-media test using the same runtime as production."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from app.runtime import MiniCPMO45Runtime  # noqa: E402
from app.settings import Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-npy", help="Optional 16 kHz float32 numpy array")
    parser.add_argument("--image")
    parser.add_argument(
        "--ref-audio",
        help="Optional short 16 kHz mono reference WAV for duplex TTS",
    )
    args = parser.parse_args()

    import numpy as np

    config = Settings(load_model=True, lora_adapter="")
    if args.ref_audio:
        config.ref_audio_path = args.ref_audio
    runtime = MiniCPMO45Runtime(config)
    runtime.prepare("official-demo", "你是中医预问诊助手。请简短回应。")
    audio = np.load(args.audio_npy).astype("float32") if args.audio_npy else np.zeros(16000, dtype="float32")
    frames = [Image.open(args.image).convert("RGB")] if args.image else []
    result = runtime.process_duplex_chunk(audio, frames)
    print({key: value for key, value in result.items() if key != "audio_waveform"})


if __name__ == "__main__":
    main()
