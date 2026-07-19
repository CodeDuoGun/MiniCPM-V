from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from app.protocol import parse_payload  # noqa: E402


def audio_b64(sample_rate=8000):
    output = io.BytesIO()
    sf.write(output, np.zeros(sample_rate // 10, dtype="float32"), sample_rate, format="WAV")
    return base64.b64encode(output.getvalue()).decode()


def test_parse_legacy_payload_and_resample():
    payload = {"messages": [{"role": "user", "content": [{
        "type": "input_audio",
        "input_audio": {"data": audio_b64(), "transcript": "睡眠不好", "end_of_turn": True},
    }]}]}
    chunk = parse_payload(payload)
    assert chunk.audio.shape == (1600,)
    assert chunk.transcript == "睡眠不好"
    assert chunk.end_of_turn is True


def test_parse_options():
    chunk = parse_payload({"messages": [{"role": "user", "content": [
        {"type": "options", "options": {"visit_type": "复诊", "patient_age": 35}}
    ]}]})
    assert chunk.options["visit_type"] == "复诊"

