from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image


TARGET_AUDIO_RATE = 16000


@dataclass
class InputChunk:
    audio: np.ndarray | None = None
    frames: list[Image.Image] = field(default_factory=list)
    transcript: str = ""
    timestamp: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    end_of_turn: bool = False


def decode_audio(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1)
    if sample_rate != TARGET_AUDIO_RATE:
        # Linear interpolation keeps protocol parsing independent of SciPy. The model
        # receives short speech chunks; production environments may replace this with
        # soxr without changing the wire protocol.
        output_size = max(1, round(len(audio) * TARGET_AUDIO_RATE / sample_rate))
        source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=output_size, endpoint=False)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def decode_image(value: str) -> Image.Image:
    raw = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        return image.convert("RGB")


def parse_payload(payload: dict[str, Any]) -> InputChunk:
    chunk = InputChunk()
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        for item in message.get("content") or []:
            item_type = item.get("type")
            if item_type == "input_audio":
                body = item.get("input_audio") or {}
                if body.get("data"):
                    chunk.audio = decode_audio(body["data"])
                chunk.transcript = str(body.get("transcript") or body.get("text") or "").strip()
                chunk.timestamp = str(body.get("timestamp") or "")
                chunk.end_of_turn = bool(body.get("end_of_turn", chunk.end_of_turn))
            elif item_type == "image_data":
                body = item.get("image_data") or {}
                if body.get("source", "realtime_video") != "realtime_video":
                    raise ValueError("streaming image_data only accepts source=realtime_video")
                if body.get("data"):
                    chunk.frames.append(decode_image(body["data"]))
            elif item_type == "input_text":
                body = item.get("input_text")
                chunk.transcript = str(body.get("text") if isinstance(body, dict) else body or "").strip()
                chunk.end_of_turn = True
            elif item_type == "options":
                chunk.options.update(item.get("options") or {})
            elif item_type == "input_control":
                body = item.get("input_control") or {}
                chunk.end_of_turn = bool(body.get("end_of_turn", chunk.end_of_turn))
            else:
                raise ValueError(f"unsupported content type: {item_type}")
    return chunk


def encode_audio(audio: Any, sample_rate: int = 24000) -> str:
    if audio is None:
        return ""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(output.getvalue()).decode("ascii")
