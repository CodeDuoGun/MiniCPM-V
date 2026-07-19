from __future__ import annotations

import inspect
import logging
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .settings import Settings


logger = logging.getLogger(__name__)


class MiniCPMO45Runtime:
    """Thin, testable adapter around the official 4.5 duplex API."""

    def __init__(self, config: Settings):
        self.config = config
        self.base_model: Any = None
        self.model: Any = None
        self.ref_audio: np.ndarray | None = None
        self.lock = threading.RLock()
        self.prepared_uid = ""
        if config.load_model:
            self.load()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import torch
        from transformers import AutoModel

        dtype = getattr(torch, self.config.torch_dtype)
        logger.info("loading %s on %s", self.config.model_id, self.config.device)
        model = AutoModel.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=dtype,
            init_vision=True,
            init_audio=True,
            init_tts=True,
        )
        if self.config.lora_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.config.lora_adapter)
            if self.config.merge_lora:
                model = model.merge_and_unload()
        model.eval().to(self.config.device)
        model.init_tts()
        self.base_model = model
        self.model = model.as_duplex() if self.config.model_mode == "duplex" else model
        ref_path = self.config.reference_audio_path
        if ref_path.is_file():
            self.ref_audio, _ = librosa.load(ref_path, sr=16000, mono=True)
        logger.info("model loaded; mode=%s", self.config.model_mode)

    def prepare(self, uid: str, prompt: str) -> None:
        if not self.loaded:
            self.prepared_uid = uid
            return
        with self.lock:
            if self.config.model_mode == "duplex":
                kwargs: dict[str, Any] = {"prefix_system_prompt": prompt}
                if self.ref_audio is not None:
                    kwargs["ref_audio"] = self.ref_audio
                    kwargs["prompt_wav_path"] = str(self.config.reference_audio_path)
                self.model.prepare(**kwargs)
            else:
                if hasattr(self.model, "reset_session"):
                    self.model.reset_session()
            self.prepared_uid = uid

    def process_duplex_chunk(self, audio: np.ndarray | None, frames: list[Any]) -> dict[str, Any]:
        if not self.loaded:
            return {
                "text": "",
                "audio_waveform": None,
                "sampling_rate": 24000,
                "is_listen": True,
                "end_of_turn": False,
                "mock": True,
            }
        with self.lock:
            self.model.streaming_prefill(
                audio_waveform=audio,
                frame_list=frames,
                max_slice_nums=1,
                batch_vision_feed=False,
            )
            result = self.model.streaming_generate(
                prompt_wav_path=str(self.config.reference_audio_path),
                max_new_speak_tokens_per_chunk=20,
                decode_mode="sampling",
            )
        if not isinstance(result, dict):
            raise TypeError("MiniCPM-o 4.5 duplex streaming_generate must return a dict")
        return result

    def capabilities(self) -> dict[str, Any]:
        signature = "unloaded"
        if self.loaded:
            signature = str(inspect.signature(self.model.streaming_prefill))
        return {
            "model": self.config.model_id,
            "mode": self.config.model_mode,
            "loaded": self.loaded,
            "prepared_uid": self.prepared_uid,
            "streaming_prefill": signature,
        }

