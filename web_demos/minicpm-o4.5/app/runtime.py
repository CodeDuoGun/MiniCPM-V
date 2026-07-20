from __future__ import annotations

import inspect
import logging
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .settings import Settings


logger = logging.getLogger(__name__)


def validate_torch_stack() -> dict[str, str]:
    expected = {"torch": "2.8.0", "torchaudio": "2.8.0", "torchvision": "0.23.0"}
    installed: dict[str, str] = {}
    for package, wanted in expected.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"missing required package: {package}=={wanted}") from exc
        if installed[package].split("+")[0] != wanted:
            raise RuntimeError(
                f"incompatible PyTorch stack: expected {package}=={wanted}, "
                f"found {installed[package]}. Reinstall torch/torchvision/torchaudio "
                "together from the same official PyTorch CUDA index."
            )
    try:
        import torch
        import torchaudio
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"PyTorch/TorchAudio binary ABI check failed: {exc}. Reinstall all three "
            "packages together from the same CUDA wheel index."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; MiniCPM-o 4.5 full inference requires an NVIDIA GPU")
    installed["torch_cuda"] = str(torch.version.cuda)
    installed["gpu"] = torch.cuda.get_device_name(0)
    return installed


def _load_reference_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1)
    if sample_rate != 16000:
        output_size = max(1, round(len(audio) * 16000 / sample_rate))
        source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=output_size, endpoint=False)
        audio = np.interp(target_positions, source_positions, audio)
    return np.ascontiguousarray(audio, dtype=np.float32)


class MiniCPMO45Runtime:
    """Thin, testable adapter around the official 4.5 duplex API."""

    def __init__(self, config: Settings):
        self.config = config
        self.base_model: Any = None
        self.model: Any = None
        self.ref_audio: np.ndarray | None = None
        self.lock = threading.RLock()
        self.prepared_uid = ""
        self._tts_prepare_cache: dict[str, Any] | None = None
        if config.load_model:
            self.load()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import torch
        from transformers import AutoModel

        stack = validate_torch_stack()
        logger.info("validated PyTorch stack: %s", stack)
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
            self.ref_audio = _load_reference_audio(ref_path)
        logger.info("model loaded; mode=%s", self.config.model_mode)

    def prepare(self, uid: str, prompt: str) -> None:
        if not self.loaded:
            self.prepared_uid = uid
            return
        with self.lock:
            started = time.perf_counter()
            if self.config.model_mode == "duplex":
                kwargs: dict[str, Any] = {"prefix_system_prompt": prompt}
                if self.ref_audio is not None:
                    kwargs["ref_audio"] = self.ref_audio
                    if self._tts_prepare_cache is None:
                        kwargs["prompt_wav_path"] = str(self.config.reference_audio_path)
                self.model.prepare(**kwargs)
                if self.ref_audio is not None:
                    if self._tts_prepare_cache is None and getattr(self.model, "token2wav_initialized", False):
                        self._tts_prepare_cache = {
                            "flow_cache_base": self.model.flow_cache_base,
                            "hift_cache_base": self.model.hift_cache_base,
                            "pre_lookahead": self.model.pre_lookahead,
                        }
                    elif self._tts_prepare_cache is not None:
                        # prepare() clears these immutable base caches. Restore them
                        # and let the official helper clone fresh per-turn state.
                        for name, value in self._tts_prepare_cache.items():
                            setattr(self.model, name, value)
                        self.model.token2wav_initialized = True
                        self.model._reset_token2wav_for_new_turn()
            else:
                if hasattr(self.model, "reset_session"):
                    self.model.reset_session()
            self.prepared_uid = uid
            logger.info(
                "prepared session uid=%s in %.2fs (cached_token2wav=%s)",
                uid,
                time.perf_counter() - started,
                self._tts_prepare_cache is not None,
            )

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
