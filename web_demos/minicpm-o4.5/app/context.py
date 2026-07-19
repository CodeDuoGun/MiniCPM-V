from __future__ import annotations

import threading
from typing import Any

from . import slot_manager
from .settings import Settings


TCM_SYSTEM_PROMPT = (
    "你是吴卫平医生的数字问诊助手。你负责真实、谨慎、连续地采集病史，并客观描述镜头中可见的舌面、面部或患处。"
    "每次优先询问一至三个关键问题，不机械重复已经明确的信息。不得仅凭图像确诊，不替代医生作最终诊断，不开处方，"
    "不承诺疗效。实时视频不得读取检查报告、处方、病历、证件或其他文档；看到疑似文档时请用户通过手动上传入口提交。"
    "遇到胸痛、呼吸困难、意识异常、严重过敏、持续高热等危险信号，明确建议立即线下就医或急诊。"
)


class ConsultationContext:
    """Owns patient profile, bounded dialogue history and the existing slot rules."""

    def __init__(self, config: Settings):
        extractor = slot_manager.OpenAICompatibleSlotExtractor(
            config.slot_llm_base_url,
            config.slot_llm_api_key,
            config.slot_llm_model,
        )
        self.slots = slot_manager.SlotConversation(
            str(config.slot_config_path),
            extractor=extractor if extractor.configured else None,
            history_turns=config.slot_history_turns,
        )
        self.transcriber = slot_manager.DashScopeAudioTranscriber(
            api_key=config.slot_asr_api_key,
            model=config.slot_asr_model,
            websocket_url=config.slot_asr_ws_url,
            vocabulary_id=config.slot_asr_vocabulary_id,
        )
        self.lock = threading.RLock()
        self.profile: dict[str, Any] = {}

    def reset(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        self.profile = {
            "visit_type": options.get("visit_type", "初诊"),
            "patient_gender": options.get("patient_gender") or options.get("gender") or "未知",
            "patient_age": options.get("patient_age") or options.get("age") or "未知",
        }
        with self.lock:
            self.slots.reset(self.profile["visit_type"], profile=self.profile)

    def update_user(self, transcript: str, audio_pcm: bytes | None = None) -> str:
        text = transcript.strip()
        if not text and audio_pcm and self.transcriber.configured:
            text = self.transcriber.transcribe_pcm(audio_pcm, 16000)
        if text:
            with self.lock:
                self.slots.process_user_turn(text)
        return text

    def record_assistant(self, text: str) -> None:
        if text.strip():
            with self.lock:
                self.slots.record_assistant(text)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.slots.snapshot()

    def build_prompt(self) -> str:
        with self.lock:
            state = self.slots.build_model_context()
            history = self.slots.history[-12:]
        history_text = "\n".join(
            f"{'患者' if item['role'] == 'user' else '助手'}：{item['content']}" for item in history
        )
        return (
            f"{TCM_SYSTEM_PROMPT}\n\n患者资料：{self.profile}\n\n"
            f"结构化问诊状态：\n{state}\n\n最近对话：\n{history_text or '无'}"
        )
