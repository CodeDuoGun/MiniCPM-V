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
    "问诊过程中应在合适时机主动、每次只要求一种资料：先请患者通过拍照入口展示舌面，再展示患处，最后通过手动上传入口提交检查报告。"
    "用户也可以随时主动上传。所有图片必须先完成质量检查；只有系统明确标记为质控通过的分析结果才可作为病情参考。"
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
        self.visual_observations: list[dict[str, Any]] = []

    def reset(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        self.profile = {
            "visit_type": options.get("visit_type", "初诊"),
            "patient_gender": options.get("patient_gender") or options.get("gender") or "未知",
            "patient_age": options.get("patient_age") or options.get("age") or "未知",
        }
        with self.lock:
            self.slots.reset(self.profile["visit_type"], profile=self.profile)
            self.visual_observations = []

    def restore(self, document: dict[str, Any]) -> None:
        condition = document.get("condition") if isinstance(document.get("condition"), dict) else {}
        patient = document.get("patient") if isinstance(document.get("patient"), dict) else {}
        snapshot = {
            "visit_type": condition.get("visit_type") or patient.get("visit_type") or "初诊",
            "turn_id": condition.get("turn_id", 0),
            "signals": condition.get("signals") or {},
            "slots": condition.get("slots") or [],
            "history": document.get("conversation") or [],
        }
        with self.lock:
            self.profile = dict(patient)
            self.slots.restore(snapshot)
            self.visual_observations = list(condition.get("visual_observations") or [])[-12:]

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
            snapshot = self.slots.snapshot()
            snapshot["visual_observations"] = list(self.visual_observations)
            snapshot["next_media_request"] = self.next_media_request()
            return snapshot

    def next_media_request(self) -> str | None:
        completed = {str(item.get("scene")) for item in self.visual_observations if item.get("status") == "analyzed"}
        return next((scene for scene in ("tongue", "lesion", "report") if scene not in completed), None)

    def record_visual_observation(self, record: dict[str, Any]) -> None:
        analysis = record.get("analysis_llm_result") or {}
        summary = str(analysis.get("summary") or "").strip()
        if record.get("status") != "analyzed" or not summary:
            return
        scene = str(record.get("scene") or "")
        with self.lock:
            compact = {
                "observation_id": record.get("observation_id"),
                "scene": scene,
                "status": "analyzed",
                "image_url": record.get("image_url"),
                "summary": summary,
                "findings": analysis.get("findings") or {},
                "limitations": analysis.get("limitations") or [],
                "created_at": record.get("created_at"),
            }
            self.visual_observations.append(compact)
            self.visual_observations = self.visual_observations[-12:]
            slot_names = {
                "tongue": ("tongue", "tongue_analysis"),
                "lesion": ("lesion_image", "lesion_analysis"),
                "report": ("exam_report", "exam_analysis"),
            }
            image_slot, analysis_slot = slot_names[scene]
            evidence = f"图片分析 {record.get('observation_id')}"
            self.slots.set_external_value(image_slot, record.get("image_url"), evidence)
            self.slots.set_external_value(analysis_slot, summary, evidence)
            if scene == "report":
                self.slots.set_signal("report_uploaded", True)
            self.slots.record_tool(
                f"{scene}图片质控通过并完成分析：{summary}",
                type="visual_observation",
                scene=scene,
                observation_id=record.get("observation_id"),
                image_url=record.get("image_url"),
            )

    def build_prompt(self) -> str:
        with self.lock:
            state = self.slots.build_model_context()
            history = self.slots.history[-12:]
            observations = list(self.visual_observations[-6:])
            next_media = self.next_media_request()
        history_text = "\n".join(
            f"{ {'user': '患者', 'assistant': '助手', 'tool': '系统资料'}.get(item['role'], '系统') }：{item['content']}"
            for item in history
        )
        observation_text = "\n".join(
            f"- {item['scene']}（{item.get('created_at', '未知时间')}）：{item['summary']}"
            for item in observations
        ) or "- 暂无"
        return (
            f"{TCM_SYSTEM_PROMPT}\n\n患者资料：{self.profile}\n\n"
            f"结构化问诊状态：\n{state}\n\n已通过质控的视觉资料：\n{observation_text}\n\n"
            f"下一项待收集图片：{next_media or '已收集完成'}。不要重复索取已完成资料。\n\n"
            f"最近对话：\n{history_text or '无'}"
        )
