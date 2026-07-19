"""多轮问诊的槽位状态、语音转写和结构化抽取。

本模块刻意不依赖 MiniCPM 模型对象。实时模型只负责自然对话，患者语音先转成
文本，再由独立的文本模型更新槽位，避免 JSON 抽取任务污染实时会话的 KV cache。
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol


_FEMALE_VALUES = {"女", "女性", "female", "f", "woman"}
_MALE_VALUES = {"男", "男性", "male", "m", "man"}


def normalize_visit_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"复诊", "复查", "随访", "followup", "follow-up", "return"}:
        return "复诊"
    return "初诊"


def normalize_gender(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _FEMALE_VALUES:
        return "female"
    if text in _MALE_VALUES:
        return "male"
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("slot extractor response does not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("slot extractor response must be a JSON object")
    return value


@dataclass(frozen=True)
class SlotDefinition:
    name: str
    group: str
    definition: str
    required: bool = False
    condition: str = ""


@dataclass
class SlotValue:
    value: Any
    evidence: str = ""
    confidence: float = 1.0
    updated_turn: int = 0
    source: str = "conversation"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "updated_turn": self.updated_turn,
            "source": self.source,
        }


class SlotExtractor(Protocol):
    def extract(
        self,
        *,
        user_text: str,
        history: List[Dict[str, str]],
        definitions: List[SlotDefinition],
        collected: Mapping[str, SlotValue],
    ) -> List[Dict[str, Any]]:
        ...


class OpenAICompatibleSlotExtractor:
    """通过 OpenAI 兼容的 ``/chat/completions`` 接口抽取槽位。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def _url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def extract(
        self,
        *,
        user_text: str,
        history: List[Dict[str, str]],
        definitions: List[SlotDefinition],
        collected: Mapping[str, SlotValue],
    ) -> List[Dict[str, Any]]:
        if not self.configured or not user_text.strip():
            return []
        allowed = [
            {
                "name": item.name,
                "group": item.group,
                "definition": item.definition,
                "required": item.required,
                "condition": item.condition or None,
            }
            for item in definitions
        ]
        known = {name: item.value for name, item in collected.items()}
        recent_history = history[-16:]
        instruction = (
            "你是医生问诊槽位抽取器。只提取患者明确说出、明确否认，或由患者上传资料明确提供的信息；"
            "不得根据医生问题、常识或诊断猜测补全。否认也属于有效值，例如‘无过敏史’。"
            "结合历史解决指代，但本轮没有新增或修正时返回空 updates。"
            "若患者明确纠正旧值，返回新值。value 使用简短中文原文摘要。"
            "只能使用给定 name，输出严格 JSON，不要 Markdown："
            '{"updates":[{"name":"槽位名","value":"值","evidence":"患者原话片段",'
            '"confidence":0.0}]}。confidence 范围 0 到 1，低于 0.65 不要输出。'
        )
        user_payload = {
            "可用槽位": allowed,
            "已收集槽位": known,
            "最近对话": recent_history,
            "患者本轮回答": user_text,
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        def request_completion(body: Dict[str, Any]) -> Dict[str, Any]:
            request = urllib.request.Request(
                self._url(),
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            result = request_completion(payload)
        except urllib.error.HTTPError as exc:
            # 部分“OpenAI 兼容”服务不实现 response_format；提示词仍要求严格 JSON。
            if exc.code not in {400, 422}:
                raise
            payload.pop("response_format", None)
            result = request_completion(payload)
        content = result["choices"][0]["message"]["content"]
        updates = extract_json_object(content).get("updates") or []
        return [item for item in updates if isinstance(item, dict)]


class DashScopeAudioTranscriber:
    """把完整的一轮 16 kHz 单声道 PCM 音频送入 DashScope 实时 ASR。"""

    def __init__(
        self,
        api_key: str,
        model: str = "paraformer-realtime-v2",
        websocket_url: str = "",
        vocabulary_id: str = "",
        completion_timeout: float = 8.0,
    ):
        self.api_key = api_key
        self.model = model
        self.websocket_url = websocket_url
        self.vocabulary_id = vocabulary_id
        self.completion_timeout = completion_timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        if not self.configured or not pcm_bytes:
            return ""
        try:
            import dashscope
            from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
        except ImportError as exc:
            raise RuntimeError("启用自动槽位 ASR 需要安装 dashscope") from exc

        done = threading.Event()
        final_sentences: List[str] = []
        latest_intermediate = [""]

        class Callback(RecognitionCallback):
            def on_event(self, result: Any) -> None:
                sentence = result.get_sentence() or {}
                text = str(sentence.get("text") or "").strip()
                if not text:
                    return
                if RecognitionResult.is_sentence_end(sentence):
                    final_sentences.append(text)
                    latest_intermediate[0] = ""
                else:
                    latest_intermediate[0] = text

            def on_complete(self) -> None:
                done.set()

            def on_close(self) -> None:
                done.set()

            def on_error(self, message: Any) -> None:
                done.set()

        dashscope.api_key = self.api_key
        if self.websocket_url:
            dashscope.base_websocket_api_url = self.websocket_url
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "format": "pcm",
            "sample_rate": sample_rate,
            "semantic_punctuation_enabled": True,
            "heartbeat": True,
            "language_hints": ["zh"],
            "callback": Callback(),
        }
        if self.vocabulary_id:
            kwargs["vocabulary_id"] = self.vocabulary_id
        recognition = Recognition(**kwargs)
        recognition.start()
        try:
            # 100 ms/块，避免一次发送过大导致服务端丢帧。
            chunk_size = sample_rate * 2 // 10
            for offset in range(0, len(pcm_bytes), chunk_size):
                recognition.send_audio_frame(pcm_bytes[offset : offset + chunk_size])
            recognition.stop()
            done.wait(self.completion_timeout)
        finally:
            if getattr(recognition, "_running", False):
                recognition.stop()
        if latest_intermediate[0] and not final_sentences:
            final_sentences.append(latest_intermediate[0])
        return "".join(final_sentences).strip()


class SlotConversation:
    """一个患者会话对应一个实例；保存历史、槽位值及当前应追问项。"""

    def __init__(
        self,
        config_path: str,
        extractor: Optional[SlotExtractor] = None,
        history_turns: int = 12,
    ):
        self.config_path = str(Path(config_path).expanduser().resolve())
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)
        self.extractor = extractor
        self.history_turns = max(2, int(history_turns))
        self.visit_type = "初诊"
        self.definitions: List[SlotDefinition] = []
        self.values: Dict[str, SlotValue] = {}
        self.signals: Dict[str, Any] = {}
        self.history: List[Dict[str, str]] = []
        self.turn_id = 0
        self.last_user_text = ""
        self.last_error = ""
        self.reset("初诊")

    def _load_definitions(self, visit_type: str) -> List[SlotDefinition]:
        root_name = "followup_slots" if visit_type == "复诊" else "slots"
        root = self.config.get(root_name) or {}
        definitions: List[SlotDefinition] = []
        for group, raw_slots in root.items():
            for raw in raw_slots or []:
                name = str(raw.get("name") or "").strip()
                if not name:
                    continue
                definitions.append(
                    SlotDefinition(
                        name=name,
                        group=str(group),
                        definition=str(raw.get("definition") or name),
                        required=bool(raw.get("required")),
                        condition=str(raw.get("condition") or "").strip(),
                    )
                )
        return definitions

    def reset(self, visit_type: Any = "初诊", profile: Optional[Mapping[str, Any]] = None) -> None:
        self.visit_type = normalize_visit_type(visit_type)
        self.definitions = self._load_definitions(self.visit_type)
        self.values = {}
        self.signals = {}
        self.history = []
        self.turn_id = 0
        self.last_user_text = ""
        self.last_error = ""
        self.seed_profile(profile or {})

    def seed_profile(self, profile: Mapping[str, Any]) -> None:
        aliases = {
            "visit_type": ("visit_type",),
            "gender": ("patient_gender", "gender"),
            "age": ("patient_age", "age"),
            "last_visit_info": ("last_visit_info",),
        }
        names = {item.name for item in self.definitions}
        for slot_name, keys in aliases.items():
            if slot_name not in names:
                continue
            value = next((profile.get(key) for key in keys if profile.get(key) not in (None, "")), None)
            if slot_name == "visit_type":
                value = self.visit_type
            if value not in (None, "", "未知"):
                self.values[slot_name] = SlotValue(
                    value=value,
                    evidence="会话开始前提供的用户基本信息",
                    updated_turn=0,
                    source="profile",
                )

    def _normalized_value(self, name: str) -> str:
        value = self.values.get(name)
        raw_value = value.value if value is not None else self.signals.get(name)
        if raw_value is None:
            return ""
        if name == "gender":
            return normalize_gender(raw_value)
        text = str(raw_value).strip().lower()
        if text in {"异常", "不正常", "有异常", "abnormal"}:
            return "abnormal"
        if text in {"正常", "无异常", "normal"}:
            return "normal"
        if text in {"true", "yes", "是", "有", "已上传"}:
            return "true"
        if text in {"false", "no", "否", "无", "未上传"}:
            return "false"
        return text

    def _condition_satisfied(self, condition: str) -> bool:
        if not condition:
            return False
        atoms = [item.strip() for item in re.split(r"[;；]", condition) if item.strip()]
        for atom in atoms:
            if "=" in atom:
                name, expected = (part.strip() for part in atom.split("=", 1))
                actual = self._normalized_value(name)
                normalized_expected = normalize_gender(expected) if name == "gender" else expected.lower()
                if actual != normalized_expected:
                    return False
                continue
            # 兼容配置中的自然语言条件，例如“过往服用过异维A酸类药物”。
            keywords = [word for word in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", atom) if len(word) >= 3]
            haystack = " ".join(str(item.value) for item in self.values.values())
            if "异维a酸" in atom.lower():
                keywords = ["异维A酸"]
            if not keywords or not any(keyword.lower() in haystack.lower() for keyword in keywords):
                return False
        return True

    def is_active(self, definition: SlotDefinition) -> bool:
        if definition.required:
            return True
        if definition.condition:
            return self._condition_satisfied(definition.condition)
        if definition.name == "ask_menstruation" or definition.group == "menstrual_status":
            return self._normalized_value("gender") == "female"
        return False

    def active_definitions(self) -> List[SlotDefinition]:
        return [item for item in self.definitions if self.is_active(item)]

    def missing_definitions(self) -> List[SlotDefinition]:
        return [item for item in self.active_definitions() if item.name not in self.values]

    def apply_updates(self, updates: Iterable[Mapping[str, Any]], source: str = "conversation") -> List[str]:
        allowed = {item.name for item in self.definitions}
        changed: List[str] = []
        for update in updates:
            name = str(update.get("name") or "").strip()
            value = update.get("value")
            try:
                confidence = float(update.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if name not in allowed or value in (None, "") or confidence < 0.65:
                continue
            self.values[name] = SlotValue(
                value=value,
                evidence=str(update.get("evidence") or "")[:300],
                confidence=min(1.0, max(0.0, confidence)),
                updated_turn=self.turn_id,
                source=source,
            )
            changed.append(name)
        return changed

    def process_user_turn(self, user_text: str) -> List[str]:
        user_text = str(user_text or "").strip()
        if not user_text:
            return []
        self.turn_id += 1
        self.last_user_text = user_text
        history_before_current = self.history[-self.history_turns * 2 :]
        changed: List[str] = []
        if self.extractor is not None:
            try:
                updates = self.extractor.extract(
                    user_text=user_text,
                    history=history_before_current,
                    definitions=self.definitions,
                    collected=self.values,
                )
                changed = self.apply_updates(updates)
                self.last_error = ""
            except Exception as exc:
                self.last_error = str(exc)
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-self.history_turns * 2 :]
        return changed

    def record_assistant(self, text: str) -> None:
        text = str(text or "").strip()
        if text:
            self.history.append({"role": "assistant", "content": text})
            self.history = self.history[-self.history_turns * 2 :]

    def set_external_value(self, name: str, value: Any, evidence: str = "外部资料") -> bool:
        self.turn_id = max(1, self.turn_id)
        return bool(self.apply_updates([{
            "name": name,
            "value": value,
            "evidence": evidence,
            "confidence": 1.0,
        }], source="external"))

    def set_signal(self, name: str, value: Any) -> None:
        self.signals[str(name)] = value

    def build_model_context(self, max_missing: int = 12) -> str:
        collected_lines = []
        definitions = {item.name: item for item in self.definitions}
        for name, slot_value in self.values.items():
            definition = definitions.get(name)
            if definition:
                collected_lines.append(f"- {definition.definition}：{slot_value.value}")
        missing = self.missing_definitions()[:max_missing]
        missing_lines = [f"- {item.name}：{item.definition}" for item in missing]
        collected_text = "\n".join(collected_lines) if collected_lines else "- 暂无"
        missing_text = "\n".join(missing_lines) if missing_lines else "- 必要槽位已收集完成"
        return (
            "【内部问诊槽位状态，不要向患者朗读本段】\n"
            f"当前为{self.visit_type}，第{self.turn_id}轮。\n"
            f"已确认信息：\n{collected_text}\n"
            f"接下来仍需收集：\n{missing_text}\n"
            "回复规则：先回应患者刚才的内容；从缺失项中按临床相关性一次自然追问1至3项；"
            "不要重复询问已确认信息。必要槽位收集完后进行简短总结并请患者核对，不要擅自诊断或开方。"
        )

    def snapshot(self) -> Dict[str, Any]:
        active = {item.name for item in self.active_definitions()}
        missing = {item.name for item in self.missing_definitions()}
        slots = []
        for definition in self.definitions:
            item: Dict[str, Any] = {
                "name": definition.name,
                "group": definition.group,
                "definition": definition.definition,
                "required": definition.required,
                "condition": definition.condition or None,
                "active": definition.name in active,
                "status": "filled" if definition.name in self.values else (
                    "missing" if definition.name in missing else "inactive"
                ),
            }
            if definition.name in self.values:
                item.update(self.values[definition.name].as_dict())
            slots.append(item)
        return {
            "visit_type": self.visit_type,
            "turn_id": self.turn_id,
            "last_user_text": self.last_user_text,
            "last_error": self.last_error,
            "filled_count": len(self.values),
            "missing_count": len(missing),
            "complete": not missing,
            "signals": dict(self.signals),
            "slots": slots,
            "history": list(self.history),
        }
