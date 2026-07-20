from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from .settings import Settings


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsultationStore:
    """Stores one complete consultation document per consultation id in Redis."""

    def __init__(self, config: Settings):
        self.config = config
        self.client: Any = None
        self.lock = threading.RLock()
        if config.redis_url:
            try:
                import redis

                self.client = redis.Redis.from_url(config.redis_url, decode_responses=True)
                self.client.ping()
            except Exception as exc:
                self.client = None
                logger.warning("consultation Redis is unavailable: %s", exc)

    @property
    def configured(self) -> bool:
        return self.client is not None

    def _key(self, consultation_id: str) -> str:
        return f"{self.config.redis_key_prefix}:{consultation_id}"

    def get(self, consultation_id: str) -> dict[str, Any] | None:
        if not self.client:
            return None
        raw = self.client.get(self._key(consultation_id))
        if not raw:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None

    def save_context(
        self,
        consultation_id: str,
        uid: str,
        profile: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.client:
            return {}
        with self.lock:
            document = self.get(consultation_id) or {
                "consultation_id": consultation_id,
                "uid": uid,
                "created_at": utc_now(),
                "images": [],
            }
            document.update({
                "uid": uid,
                "patient": dict(profile),
                "condition": {
                    "visit_type": snapshot.get("visit_type"),
                    "turn_id": snapshot.get("turn_id"),
                    "signals": snapshot.get("signals") or {},
                    "slots": snapshot.get("slots") or [],
                    "visual_observations": snapshot.get("visual_observations") or [],
                },
                "conversation": snapshot.get("history") or [],
                "updated_at": utc_now(),
            })
            self._write(consultation_id, document)
            return document

    def append_image(self, consultation_id: str, uid: str, record: dict[str, Any]) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("consultation Redis is not configured")
        with self.lock:
            document = self.get(consultation_id) or {
                "consultation_id": consultation_id,
                "uid": uid,
                "created_at": utc_now(),
                "patient": {},
                "condition": {},
                "conversation": [],
                "images": [],
            }
            images = document.setdefault("images", [])
            images.append(record)
            document["updated_at"] = utc_now()
            self._write(consultation_id, document)
            return document

    def _write(self, consultation_id: str, document: dict[str, Any]) -> None:
        payload = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        ttl = max(0, int(self.config.consultation_ttl_seconds))
        if ttl:
            self.client.setex(self._key(consultation_id), ttl, payload)
        else:
            self.client.set(self._key(consultation_id), payload)

