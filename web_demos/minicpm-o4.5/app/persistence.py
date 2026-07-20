from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import Settings


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsultationStore:
    """Stores one complete consultation document per consultation id on disk."""

    def __init__(self, config: Settings):
        self.config = config
        self.root = config.consultation_store_path
        self.lock = threading.RLock()
        self._configured = False
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._configured = self.root.is_dir() and os.access(self.root, os.R_OK | os.W_OK)
        except OSError as exc:
            logger.warning("local consultation store is unavailable: %s", exc)

    @property
    def configured(self) -> bool:
        return self._configured

    def _path(self, consultation_id: str) -> Path:
        # Keep filenames recognizable while preventing path traversal and collisions.
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", consultation_id).strip("-.")[:64]
        digest = hashlib.sha256(consultation_id.encode("utf-8")).hexdigest()[:12]
        return self.root / f"{slug or 'consultation'}-{digest}.json"

    def get(self, consultation_id: str) -> dict[str, Any] | None:
        if not self.configured:
            return None
        path = self._path(consultation_id)
        with self.lock:
            if not path.is_file():
                return None
            ttl = max(0, int(self.config.consultation_ttl_seconds))
            if ttl and datetime.now(timezone.utc).timestamp() - path.stat().st_mtime >= ttl:
                path.unlink(missing_ok=True)
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("failed to read consultation %s: %s", consultation_id, exc)
                return None
            return value if isinstance(value, dict) else None

    def save_context(
        self,
        consultation_id: str,
        uid: str,
        profile: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.configured:
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
        if not self.configured:
            raise RuntimeError("local consultation store is not configured")
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
        path = self._path(consultation_id)
        temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
        payload = json.dumps(document, ensure_ascii=False, indent=2)
        try:
            with temporary.open("w", encoding="utf-8") as output:
                output.write(payload)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            path.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
