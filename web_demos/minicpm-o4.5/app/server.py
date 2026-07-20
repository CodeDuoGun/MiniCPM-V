from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from .context import ConsultationContext
from .image_analysis import (
    SCENES,
    SOURCES,
    ImageAnalyzer,
    QiniuImageStorage,
    decode_image as decode_uploaded_image,
    make_image_record,
)
from .persistence import ConsultationStore, utc_now
from .protocol import InputChunk, encode_audio, parse_payload
from .runtime import MiniCPMO45Runtime
from .settings import settings


logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("minicpmo45.tcm")

REQUESTS = Counter("minicpmo45_requests_total", "Input chunks", ["transport", "status"])
CHUNK_LATENCY = Histogram("minicpmo45_chunk_seconds", "End-to-end model chunk latency")
MODEL_LATENCY = Histogram("minicpmo45_model_seconds", "GPU model chunk latency")
ACTIVE_SESSIONS = Gauge("minicpmo45_active_sessions", "Active consultation sessions")
INTERRUPTS = Counter("minicpmo45_interrupts_total", "Client requested interruptions")


@dataclass
class Session:
    uid: str
    context: ConsultationContext
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    options: dict[str, Any] = field(default_factory=dict)
    started: bool = False
    assistant_parts: list[str] = field(default_factory=list)
    pending_transcript: str = ""
    input_audio_parts: list[np.ndarray] = field(default_factory=list)
    consultation_id: str = ""

    def touch(self) -> None:
        self.last_seen = time.time()


class SessionRegistry:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.lock = asyncio.Lock()

    async def get(self, uid: str, create: bool = True) -> Session:
        async with self.lock:
            now = time.time()
            expired = [
                key for key, item in self.sessions.items()
                if now - item.last_seen > settings.session_idle_seconds
            ]
            for key in expired:
                self.sessions.pop(key, None)
            if uid in self.sessions:
                session = self.sessions[uid]
                session.touch()
                ACTIVE_SESSIONS.set(len(self.sessions))
                return session
            if not create:
                raise KeyError(uid)
            if len(self.sessions) >= settings.max_active_sessions:
                raise RuntimeError("all model session slots are busy")
            context = ConsultationContext(settings)
            context.reset()
            session = Session(uid=uid, context=context)
            self.sessions[uid] = session
            ACTIVE_SESSIONS.set(len(self.sessions))
            return session

    async def remove(self, uid: str) -> None:
        async with self.lock:
            self.sessions.pop(uid, None)
            ACTIVE_SESSIONS.set(len(self.sessions))


runtime = MiniCPMO45Runtime(settings)
consultation_store = ConsultationStore(settings)
image_storage = QiniuImageStorage(settings)
image_analyzer = ImageAnalyzer(settings)
registry = SessionRegistry()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.max_active_sessions != 1:
        raise RuntimeError("one duplex model process must use MAX_ACTIVE_SESSIONS=1; scale with replicas")
    if settings.load_model and not runtime.loaded:
        raise RuntimeError("LOAD_MODEL=true but MiniCPM-o 4.5 failed to load")
    logger.info("service ready: %s", runtime.capabilities())
    yield


app = FastAPI(
    title="MiniCPM-o 4.5 TCM Realtime Service",
    version="4.5.0",
    lifespan=lifespan,
)


def rollout_enabled(uid: str) -> bool:
    percent = max(0, min(100, settings.rollout_percent))
    digest = hashlib.sha256(f"{settings.rollout_salt}:{uid}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return bucket < percent


def require_uid(uid: str | None) -> str:
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid")
    return uid


def _pcm_bytes(audio: np.ndarray | None) -> bytes | None:
    if audio is None:
        return None
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


async def configure_session(session: Session, options: dict[str, Any]) -> None:
    previous_consultation_id = session.consultation_id
    session.options.update(options)
    session.consultation_id = str(
        session.options.get("consultation_id") or previous_consultation_id or session.uid
    ).strip()
    if not session.started or session.consultation_id != previous_consultation_id:
        session.context.reset(session.options)
        persisted = await asyncio.to_thread(consultation_store.get, session.consultation_id)
        if persisted:
            session.context.restore(persisted)
    prompt = session.context.build_prompt()
    await asyncio.to_thread(runtime.prepare, session.uid, prompt)
    session.started = True
    await asyncio.to_thread(
        consultation_store.save_context,
        session.consultation_id,
        session.uid,
        session.context.profile,
        session.context.snapshot(),
    )


async def process_chunk(session: Session, chunk: InputChunk) -> dict[str, Any]:
    started = time.perf_counter()
    if chunk.options or not session.started:
        await configure_session(session, chunk.options)

    if chunk.transcript:
        # Browser ASR commonly sends the latest complete partial sentence on every
        # chunk. Keep the latest value instead of appending duplicated fragments.
        session.pending_transcript = chunk.transcript
    if chunk.audio is not None:
        session.input_audio_parts.append(chunk.audio)
        max_samples = settings.max_turn_audio_seconds * 16000
        while len(session.input_audio_parts) > 1 and sum(item.size for item in session.input_audio_parts) > max_samples:
            session.input_audio_parts.pop(0)
    model_started = time.perf_counter()
    result = await asyncio.to_thread(runtime.process_duplex_chunk, chunk.audio, chunk.frames)
    MODEL_LATENCY.observe(time.perf_counter() - model_started)

    text = str(result.get("text") or "")
    if text:
        session.assistant_parts.append(text)
    ended = bool(result.get("end_of_turn") or chunk.end_of_turn)
    transcript = ""
    if ended:
        full_audio = np.concatenate(session.input_audio_parts) if session.input_audio_parts else None
        transcript = await asyncio.to_thread(
            session.context.update_user,
            session.pending_transcript,
            _pcm_bytes(full_audio),
        )
        assistant_text = "".join(session.assistant_parts).strip()
        session.context.record_assistant(assistant_text)
        session.assistant_parts.clear()
        session.pending_transcript = ""
        session.input_audio_parts.clear()
        # Refreshing at a turn boundary intentionally trades acoustic KV continuity for
        # deterministic slot/history injection. It never happens in the middle of speech.
        await asyncio.to_thread(runtime.prepare, session.uid, session.context.build_prompt())
        await asyncio.to_thread(
            consultation_store.save_context,
            session.consultation_id or session.uid,
            session.uid,
            session.context.profile,
            session.context.snapshot(),
        )

    sampling_rate = int(result.get("sampling_rate") or 24000)
    response = {
        "id": session.uid,
        "response_id": str(uuid.uuid4()),
        "event": "response.chunk",
        "timestamp": chunk.timestamp,
        "transcript": transcript,
        "choices": [{
            "role": "assistant",
            "audio": encode_audio(result.get("audio_waveform"), sampling_rate),
            "text": text,
            "is_listen": bool(result.get("is_listen", False)),
            "finish_reason": "done" if ended else "processing",
        }],
        "slots": session.context.snapshot() if ended else None,
        "model": settings.model_id,
        "mode": settings.model_mode,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    session.touch()
    CHUNK_LATENCY.observe(time.perf_counter() - started)
    await session.output_queue.put(response)
    return response


async def cancel_session(session: Session) -> None:
    """Discard an interrupted turn and reset the duplex model to saved context."""
    INTERRUPTS.inc()
    session.assistant_parts.clear()
    session.pending_transcript = ""
    session.input_audio_parts.clear()
    # The legacy frontend aborts its current SSE request after /stop. Replacing the
    # queue prevents chunks generated just before the interruption from leaking into
    # the next SSE connection.
    session.output_queue = asyncio.Queue()
    if session.started:
        await asyncio.to_thread(runtime.prepare, session.uid, session.context.build_prompt())


@app.get("/health")
@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    return {
        "status": "OK",
        **runtime.capabilities(),
        "integrations": {
            "redis": consultation_store.configured,
            "qiniu": image_storage.configured,
            "vision_vlm": image_analyzer.configured,
        },
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    ok = runtime.loaded or not settings.load_model
    return JSONResponse({"ready": ok, **runtime.capabilities()}, status_code=200 if ok else 503)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/rollout/{uid}")
async def rollout(uid: str) -> dict[str, Any]:
    return {"uid": uid, "enabled": rollout_enabled(uid), "percent": settings.rollout_percent}


@app.post("/init_options")
@app.post("/api/v1/init_options")
async def init_options(request: Request, uid: str | None = Header(None)) -> JSONResponse:
    session = await registry.get(require_uid(uid))
    chunk = parse_payload(await request.json())
    await configure_session(session, chunk.options)
    return JSONResponse({
        "id": session.uid,
        "choices": {"role": "assistant", "content": "4.5", "finish_reason": "done"},
        "consultation_id": session.consultation_id,
        "next_media_request": session.context.next_media_request(),
    })


@app.post("/stream")
@app.post("/api/v1/stream")
async def stream(request: Request, uid: str | None = Header(None)) -> JSONResponse:
    session = await registry.get(require_uid(uid))
    try:
        result = await process_chunk(session, parse_payload(await request.json()))
        REQUESTS.labels("http", "ok").inc()
        return JSONResponse({
            "id": session.uid,
            "choices": {"role": "assistant", "content": "success", "finish_reason": result["choices"][0]["finish_reason"]},
        })
    except Exception as exc:
        REQUESTS.labels("http", "error").inc()
        logger.exception("HTTP stream failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.websocket("/ws/stream")
@app.websocket("/ws/api/v1/stream")
async def websocket_stream(websocket: WebSocket, uid: str | None = Query(None)) -> None:
    if not uid:
        await websocket.close(code=4400, reason="Missing uid")
        return
    try:
        session = await registry.get(uid)
    except RuntimeError as exc:
        await websocket.close(code=4429, reason=str(exc))
        return
    await websocket.accept()
    try:
        while True:
            payload = json.loads(await websocket.receive_text())
            if payload.get("event") == "response.cancel":
                await cancel_session(session)
                await websocket.send_json({"event": "response.cancelled", "id": uid})
                continue
            result = await process_chunk(session, parse_payload(payload))
            REQUESTS.labels("websocket", "ok").inc()
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        REQUESTS.labels("websocket", "error").inc()
        logger.exception("WebSocket stream failed")
        await websocket.close(code=1011, reason=str(exc)[:120])


async def _sse(session: Session):
    while True:
        result = await session.output_queue.get()
        yield f"event: message\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"
        if result["choices"][0]["finish_reason"] == "done":
            return


@app.post("/completions")
@app.post("/api/v1/completions")
async def completions(uid: str | None = Header(None)) -> StreamingResponse:
    session = await registry.get(require_uid(uid))
    return StreamingResponse(_sse(session), media_type="text/event-stream")


@app.post("/stop")
@app.post("/api/v1/stop")
async def stop_response(uid: str | None = Header(None)) -> JSONResponse:
    """Compatibility endpoint used by the MiniCPM-o 2.6 HTTP/SSE frontend."""
    requested_uid = require_uid(uid)
    try:
        session = await registry.get(requested_uid, create=False)
    except KeyError:
        # The legacy UI probes /stop when a page is mounted, then creates a new UID
        # when the call starts. A probe must not occupy the single model session.
        session = None
    if session is not None:
        await cancel_session(session)
    return JSONResponse({
        "id": requested_uid,
        "choices": {
            "role": "assistant",
            "content": "success",
            "finish_reason": "stop",
        },
    })


@app.get("/api/v1/slots")
async def get_slots(uid: str | None = Header(None)) -> JSONResponse:
    session = await registry.get(require_uid(uid))
    return JSONResponse(session.context.snapshot())


@app.post("/api/v1/slots/update")
async def update_slots(request: Request, uid: str | None = Header(None)) -> JSONResponse:
    session = await registry.get(require_uid(uid))
    payload = await request.json()
    transcript = str(payload.get("transcript") or payload.get("input_text") or "").strip()
    with session.context.lock:
        if transcript:
            session.context.slots.process_user_turn(transcript)
        updates = payload.get("updates") or []
        if isinstance(updates, list):
            session.context.slots.apply_updates(updates, source="api")
        if payload.get("assistant_text"):
            session.context.slots.record_assistant(str(payload["assistant_text"]))
    await asyncio.to_thread(runtime.prepare, session.uid, session.context.build_prompt())
    await asyncio.to_thread(
        consultation_store.save_context,
        session.consultation_id or session.uid,
        session.uid,
        session.context.profile,
        session.context.snapshot(),
    )
    return JSONResponse(session.context.snapshot())


@app.post("/api/v1/slots/reset")
async def reset_slots(request: Request, uid: str | None = Header(None)) -> JSONResponse:
    session = await registry.get(require_uid(uid))
    payload = await request.json()
    profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    if "visit_type" in payload:
        profile["visit_type"] = payload["visit_type"]
    session.options.update(profile)
    session.context.reset(session.options)
    await asyncio.to_thread(runtime.prepare, session.uid, session.context.build_prompt())
    await asyncio.to_thread(
        consultation_store.save_context,
        session.consultation_id or session.uid,
        session.uid,
        session.context.profile,
        session.context.snapshot(),
    )
    return JSONResponse(session.context.snapshot())


@app.post("/api/v1/reports/analyze")
@app.post("/api/v1/images/analyze")
async def analyze_consultation_image(request: Request, uid: str | None = Header(None)) -> JSONResponse:
    """Quality-check, analyze and persist a tongue, lesion or report image."""
    session = await registry.get(require_uid(uid))
    payload = await request.json()
    legacy_report_request = request.url.path == "/api/v1/reports/analyze"
    scene = str(payload.get("scene") or ("report" if legacy_report_request else "")).strip().lower()
    source = str(payload.get("source") or "manual_upload").strip().lower()
    consultation_id = str(payload.get("consultation_id") or session.consultation_id or session.uid).strip()
    image_data = payload.get("image_data")
    mime_type = str(payload.get("mime_type") or "image/jpeg").lower()
    if scene not in SCENES:
        raise HTTPException(status_code=400, detail="scene must be tongue, lesion or report")
    if source not in SOURCES:
        raise HTTPException(status_code=400, detail="source must be assistant_requested or manual_upload")
    if not isinstance(image_data, str) or not image_data:
        raise HTTPException(status_code=400, detail="Missing base64 image_data")
    if session.started and session.consultation_id and consultation_id != session.consultation_id:
        raise HTTPException(status_code=409, detail="consultation_id does not match the active session")
    if not consultation_store.configured:
        raise HTTPException(status_code=503, detail="consultation Redis is not configured")
    if not image_storage.configured:
        raise HTTPException(status_code=503, detail="Qiniu image storage is not configured")
    if not image_analyzer.configured:
        raise HTTPException(status_code=503, detail="vision VLM is not configured")
    if not session.started:
        await configure_session(session, {"consultation_id": consultation_id})
    try:
        raw, _ = decode_uploaded_image(image_data, mime_type)
        upload = await asyncio.to_thread(image_storage.upload, raw, consultation_id, scene, mime_type)
        quality = await asyncio.to_thread(image_analyzer.quality_check, scene, image_data, mime_type)
        analysis = None
        if quality["passed"]:
            analysis = await asyncio.to_thread(image_analyzer.analyze, scene, image_data, mime_type)
        record = make_image_record(
            consultation_id=consultation_id,
            scene=scene,
            source=source,
            upload=upload,
            quality=quality,
            analysis=analysis,
            model=settings.vision_vlm_model,
        )
        await asyncio.to_thread(consultation_store.append_image, consultation_id, session.uid, record)
        if analysis is not None:
            session.context.record_visual_observation(record)
            await asyncio.to_thread(runtime.prepare, session.uid, session.context.build_prompt())
            await asyncio.to_thread(
                consultation_store.save_context,
                consultation_id,
                session.uid,
                session.context.profile,
                session.context.snapshot(),
            )
        return JSONResponse({
            "id": session.uid,
            "consultation_id": consultation_id,
            "observation": record,
            # Compatibility fields consumed by the legacy report uploader.
            "analysis": analysis or {},
            "disclaimer": "结果仅用于预问诊信息整理，不能替代医生诊断。",
            "next_media_request": session.context.next_media_request(),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("image analysis failed")
        if "upload" in locals():
            failed_record = {
                "observation_id": str(uuid.uuid4()),
                "consultation_id": consultation_id,
                "scene": scene,
                "source": source,
                **upload,
                "status": "failed",
                "quality_llm_result": locals().get("quality"),
                "analysis_llm_result": None,
                "error": type(exc).__name__,
                "model": settings.vision_vlm_model,
                "created_at": utc_now(),
            }
            await asyncio.to_thread(
                consultation_store.append_image, consultation_id, session.uid, failed_record
            )
        raise HTTPException(status_code=502, detail="image analysis service failed") from exc


@app.get("/api/v1/consultations/{consultation_id}")
async def get_consultation(consultation_id: str, uid: str | None = Header(None)) -> JSONResponse:
    owner_uid = require_uid(uid)
    document = await asyncio.to_thread(consultation_store.get, consultation_id)
    if not document or document.get("uid") != owner_uid:
        raise HTTPException(status_code=404, detail="consultation not found")
    return JSONResponse(document)


@app.post("/api/v1/session/close")
async def close_session(uid: str | None = Header(None)) -> dict[str, Any]:
    owner_uid = require_uid(uid)
    try:
        session = await registry.get(owner_uid, create=False)
    except KeyError:
        return {"closed": True}
    await asyncio.to_thread(
        consultation_store.save_context,
        session.consultation_id or session.uid,
        session.uid,
        session.context.profile,
        session.context.snapshot(),
    )
    await registry.remove(owner_uid)
    return {"closed": True}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
