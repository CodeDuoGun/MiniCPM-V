#!/usr/bin/env python3
"""Concurrent latency test for the compatible HTTP streaming endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time

import httpx
import numpy as np
import soundfile as sf


def silence_wav(duration_ms: int = 200) -> str:
    output = io.BytesIO()
    sf.write(output, np.zeros(16000 * duration_ms // 1000, dtype="float32"), 16000, format="WAV")
    return base64.b64encode(output.getvalue()).decode("ascii")


async def worker(client: httpx.AsyncClient, base_url: str, worker_id: int, rounds: int, audio: str):
    latencies = []
    failures = []
    uid = f"load-{worker_id}"
    headers = {"uid": uid}
    init = {
        "messages": [{"role": "user", "content": [{"type": "options", "options": {"visit_type": "初诊"}}]}]
    }
    response = await client.post(f"{base_url}/api/v1/init_options", json=init, headers=headers)
    if response.status_code != 200:
        return latencies, [f"init:{response.status_code}"]
    payload = {
        "messages": [{"role": "user", "content": [{
            "type": "input_audio",
            "input_audio": {"data": audio, "format": "wav", "transcript": "最近睡眠不太好", "end_of_turn": True},
        }]}]
    }
    for _ in range(rounds):
        started = time.perf_counter()
        response = await client.post(f"{base_url}/api/v1/stream", json=payload, headers=headers)
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code == 200:
            latencies.append(elapsed)
        else:
            failures.append(str(response.status_code))
    await client.post(f"{base_url}/api/v1/session/close", headers=headers)
    return latencies, failures


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * ratio))]


async def run(args):
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        results = await asyncio.gather(*[
            worker(client, args.base_url.rstrip("/"), index, args.rounds, silence_wav(args.chunk_ms))
            for index in range(args.concurrency)
        ])
    latencies = [value for values, _ in results for value in values]
    failures = [value for _, values in results for value in values]
    report = {
        "requests": len(latencies) + len(failures),
        "success": len(latencies),
        "failures": len(failures),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies) if latencies else 0,
        "failure_codes": {code: failures.count(code) for code in sorted(set(failures))},
    }
    print(json.dumps(report, indent=2))
    if failures or report["p95_ms"] > args.max_p95_ms:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:32560")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-p95-ms", type=float, default=2000)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

