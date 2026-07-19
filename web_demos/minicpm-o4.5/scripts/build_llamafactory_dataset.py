#!/usr/bin/env python3
"""Convert the existing TCM conversations to LLaMA-Factory ShareGPT multimodal JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
UNSAFE_ASSISTANT = re.compile(
    r"(?:给你开|开点|处方|每日[一二两三四五六七八九十\d]+次|每次\d+|"
    r"口服|外用药改为|酒精调涂|停药|减量|加量|确诊为|诊断为)"
)
DOCUMENT_LEAK = re.compile(r"(?:身份证|手机号|住院号|就诊号|病历号|检查报告显示|报告单上|处方单)")


def resolve_image(value: str, source_file: Path) -> str:
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([source_file.parent / path, PROJECT_ROOT / path])
    else:
        parts = path.parts
        if PROJECT_ROOT.name in parts:
            index = parts.index(PROJECT_ROOT.name)
            candidates.append(PROJECT_ROOT.joinpath(*parts[index + 1 :]))
        if "data" in parts:
            index = parts.index("data")
            candidates.append(PROJECT_ROOT.joinpath(*parts[index:]))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"image not found: {value}")


def convert_sample(sample: dict[str, Any], source_file: Path, strict_medical: bool) -> tuple[dict[str, Any] | None, str]:
    raw_messages = sample.get("messages") or sample.get("conversations")
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        return None, "invalid_messages"
    messages = []
    expected = "user"
    for raw in raw_messages:
        role = ROLE_MAP.get(str(raw.get("role") or raw.get("from") or "").lower())
        content = str(raw.get("content") or raw.get("value") or "").strip()
        if role not in {"user", "assistant", "system"} or not content:
            return None, "invalid_turn"
        if role != "system":
            if role != expected:
                return None, "role_order"
            expected = "assistant" if role == "user" else "user"
        if role == "assistant":
            if DOCUMENT_LEAK.search(content):
                return None, "document_leak"
            if strict_medical and UNSAFE_ASSISTANT.search(content):
                return None, "unsafe_medical_advice"
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        return None, "incomplete_pair"

    images: list[str] = []
    raw_images = sample.get("images", sample.get("image"))
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    elif isinstance(raw_images, dict):
        raw_images = list(raw_images.values())
    for value in raw_images or []:
        images.append(resolve_image(str(value), source_file))
    placeholders = sum(item["content"].count("<image>") for item in messages)
    if images and placeholders == 0:
        # Some legacy multi-view samples stored an image dictionary but omitted the
        # textual placeholders. Prepending one marker per ordered image is lossless.
        first_user = next((item for item in messages if item["role"] == "user"), None)
        if first_user is None:
            return None, "image_without_user"
        first_user["content"] = "\n".join(["<image>"] * len(images) + [first_user["content"]])
        placeholders = len(images)
    if placeholders != len(images):
        return None, "image_placeholder_mismatch"

    fingerprint = hashlib.sha256(
        json.dumps(sample, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result: dict[str, Any] = {
        "id": f"{source_file.parent.name}_{source_file.stem}_{sample.get('id') or ''}_{fingerprint[:10]}",
        "messages": messages,
        "metadata": sample.get("metadata") or {},
    }
    if images:
        result["images"] = images
    return result, "accepted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--allow-medical-advice", action="store_true")
    args = parser.parse_args()

    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_fingerprints: set[str] = set()
    for source_value in args.source:
        source = Path(source_value).expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"source must contain a JSON list: {source}")
        for sample in data:
            fingerprint = hashlib.sha256(
                json.dumps(sample, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if fingerprint in seen_fingerprints:
                counts["duplicate_sample"] += 1
                continue
            seen_fingerprints.add(fingerprint)
            try:
                converted, reason = convert_sample(sample, source, not args.allow_medical_advice)
            except (OSError, TypeError, ValueError):
                converted, reason = None, "missing_image"
            counts[reason] += 1
            if converted:
                output.append(converted)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "sources": args.source,
        "output": str(output_path.resolve()),
        "strict_medical": not args.allow_medical_advice,
        "accepted": len(output),
        "counts": dict(counts),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
