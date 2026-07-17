#!/usr/bin/env python3
"""Validate schema, local images, split leakage, privacy, and safety wording."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from image_integrity import is_complete_image


PII_RE = re.compile(r"(?<!\d)(?:1\d{10}|\d{17}[\dXx]|\d{15})(?!\d)")
UNSAFE_RE = re.compile(r"(?:处方|诊断为|确诊为)\s*[^，。；\n]{1,40}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    ids: set[str] = set()
    image_count = 0
    images_by_split: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        path = args.dataset_dir / f"{split}.json"
        samples = json.loads(path.read_text(encoding="utf-8"))
        images_by_split[split] = set()
        for sample in samples:
            sid = sample.get("id", "")
            if sid in ids:
                errors.append(f"duplicate id: {sid}")
            ids.add(sid)
            conversations = sample.get("conversations", [])
            if [m.get("role") for m in conversations] != ["user", "assistant"]:
                errors.append(f"{sid}: invalid roles")
            image_value = sample.get("image")
            if isinstance(image_value, str):
                image_paths = [image_value]
                expected_placeholders = {"<image>"}
            elif isinstance(image_value, dict):
                image_paths = list(image_value.values())
                expected_placeholders = set(image_value)
            else:
                image_paths = []
                expected_placeholders = set()
                errors.append(f"{sid}: invalid image field")
            user_content = conversations[0].get("content", "") if conversations else ""
            for placeholder in expected_placeholders:
                if user_content.count(placeholder) != 1:
                    errors.append(f"{sid}: placeholder mismatch for {placeholder}")
            found_placeholders = set(re.findall(r"<image(?:_\d+)?>", user_content))
            if found_placeholders != expected_placeholders:
                errors.append(f"{sid}: image placeholder set mismatch")
            blob = json.dumps(sample, ensure_ascii=False)
            if PII_RE.search(blob):
                errors.append(f"{sid}: possible direct identifier")
            answer = conversations[-1].get("content", "") if conversations else ""
            if UNSAFE_RE.search(answer):
                errors.append(f"{sid}: unsafe diagnosis/prescription-like wording")
            if "急诊" not in answer:
                errors.append(f"{sid}: missing escalation wording")
            if not any(
                wording in answer
                for wording in ("不能仅凭图片", "不能据此确诊", "不能单独用于确诊", "不应因继续拍摄或分析")
            ):
                errors.append(f"{sid}: missing image limitation wording")
            for rel in image_paths:
                image_count += 1
                images_by_split[split].add(str(rel))
                image_path = Path(rel)
                if not image_path.is_absolute():
                    image_path = args.dataset_dir / image_path
                if not image_path.is_file():
                    errors.append(f"{sid}: missing image {rel}")
                    continue
                if not is_complete_image(image_path):
                    errors.append(f"{sid}: invalid or incomplete image {rel}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = images_by_split.get(left, set()) & images_by_split.get(right, set())
        if overlap:
            errors.append(f"cross-split image leakage: {left}/{right} share {len(overlap)} images")
    report = {"valid": not errors, "samples": len(ids), "image_references": image_count, "errors": errors[:100]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
