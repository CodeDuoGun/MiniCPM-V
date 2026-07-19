#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    path = Path(args.dataset).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = set()
    stats = Counter()
    errors = []
    for index, sample in enumerate(data):
        sample_id = str(sample.get("id") or index)
        if sample_id in ids:
            errors.append(f"duplicate id: {sample_id}")
        ids.add(sample_id)
        messages = sample.get("messages") or []
        images = sample.get("images") or []
        placeholders = sum(str(item.get("content") or "").count("<image>") for item in messages)
        if placeholders != len(images):
            errors.append(f"{sample_id}: {placeholders} placeholders != {len(images)} images")
        for image in images:
            if not Path(image).is_file():
                errors.append(f"{sample_id}: missing image {image}")
        stats["samples"] += 1
        stats["images"] += len(images)
        stats["turns"] += len(messages)
    if errors:
        raise SystemExit("\n".join(errors[:50]))
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

