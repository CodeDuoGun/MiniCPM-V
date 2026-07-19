#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--max-image-pixels", type=int, default=178_956_970)
    parser.add_argument("--skip-image-files", action="store_true")
    parser.add_argument(
        "--image-path-map",
        metavar="SOURCE=TARGET",
        help="replace an image path prefix, for example a server project root with a local checkout",
    )
    args = parser.parse_args()
    path = Path(args.dataset).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    image_path_map = None
    if args.image_path_map:
        if "=" not in args.image_path_map:
            parser.error("--image-path-map must use SOURCE=TARGET")
        source, target = args.image_path_map.split("=", 1)
        image_path_map = (source.rstrip("/"), target.rstrip("/"))
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
        if args.skip_image_files:
            stats["samples"] += 1
            stats["images"] += len(images)
            stats["turns"] += len(messages)
            continue
        for image in images:
            resolved_image = str(image)
            if image_path_map and (
                resolved_image == image_path_map[0]
                or resolved_image.startswith(image_path_map[0] + "/")
            ):
                resolved_image = image_path_map[1] + resolved_image[len(image_path_map[0]) :]
            image_path = Path(resolved_image)
            if not image_path.is_file():
                errors.append(f"{sample_id}: missing image {image_path}")
                continue
            try:
                Image.MAX_IMAGE_PIXELS = None
                with Image.open(image_path) as handle:
                    width, height = handle.size
                    if width * height > args.max_image_pixels:
                        errors.append(f"{sample_id}: image too large {width}x{height} {image_path}")
                        continue
                    # Image.open() is lazy; force full pixel decoding so truncated
                    # files fail here instead of inside LLaMA-Factory workers.
                    handle.load()
            except Exception as exc:
                errors.append(
                    f"{sample_id}: invalid image {type(exc).__name__}: {exc}: {image_path}"
                )
        stats["samples"] += 1
        stats["images"] += len(images)
        stats["turns"] += len(messages)
    if errors:
        raise SystemExit("\n".join(errors[:50]))
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
