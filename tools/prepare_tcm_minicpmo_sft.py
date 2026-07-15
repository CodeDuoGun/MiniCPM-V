#!/usr/bin/env python3
"""Prepare TCM consultation SFT data for the official MiniCPM-V finetune scripts.

Input can be either:
1. LLaMA-Factory style samples with a "messages" field.
2. Official MiniCPM finetune style samples with a "conversations" field.

Output format:
[
  {
    "id": "...",
    "conversations": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]

For vision-stage SFT, provide --image-manifest. The manifest should be a JSON
object keyed by record id:
{
  "777197": {
    "<image_00>": "/abs/path/tongue.jpg",
    "<image_01>": "/abs/path/face.jpg"
  }
}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from tkinter import NO
from typing import Any, Dict, List, Optional, Tuple


SYSTEM_PREFIX = (
    "你是中医问诊助手，任务是围绕患者主诉、现病史、既往史、用药反应、饮食、睡眠、二便、"
    "寒热汗出、情绪、女性月经/孕产情况等进行谨慎、连续的病情采集。"
    "不要直接替代医生诊断或开方；遇到急症、严重过敏、持续高热、胸痛、呼吸困难、意识异常等情况，"
    "应建议及时线下就医。"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = item.get("conversations") or item.get("messages") or []
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] += "\n" + content
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def build_sample(
    item: Dict[str, Any],
    *,
    sample_index: int,
    add_system_prefix: bool,
    image_manifest: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    conversations = normalize_messages(item)
    if len(conversations) < 2:
        return None, "too_few_turns"
    if conversations[0]["role"] != "user":
        return None, "first_turn_not_user"
    if conversations[-1]["role"] == "user":
        conversations = conversations[:-1]
    if not any(msg["role"] == "assistant" for msg in conversations):
        return None, "no_assistant_target"

    if add_system_prefix:
        conversations[0]["content"] = SYSTEM_PREFIX + "\n\n" + conversations[0]["content"]

    record_id = str(item.get("record_id") or item.get("id") or sample_index)
    sample: Dict[str, Any] = {
        "id": record_id,
        "conversations": conversations,
    }

    if image_manifest is not None:
        image_entry = image_manifest.get(record_id)
        if image_entry is None:
            return None, "missing_image_manifest"
        if isinstance(image_entry, str):
            sample["image"] = image_entry
            if "<image>" not in sample["conversations"][0]["content"]:
                sample["conversations"][0]["content"] = "<image>\n" + sample["conversations"][0]["content"]
        elif isinstance(image_entry, dict):
            sample["image"] = image_entry
            placeholders = [k for k in image_entry.keys() if k.startswith("<image_")]
            missing = [k for k in placeholders if k not in sample["conversations"][0]["content"]]
            if missing:
                sample["conversations"][0]["content"] = "\n".join(missing) + "\n" + sample["conversations"][0]["content"]
        else:
            return None, "bad_image_manifest"

    return sample, None


def split_samples(
    samples: List[Dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    return shuffled[test_n + val_n :], shuffled[test_n : test_n + val_n], shuffled[:test_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MiniCPM-o TCM SFT data.")
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--dataset-name", default="tcm_consult_minicpmo")
    parser.add_argument("--image-manifest", help="Optional record_id -> image path(s) JSON for vision-stage SFT.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--no-system-prefix", action="store_true")
    args = parser.parse_args()

    raw = load_json(Path(args.input))
    print(f"raw:{len(raw)}")
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list.")

    image_manifest = load_json(Path(args.image_manifest)) if args.image_manifest else None
    if image_manifest is not None and not isinstance(image_manifest, dict):
        raise ValueError("--image-manifest must be a JSON object keyed by record id.")

    samples: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            skipped["non_dict"] = skipped.get("non_dict", 0) + 1
            continue
        sample, reason = build_sample(
            item,
            sample_index=idx,
            add_system_prefix=not args.no_system_prefix,
            image_manifest=image_manifest,
        )
        print(f"sample is None: {sample is None}")
        if sample is None:
            skipped[reason or "unknown"] = skipped.get(reason or "unknown", 0) + 1
            continue

        samples.append(sample)

    train, val, test = split_samples(
        samples,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    output_dir = Path(args.output_dir)
    prefix = args.dataset_name or "tcm_consult_minicpmo"
    print(f"***samples {len(samples)}")
    write_json(output_dir / f"{prefix}.json", samples)
    write_json(output_dir / f"{prefix}_train.json", train)
    write_json(output_dir / f"{prefix}_val.json", val)
    write_json(output_dir / f"{prefix}_test.json", test)
    write_json(
        output_dir / f"{prefix}_report.json",
        {
            "input": args.input,
            "image_manifest": args.image_manifest,
            "total": len(samples),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "skipped": skipped,
        },
    )

    print(f"wrote {len(samples)} samples to {output_dir}")
    print(f"train={len(train)} val={len(val)} test={len(test)} skipped={skipped}")


if __name__ == "__main__":
    main()
