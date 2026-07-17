#!/usr/bin/env python3
"""Merge a MiniCPM-o LoRA adapter into the base model for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge MiniCPM-o LoRA adapter.")
    parser.add_argument("--base-model", default="openbmb/MiniCPM-o-2_6")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        init_audio=False,
        init_tts=False,
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        torch_dtype=dtype,
        torch_device="cpu",
        low_cpu_mem_usage=True,
    )
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
