#!/usr/bin/env python3
"""Patch MiniCPM-o 4.5 remote code for gradient-checkpointed training."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


START = '        bs = len(data["input_ids"])'
END = "        return vllm_embedding, vision_hidden_states"
PATCH_MARKER = "updated_vllm_embedding = torch.empty_like(vllm_embedding)"
REPLACEMENT = '''        bs = len(data["input_ids"])
        updated_vllm_embedding = torch.empty_like(vllm_embedding)
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            cur_vllm_emb = vllm_embedding[i]
            if len(cur_vs_hs) > 0:
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)
                    updated_emb = cur_vllm_emb.clone()
                    updated_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                    updated_vllm_embedding[i] = updated_emb
                elif self.training:
                    updated_vllm_embedding[i] = cur_vllm_emb + cur_vs_hs[0].mean() * 0
                else:
                    updated_vllm_embedding[i] = cur_vllm_emb
            else:
                updated_vllm_embedding[i] = cur_vllm_emb
        return updated_vllm_embedding, vision_hidden_states'''


def default_model_files() -> list[Path]:
    cache = Path.home() / ".cache/huggingface"
    patterns = (
        "hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/modeling_minicpmo.py",
        "modules/transformers_modules/openbmb/MiniCPM_hyphen_o_hyphen_4_5/*/modeling_minicpmo.py",
    )
    return sorted({path for pattern in patterns for path in cache.glob(pattern)})


def patch_model_file(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return "already patched"
    if START not in source or END not in source:
        raise RuntimeError(f"无法识别 MiniCPM-o 4.5 get_vllm_embedding 实现: {path}")
    pattern = re.escape(START) + r".*?" + re.escape(END)
    patched, count = re.subn(pattern, REPLACEMENT, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"补丁匹配数量异常 ({count}): {path}")
    backup = path.with_suffix(path.suffix + ".pre-training-fix")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", nargs="*", type=Path)
    args = parser.parse_args()
    files = args.model_file or default_model_files()
    if not files:
        raise SystemExit("未找到 MiniCPM-o 4.5 Hugging Face 缓存，请先完成一次模型加载")
    for path in files:
        print(f"{patch_model_file(path)}: {path}")


if __name__ == "__main__":
    main()
