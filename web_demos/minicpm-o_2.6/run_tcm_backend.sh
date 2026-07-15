#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-openbmb/MiniCPM-o-2_6}"
ADAPTER="${ADAPTER:-$PROJECT_DIR/outputs/tcm_minicpmo_lora_stage1_text}"
PORT="${PORT:-32550}"

cd "$SCRIPT_DIR"

python model_server.py \
    --port "$PORT" \
    --model "$MODEL" \
    --adapter "$ADAPTER"
