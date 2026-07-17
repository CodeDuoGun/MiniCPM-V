#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-openbmb/MiniCPM-o-2_6}"
ADAPTER="${ADAPTER:-$PROJECT_DIR/outputs/tcm_minicpmo_lora_stage1_text}"
PORT="${PORT:-32550}"
HUMAN_PORT="${HUMAN_PORT:-8010}"
HUMAN_SERVICE_URL="${HUMAN_SERVICE_URL:-http://127.0.0.1:${HUMAN_PORT}/aihuman}"
START_HUMAN_SERVICE="${START_HUMAN_SERVICE:-0}"
HUMAN_PYTHON="${HUMAN_PYTHON:-python}"

HUMAN_PID=""
cleanup() {
    if [[ -n "$HUMAN_PID" ]]; then
        kill "$HUMAN_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "$START_HUMAN_SERVICE" == "1" ]]; then
    (
        cd "$PROJECT_DIR/human"
        exec "$HUMAN_PYTHON" app/main.py --listenport "$HUMAN_PORT"
    ) &
    HUMAN_PID=$!
fi

cd "$SCRIPT_DIR"

python model_server.py \
    --port "$PORT" \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --human-service-url "$HUMAN_SERVICE_URL"
