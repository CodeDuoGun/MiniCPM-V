#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

python3 "$PROJECT_DIR/web_demos/minicpm-o4.5/scripts/validate_llamafactory_dataset.py" \
  "$PROJECT_DIR/web_demos/minicpm-o4.5/data/tcm_o45_train.json"
python3 "$PROJECT_DIR/web_demos/minicpm-o4.5/scripts/validate_llamafactory_dataset.py" \
  "$PROJECT_DIR/web_demos/minicpm-o4.5/data/tcm_o45_validation.json"
python3 "$PROJECT_DIR/web_demos/minicpm-o4.5/scripts/patch_minicpmo45_training.py"

llamafactory-cli train "$SCRIPT_DIR/minicpmo45_lora_sft.yaml"
