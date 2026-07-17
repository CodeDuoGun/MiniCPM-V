#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6002}"

MODEL="${MODEL:-openbmb/MiniCPM-o-2_6}"
DATA="${DATA:-$PROJECT_DIR/data/wuweiping_vlm_pretriage/train.json}"
EVAL_DATA="${EVAL_DATA:-$PROJECT_DIR/data/wuweiping_vlm_pretriage/validation.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/tcm_minicpmo_lora_stage2_vision}"
LOGGING_DIR="${LOGGING_DIR:-$OUTPUT_DIR/logs}"
export TENSORBOARD_LOGGING_DIR="${TENSORBOARD_LOGGING_DIR:-$LOGGING_DIR}"

LLM_TYPE="qwen"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
"

cd "$SCRIPT_DIR"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path "$MODEL" \
    --llm_type "$LLM_TYPE" \
    --data_path "$DATA" \
    --eval_data_path "$EVAL_DATA" \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_vision false \
    --tune_llm false \
    --use_lora true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --max_slice_nums "${MAX_SLICE_NUMS:-4}" \
    --max_steps "${MAX_STEPS:-3000}" \
    --eval_steps "${EVAL_STEPS:-50}" \
    --save_steps "${SAVE_STEPS:-50}" \
    --output_dir "$OUTPUT_DIR" \
    --logging_strategy "steps" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_total_limit 5 \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_steps "${WARMUP_STEPS:-100}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "${REPORT_TO:-none}"
