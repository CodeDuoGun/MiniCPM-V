#!/usr/bin/env bash
set -euo pipefail

# MiniCPM-o 2.6 医疗实时问诊混合数据 LoRA
# 硬件目标：单卡 NVIDIA A100 80GB
# 数据范围：舌象、面象、患处视觉样本 + 已清洗的初诊/复诊多轮问诊样本
# 不包含：检查报告上传数据 report_upload_*.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# 建议正式训练时把 MODEL 指向本地模型的绝对路径，避免训练时下载失败。
MODEL="${MODEL:-openbmb/MiniCPM-o-2_6}"
DATA="${DATA:-$PROJECT_DIR/data/wuweiping_vlm_pretriage/train.json}"
EVAL_DATA="${EVAL_DATA:-$PROJECT_DIR/data/wuweiping_vlm_pretriage/validation.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/tcm_minicpmo_lora_a100_80g}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
MAX_SLICE_NUMS="${MAX_SLICE_NUMS:-4}"
MAX_STEPS="${MAX_STEPS:-600}"
WARMUP_STEPS="${WARMUP_STEPS:-30}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
REPORT_TO="${REPORT_TO:-none}"

if [[ ! -f "$DATA" ]]; then
    echo "训练集不存在：$DATA" >&2
    exit 1
fi

if [[ ! -f "$EVAL_DATA" ]]; then
    echo "验证集不存在：$EVAL_DATA" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "EVAL_DATA=$EVAL_DATA"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "effective_batch_size=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"

cd "$SCRIPT_DIR"

torchrun \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port "${MASTER_PORT:-6002}" \
    finetune.py \
    --model_name_or_path "$MODEL" \
    --llm_type qwen \
    --data_path "$DATA" \
    --eval_data_path "$EVAL_DATA" \
    --remove_unused_columns false \
    --label_names labels \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --tf32 true \
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
    --max_slice_nums "$MAX_SLICE_NUMS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --warmup_steps "$WARMUP_STEPS" \
    --lr_scheduler_type cosine \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 5 \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "$REPORT_TO"
