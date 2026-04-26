#!/bin/bash
# Two-task continual learning smoke test.
# Task 1: ScienceQA (5 steps, saves adapter_model.bin).
# Task 2: ScienceQA again (5 steps, loads Task 1 adapter via --previous_task_model_path).
# Purpose: verify save_trained_model + load_model_from_previous_task round-trip works.
set -euo pipefail

PROMPT_VERSION=v1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

DATA_PATH="${COIN_INSTR_ROOT}/ScienceQA/train.json"
TASK1_OUTPUT="${COIN_OUTPUT_ROOT}/continual_smoke_task1"
TASK2_OUTPUT="${COIN_OUTPUT_ROOT}/continual_smoke_task2"

require_path "${COIN_BASE_MODEL}" "base model"
require_path "${COIN_PRETRAIN_PROJECTOR}" "mm projector"
require_path "${COIN_VISION_TOWER}" "vision tower"
require_path "${DATA_PATH}" "ScienceQA train json"

COMMON_ARGS=(
    --deepspeed ./scripts/zero3_offload.json
    --moe_moka_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5
    --expert_num 4
    --model_name_or_path "${COIN_BASE_MODEL}"
    --pretrain_mm_mlp_adapter "${COIN_PRETRAIN_PROJECTOR}"
    --version "${PROMPT_VERSION}"
    --data_path "${DATA_PATH}"
    --image_folder "${COIN_IMAGE_ROOT}"
    --vision_tower "${COIN_VISION_TOWER}"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 True
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 1
    --evaluation_strategy "no"
    --save_strategy "no"
    --learning_rate 2e-4
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --tf32 True
    --model_max_length 512
    --gradient_checkpointing True
    --dataloader_num_workers 0
    --lazy_preprocess True
    --max_steps 5
    --report_to none
)

echo "=========================================="
echo "[continual_smoke] TASK 1 — ScienceQA (5 steps, save adapter)"
echo "[continual_smoke] OUTPUT: ${TASK1_OUTPUT}"
echo "=========================================="

${COIN_DEEPSPEED} --include "${COIN_DS_INCLUDE}" --master_port 29602 \
    ETrain/Train/LLaVA/train_mem.py \
    --output_dir "${TASK1_OUTPUT}" \
    "${COMMON_ARGS[@]}"

echo ""
echo "=========================================="
echo "[continual_smoke] TASK 1 done. Checking saved files..."
echo "=========================================="
ls -lh "${TASK1_OUTPUT}/"
echo ""

# Verify adapter_model.bin was saved
if [[ ! -f "${TASK1_OUTPUT}/adapter_model.bin" ]]; then
    echo "[ERROR] adapter_model.bin not found in ${TASK1_OUTPUT}" >&2
    exit 1
fi
if [[ ! -f "${TASK1_OUTPUT}/non_lora_trainables.bin" ]]; then
    echo "[ERROR] non_lora_trainables.bin not found in ${TASK1_OUTPUT}" >&2
    exit 1
fi
echo "[continual_smoke] adapter_model.bin + non_lora_trainables.bin present. Task 1 save: OK"

echo ""
echo "=========================================="
echo "[continual_smoke] TASK 2 — ScienceQA (5 steps, load Task 1 adapter)"
echo "[continual_smoke] PREVIOUS_TASK: ${TASK1_OUTPUT}"
echo "[continual_smoke] OUTPUT: ${TASK2_OUTPUT}"
echo "=========================================="

${COIN_DEEPSPEED} --include "${COIN_DS_INCLUDE}" --master_port 29602 \
    ETrain/Train/LLaVA/train_mem.py \
    --output_dir "${TASK2_OUTPUT}" \
    --previous_task_model_path "${TASK1_OUTPUT}" \
    "${COMMON_ARGS[@]}"

echo ""
echo "=========================================="
echo "[continual_smoke] TASK 2 done. Checking saved files..."
echo "=========================================="
ls -lh "${TASK2_OUTPUT}/"

echo ""
echo "=========================================="
echo "[continual_smoke] PASS — continual learning round-trip complete."
echo "=========================================="
