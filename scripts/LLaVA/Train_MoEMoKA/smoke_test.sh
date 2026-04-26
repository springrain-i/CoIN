#!/bin/bash
# Smoke test: 10 steps with batch_size=1, gradient_accumulation=1.
# Purpose: verify MoE-MoKA forward/backward pass runs without error on real data.
set -euo pipefail

PROMPT_VERSION=v1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

DATA_PATH="${COIN_INSTR_ROOT}/ScienceQA/train.json"
OUTPUT_DIR="${COIN_OUTPUT_ROOT}/smoke_test_MoEMoKA"

require_path "${COIN_BASE_MODEL}" "base model"
require_path "${COIN_PRETRAIN_PROJECTOR}" "mm projector"
require_path "${COIN_VISION_TOWER}" "vision tower"
require_path "${DATA_PATH}" "ScienceQA train json"

echo "[smoke_test] BASE_MODEL: ${COIN_BASE_MODEL}"
echo "[smoke_test] OUTPUT:     ${OUTPUT_DIR}"

${COIN_DEEPSPEED} --include "${COIN_DS_INCLUDE}" --master_port 29601 ETrain/Train/LLaVA/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --moe_moka_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --expert_num 4 \
    --model_name_or_path "${COIN_BASE_MODEL}" \
    --pretrain_mm_mlp_adapter "${COIN_PRETRAIN_PROJECTOR}" \
    --version $PROMPT_VERSION \
    --data_path "${DATA_PATH}" \
    --image_folder "${COIN_IMAGE_ROOT}" \
    --vision_tower "${COIN_VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --max_steps 10 \
    --report_to none
