#!/bin/bash
set -euo pipefail

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################


################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

DATA_PATH="${COIN_INSTR_ROOT}/ImageNet/train.json"
PREVIOUS_TASK_MODEL_PATH="${COIN_OUTPUT_ROOT}/TextVQA_llava_MoEMoKA_lora"
OUTPUT_DIR="${COIN_OUTPUT_ROOT}/ImageNet_llava_MoEMoKA_lora"

require_path "${COIN_BASE_MODEL}" "base model"
require_path "${COIN_VISION_TOWER}" "vision tower"
require_path "${DATA_PATH}" "ImageNet train json"
require_path "${PREVIOUS_TASK_MODEL_PATH}" "previous task checkpoint"

${COIN_DEEPSPEED} --include "${COIN_DS_INCLUDE}" --master_port 29600 ETrain/Train/LLaVA/train_mem.py \
    --deepspeed "${COIN_DS_CONFIG:-./scripts/zero3.json}" \
    --moe_moka_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --expert_num 8 \
    --model_name_or_path "${COIN_BASE_MODEL}" \
    --previous_task_model_path "${PREVIOUS_TASK_MODEL_PATH}" \
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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none