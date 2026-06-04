#!/bin/bash
# run_grad_t1_2gpu.sh — gradient analysis on 2 GPUs (6,7) with DeepSpeed ZeRO-3 + CPU offload
#
# Usage:
#   bash scripts/LLaVA/Train_MOE/run_grad_t1_2gpu.sh [task_num]
#   Default: task 1 (ScienceQA)
#
# Multi-GPU notes:
#   - ZeRO-3 + offload_optimizer: optimizer states on CPU, params/grads sharded across GPUs.
#   - Gradient hooks (hook_A / hook_B) capture activation gradients which are correct under
#     ZeRO-3 (activation tensors are not sharded; only parameter gradients are).
#   - Only rank 0 writes gradient CSV (gradient_logger._flush has rank guard).
#   - Expected peak GPU memory per card: ~18-20 GB (7B param shard + grads + activations).
#
# Exact metric vs single-GPU run:
#   G^t_A = ||einsum(g_out_A_text, x_text)||_F  (not proxy, full outer-product sum)
#   Peak overhead from x_ref: ~7 layers × 16 MB per checkpoint segment ≈ 112 MB.

set -euo pipefail

TASK=${1:-1}

export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# DeepSpeed CPUAdam build env — CUDA_HOME must point to a real nvcc.
# System CUDA 11.8 is compatible with PyTorch 11.7 (APIs match, minor version difference is OK).
export CUDA_HOME=/usr/local/cuda-11.8
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

DS_CONFIG="${COIN_REPO_ROOT}/scripts/zero3_offload.json"
GRAD_OUT_DIR="${COIN_REPO_ROOT}/analysis/gradient_dominance"
LOG_DIR="${COIN_REPO_ROOT}/logs/LLaVA/grad_analysis"
mkdir -p "${GRAD_OUT_DIR}" "${LOG_DIR}"

TASK_NAMES=("" "ScienceQA" "TextVQA" "ImageNet" "GQA" "VizWiz" "Grounding" "VQAv2" "OCRVQA")
TASK_DATA=(
    ""
    "${COIN_INSTR_ROOT}/ScienceQA/train.json"
    "${COIN_INSTR_ROOT}/TextVQA/train.json"
    "${COIN_INSTR_ROOT}/ImageNet/train.json"
    "${COIN_INSTR_ROOT}/GQA/train.json"
    "${COIN_INSTR_ROOT}/VizWiz/train.json"
    "${COIN_INSTR_ROOT}/Grounding/train.json"
    "${COIN_INSTR_ROOT}/VQAv2/train.json"
    "${COIN_INSTR_ROOT}/OCRVQA/train.json"
)
CKPT_NAMES=(
    ""
    "ScienceQA_llava_MOE_lora_grad"
    "TextVQA_llava_MOE_lora_grad"
    "ImageNet_llava_MOE_lora_grad"
    "GQA_llava_MOE_lora_grad"
    "VizWiz_llava_MOE_lora_grad"
    "Grounding_llava_MOE_lora_grad"
    "VQAv2_llava_MOE_lora_grad"
    "OCRVQA_llava_MOE_lora_grad"
)

task_name="${TASK_NAMES[$TASK]}"
data_path="${TASK_DATA[$TASK]}"
output_dir="${COIN_OUTPUT_ROOT}/${CKPT_NAMES[$TASK]}_2gpu"
log_file="${LOG_DIR}/task${TASK}_${task_name}_2gpu.log"

if [ "$TASK" -eq 1 ]; then
    prev_ckpt_arg=""
else
    prev_k=$((TASK - 1))
    prev_ckpt_arg="--previous_task_model_path ${COIN_OUTPUT_ROOT}/${CKPT_NAMES[$prev_k]}_2gpu"
fi

echo "Task ${TASK}/8: ${task_name}  |  GPUs: ${CUDA_VISIBLE_DEVICES}  |  ZeRO-3 offload"
echo "Output: ${output_dir}"
echo "Log:    ${log_file}"

deepspeed --num_gpus 2 \
    ETrain/Train/LLaVA/train_grad.py \
    --deepspeed "${DS_CONFIG}" \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --expert_num 8 \
    --model_name_or_path "${COIN_BASE_MODEL}" \
    --pretrain_mm_mlp_adapter "${COIN_PRETRAIN_PROJECTOR}" \
    --version v1 \
    --data_path "${data_path}" \
    --image_folder "${COIN_IMAGE_ROOT}" \
    --vision_tower "${COIN_VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${output_dir}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
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
    --report_to none \
    ${prev_ckpt_arg} \
    --log_gradient_stats True \
    --grad_task_name "T${TASK}_${task_name}" \
    --grad_output_dir "${GRAD_OUT_DIR}" \
    --grad_log_every 5 \
    2>&1 | tee "${log_file}"

echo "Task ${TASK} done. CSV: ${GRAD_OUT_DIR}/T${TASK}_${task_name}_grad_stats.csv"
