#!/bin/bash
# run_grad_analysis.sh — run MoELoRA T1→T8 with gradient logging (GPU 6,7 only)
#
# Usage:
#   bash scripts/LLaVA/Train_MOE/run_grad_analysis.sh [start_task] [end_task]
#
# Examples:
#   bash scripts/LLaVA/Train_MOE/run_grad_analysis.sh        # T1→T8
#   bash scripts/LLaVA/Train_MOE/run_grad_analysis.sh 1 1    # T1 only (smoke test)
#   bash scripts/LLaVA/Train_MOE/run_grad_analysis.sh 2 4    # T2→T4

set -euo pipefail

START_TASK=${1:-1}
END_TASK=${2:-8}

# ── GPU constraint: only 6,7 ─────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=6,7
export COIN_DS_INCLUDE="localhost:0,1"   # DeepSpeed sees GPU 0,1 (mapped to physical 6,7)

# ── DeepSpeed CPUAdam env ─────────────────────────────────────────────────────
export CUDA_HOME=${CUDA_HOME:-/data4/home/sqx/.conda/envs/coin}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

GRAD_OUT_DIR="${COIN_REPO_ROOT}/analysis/gradient_dominance"
LOG_DIR="${COIN_REPO_ROOT}/logs/LLaVA/grad_analysis"
mkdir -p "${GRAD_OUT_DIR}" "${LOG_DIR}"

# ── Task definitions ──────────────────────────────────────────────────────────
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

run_task() {
    local k=$1
    local task_name="${TASK_NAMES[$k]}"
    local data_path="${TASK_DATA[$k]}"
    local output_dir="${COIN_OUTPUT_ROOT}/${CKPT_NAMES[$k]}"
    local log_file="${LOG_DIR}/task${k}_${task_name}.log"

    echo "════════════════════════════════════════════"
    echo "  Task ${k}/8: ${task_name}"
    echo "  Output: ${output_dir}"
    echo "  Log:    ${log_file}"
    echo "════════════════════════════════════════════"

    # Previous task checkpoint
    if [ "$k" -eq 1 ]; then
        prev_ckpt_arg=""   # T1 starts from base model
    else
        prev_k=$((k - 1))
        prev_ckpt_arg="--previous_task_model_path ${COIN_OUTPUT_ROOT}/${CKPT_NAMES[$prev_k]}"
    fi

    require_path "${data_path}" "task ${k} train json"

    deepspeed \
        --include "${COIN_DS_INCLUDE}" \
        --master_port 29610 \
        ETrain/Train/LLaVA/train_grad.py \
        --deepspeed ./scripts/zero3_offload.json \
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
        --gradient_accumulation_steps 64 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 1024 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to none \
        ${prev_ckpt_arg} \
        --log_gradient_stats True \
        --grad_task_name "T${k}_${task_name}" \
        --grad_output_dir "${GRAD_OUT_DIR}" \
        --grad_log_interval 1 \
        2>&1 | tee "${log_file}"

    echo "[run_grad_analysis] Task ${k} done."
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "Starting gradient analysis: tasks ${START_TASK}→${END_TASK}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}  |  Output: ${GRAD_OUT_DIR}"

for k in $(seq "$START_TASK" "$END_TASK"); do
    run_task "$k"
done

echo ""
echo "All tasks complete. CSV files in: ${GRAD_OUT_DIR}"
ls -lh "${GRAD_OUT_DIR}"/*.csv 2>/dev/null || echo "(no CSV yet)"
