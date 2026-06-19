#!/bin/bash
# Run MoELoRA T1->T8 continual training with gradient and attention logging.
# A100 single-GPU defaults are used on GPUShare; override paths/envs as needed.

set -euo pipefail

START_TASK=${1:-1}
END_TASK=${2:-8}

if [[ "$START_TASK" -lt 1 || "$END_TASK" -gt 8 || "$START_TASK" -gt "$END_TASK" ]]; then
  echo "Usage: bash scripts/LLaVA/Train_MOE/run_grad_attn_analysis.sh [start_task(1-8)] [end_task(1-8)]" >&2
  exit 1
fi

# Activate the server conda env when the script is launched from tmux/non-login shell.
COIN_CONDA_ENV=${COIN_CONDA_ENV:-coin}
if [[ x${CONDA_DEFAULT_ENV:-} != x${COIN_CONDA_ENV} && -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate ${COIN_CONDA_ENV}
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export COIN_USE_SDPA_PATCH="${COIN_USE_SDPA_PATCH:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

# Single A100 default: avoid ZeRO offload overhead. If this OOMs, rerun with
# COIN_USE_DEEPSPEED=1 COIN_DS_CONFIG=/root/CoIN/scripts/zero2.json.
COIN_USE_DEEPSPEED="${COIN_USE_DEEPSPEED:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COIN_EXPERIMENT_NAME="${COIN_EXPERIMENT_NAME:-grad_attn_expert}"
COIN_OUTPUT_ROOT="${COIN_OUTPUT_ROOT:-/hy-tmp/CoIN/checkpoints/LLaVA/${COIN_EXPERIMENT_NAME}}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/coin_paths.sh"

GRAD_OUT_ROOT="${GRAD_OUT_ROOT:-${GRAD_OUT_DIR:-${COIN_REPO_ROOT}/analysis/gradient_dominance}}"
ATTN_OUT_ROOT="${ATTN_OUT_ROOT:-${ATTN_OUT_DIR:-${COIN_REPO_ROOT}/analysis/attn_dominance}}"
LOG_ROOT="${LOG_ROOT:-${LOG_DIR:-${COIN_REPO_ROOT}/logs/LLaVA/${COIN_EXPERIMENT_NAME}}}"
mkdir -p "${GRAD_OUT_ROOT}" "${ATTN_OUT_ROOT}" "${LOG_ROOT}" "${COIN_OUTPUT_ROOT}"

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
TASK_IMAGE_GATES=(
  ""
  "${COIN_IMAGE_ROOT}/ScienceQA"
  "${COIN_IMAGE_ROOT}/TextVQA"
  "${COIN_IMAGE_ROOT}/ImageNet_withlabel"
  "${COIN_IMAGE_ROOT}/GQA"
  "${COIN_IMAGE_ROOT}/VizWiz"
  "${COIN_IMAGE_ROOT}/COCO2014"
  "${COIN_IMAGE_ROOT}/COCO2014"
  "${COIN_IMAGE_ROOT}/OCR-VQA"
)
CKPT_NAMES=(
  ""
  "ScienceQA_llava_MOE_lora_grad_attn"
  "TextVQA_llava_MOE_lora_grad_attn"
  "ImageNet_llava_MOE_lora_grad_attn"
  "GQA_llava_MOE_lora_grad_attn"
  "VizWiz_llava_MOE_lora_grad_attn"
  "Grounding_llava_MOE_lora_grad_attn"
  "VQAv2_llava_MOE_lora_grad_attn"
  "OCRVQA_llava_MOE_lora_grad_attn"
)
# Match the effective batch size of the current 8-GPU MoELoRA scripts:
# effective_bs = per_device_train_batch_size * gradient_accumulation_steps * world_size.
TASK_ORIG_TRAIN_BS=("" 8 8 4 4 4 4 4 4)
TASK_ORIG_GRAD_ACCUM=("" 8 16 16 12 8 8 8 8)
TASK_ORIG_WORLD_SIZE="${TASK_ORIG_WORLD_SIZE:-8}"
TASK_EFFECTIVE_BS=("" 512 1024 512 384 256 256 256 256)
TASK_TRAIN_BS=("" 4 4 4 4 4 4 4 4)
TASK_TOTAL_STEPS=("" 24 33 253 187 80 218 323 646)

COIN_WAIT_FOR_DATA="${COIN_WAIT_FOR_DATA:-1}"
COIN_DATA_WAIT_SECONDS="${COIN_DATA_WAIT_SECONDS:-86400}"
COIN_DATA_POLL_SECONDS="${COIN_DATA_POLL_SECONDS:-60}"

wait_for_path() {
  local path="$1"
  local desc="$2"
  local waited=0
  while [[ ! -e "$path" ]]; do
    if [[ "$COIN_WAIT_FOR_DATA" != "1" ]]; then
      echo "[run_grad_attn_analysis][PathError] Missing ${desc}: ${path}" >&2
      exit 1
    fi
    if [[ "$waited" -ge "$COIN_DATA_WAIT_SECONDS" ]]; then
      echo "[run_grad_attn_analysis][Timeout] Still missing ${desc} after ${waited}s: ${path}" >&2
      exit 1
    fi
    echo "[run_grad_attn_analysis] Waiting for ${desc}: ${path} (${waited}s elapsed)"
    sleep "$COIN_DATA_POLL_SECONDS"
    waited=$((waited + COIN_DATA_POLL_SECONDS))
  done
}

run_task() {
  local k=$1
  local task_name="${TASK_NAMES[$k]}"
  local task_id="T${k}_${task_name}"
  local data_path="${TASK_DATA[$k]}"
  local image_gate="${TASK_IMAGE_GATES[$k]}"
  local task_grad_out_dir="${GRAD_OUT_ROOT}/${task_id}"
  local task_attn_out_dir="${ATTN_OUT_ROOT}/${task_id}"
  local task_log_dir="${LOG_ROOT}/${task_id}"
  local task_output_root="${COIN_OUTPUT_ROOT}/${task_id}"
  local output_dir="${task_output_root}/${CKPT_NAMES[$k]}"
  local log_file="${task_log_dir}/task${k}_${task_name}.log"
  local target_effective_bs="${TASK_EFFECTIVE_BS[$k]}"
  local train_bs="${COIN_TRAIN_BS:-${TASK_TRAIN_BS[$k]}}"
  local grad_accum="${COIN_GRAD_ACCUM:-}"
  if [[ -z "${grad_accum}" ]]; then
    if (( target_effective_bs % train_bs != 0 )); then
      echo "[run_grad_attn_analysis][BatchError] target effective batch ${target_effective_bs} is not divisible by train_bs=${train_bs}" >&2
      exit 1
    fi
    grad_accum=$((target_effective_bs / train_bs))
  fi
  local effective_bs=$((train_bs * grad_accum))
  if (( effective_bs != target_effective_bs )) && [[ "${COIN_ALLOW_EFFECTIVE_BS_MISMATCH:-0}" != "1" ]]; then
    echo "[run_grad_attn_analysis][BatchError] effective batch ${effective_bs} != target ${target_effective_bs}; set COIN_ALLOW_EFFECTIVE_BS_MISMATCH=1 to override" >&2
    exit 1
  fi
  local log_gradient_stats="${COIN_LOG_GRADIENT_STATS:-True}"
  local log_attn_stats="${COIN_LOG_ATTN_STATS:-True}"
  local grad_total_steps="${COIN_GRAD_TOTAL_STEPS:-${TASK_TOTAL_STEPS[$k]}}"
  local grad_step_schedule="${COIN_GRAD_STEP_SCHEDULE:-staged}"
  local grad_layer_blocks="${COIN_GRAD_LAYER_BLOCKS:-0,4,8,12,16,20,24,28,31}"
  local grad_microbatch_sample="${COIN_GRAD_MICROBATCH_SAMPLE:-0.25}"
  local max_steps_arg=()
  if [[ -n "${COIN_MAX_STEPS:-}" ]]; then
    max_steps_arg=(--max_steps "${COIN_MAX_STEPS}")
  fi
  local prev_ckpt_arg=()
  mkdir -p "${task_grad_out_dir}" "${task_attn_out_dir}" "${task_log_dir}" "${task_output_root}"

  echo "============================================================"
  echo "Task ${k}/8: ${task_name}"
  echo "GPU: ${CUDA_VISIBLE_DEVICES}"
  echo "Use DeepSpeed: ${COIN_USE_DEEPSPEED}"
  echo "Original MoELoRA per-device batch size: ${TASK_ORIG_TRAIN_BS[$k]}"
  echo "Original MoELoRA gradient accumulation steps: ${TASK_ORIG_GRAD_ACCUM[$k]}"
  echo "Original MoELoRA world size: ${TASK_ORIG_WORLD_SIZE}"
  echo "Target effective batch size: ${target_effective_bs}"
  echo "Single-card per-device train batch size: ${train_bs}"
  echo "Single-card gradient accumulation steps: ${grad_accum}"
  echo "Single-card effective batch size: ${effective_bs}"
  echo "Log gradient stats: ${log_gradient_stats}"
  echo "Log attention stats: ${log_attn_stats}"
  echo "Grad total optimizer steps: ${grad_total_steps}"
  echo "Grad step schedule: ${grad_step_schedule}"
  echo "Grad layer blocks: ${grad_layer_blocks}"
  echo "Grad microbatch sample: ${grad_microbatch_sample}"
  echo "Max steps: ${COIN_MAX_STEPS:-full epoch}"
  echo "Output: ${output_dir}"
  echo "Log: ${log_file}"
  echo "Grad CSV dir: ${task_grad_out_dir}"
  echo "Attn CSV dir: ${task_attn_out_dir}"
  echo "============================================================"

  if [[ "$k" -gt 1 ]]; then
    local prev_k=$((k - 1))
    local prev_task_id="T${prev_k}_${TASK_NAMES[$prev_k]}"
    local prev_ckpt="${COIN_OUTPUT_ROOT}/${prev_task_id}/${CKPT_NAMES[$prev_k]}"
    prev_ckpt_arg=(--previous_task_model_path "${prev_ckpt}")
  fi

  if [[ "${COIN_DRY_RUN:-0}" != "1" ]]; then
    require_path "${COIN_BASE_MODEL}" "base model"
    require_path "${COIN_PRETRAIN_PROJECTOR}" "mm projector"
    require_path "${COIN_VISION_TOWER}" "vision tower"
    require_path "${COIN_INSTR_ROOT}" "instruction root"
    require_path "${COIN_IMAGE_ROOT}" "image root"
    wait_for_path "${data_path}" "task ${k} train json"
    wait_for_path "${image_gate}" "task ${k} image data"
    if [[ "$k" -gt 1 ]]; then
      wait_for_path "${prev_ckpt_arg[1]}" "previous task checkpoint T$((k - 1))"
    fi
  fi

  local train_args=(
    ETrain/Train/LLaVA/train_grad.py
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5
    --expert_num 8
    --model_name_or_path "${COIN_BASE_MODEL}"
    --pretrain_mm_mlp_adapter "${COIN_PRETRAIN_PROJECTOR}"
    --version v1
    --data_path "${data_path}"
    --image_folder "${COIN_IMAGE_ROOT}"
    --vision_tower "${COIN_VISION_TOWER}"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 True
    --output_dir "${output_dir}"
    --num_train_epochs 1
    --per_device_train_batch_size "${train_bs}"
    --per_device_eval_batch_size 2
    --gradient_accumulation_steps "${grad_accum}"
    --evaluation_strategy no
    --save_strategy epoch
    --learning_rate 2e-4
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --tf32 True
    --model_max_length 1024
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to none
    "${max_steps_arg[@]}"
    "${prev_ckpt_arg[@]}"
    --log_gradient_stats "${log_gradient_stats}"
    --log_attn_stats "${log_attn_stats}"
    --grad_total_steps "${grad_total_steps}"
    --grad_step_schedule "${grad_step_schedule}"
    --grad_layer_blocks "${grad_layer_blocks}"
    --grad_accum_steps "${grad_accum}"
    --grad_microbatch_sample "${grad_microbatch_sample}"
    --grad_task_name "${task_id}"
    --grad_output_dir "${task_grad_out_dir}"
    --attn_output_dir "${task_attn_out_dir}"
    --grad_log_interval 1
  )

  if [[ "${COIN_DRY_RUN:-0}" == "1" ]]; then
    printf '[run_grad_attn_analysis][DRY_RUN]'
    if [[ "$COIN_USE_DEEPSPEED" == "1" ]]; then
      printf ' deepspeed --include %q --master_port 29610 %q --deepspeed %q' "${COIN_DS_INCLUDE}" "${train_args[0]}" "${COIN_DS_CONFIG}"
      printf ' %q' "${train_args[@]:1}"
    else
      printf ' python -u'
      printf ' %q' "${train_args[@]}"
    fi
    printf '\n'
    return 0
  fi

  if [[ "$COIN_USE_DEEPSPEED" == "1" ]]; then
    require_path "${COIN_DS_CONFIG}" "DeepSpeed config"
    deepspeed --include "${COIN_DS_INCLUDE}" --master_port 29610 \
      "${train_args[0]}" --deepspeed "${COIN_DS_CONFIG}" "${train_args[@]:1}" \
      2>&1 | tee "${log_file}"
  else
    python -u "${train_args[@]}" 2>&1 | tee "${log_file}"
  fi

  echo "[run_grad_attn_analysis] Task ${k} done."
}

echo "Starting gradient+attention analysis: tasks ${START_TASK}->${END_TASK}"
echo "Repo: ${COIN_REPO_ROOT}"
echo "Base model: ${COIN_BASE_MODEL}"
echo "Vision tower: ${COIN_VISION_TOWER}"
echo "Projector: ${COIN_PRETRAIN_PROJECTOR}"
echo "Instruction root: ${COIN_INSTR_ROOT}"
echo "Image root: ${COIN_IMAGE_ROOT}"
echo "Output root: ${COIN_OUTPUT_ROOT}"
echo "Log root: ${LOG_ROOT}"
echo "Grad root: ${GRAD_OUT_ROOT}"
echo "Attn root: ${ATTN_OUT_ROOT}"

for k in $(seq "$START_TASK" "$END_TASK"); do
  run_task "$k"
done

echo "All requested tasks complete."
echo "Grad root: ${GRAD_OUT_ROOT}"
echo "Attn root: ${ATTN_OUT_ROOT}"
find "${GRAD_OUT_ROOT}" -maxdepth 2 -type f -name "*.csv" -printf "%p %s bytes\n" 2>/dev/null || echo "(no grad CSV yet)"
find "${ATTN_OUT_ROOT}" -maxdepth 2 -type f -name "*.csv" -printf "%p %s bytes\n" 2>/dev/null || echo "(no attn CSV yet)"
