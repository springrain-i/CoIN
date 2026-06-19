#!/bin/bash
set -euo pipefail

# Re-evaluate MoELoRA single-task checkpoints (each task on itself) in vision-only mode.
# Fixes the generation-token mask bug (mask=3) introduced in fix/vision-gen-token.
# Results go to new stage subdirs; existing results are NOT overwritten.
#
# Usage:
#   bash scripts/LLaVA/Train_MOE/reeval_visual_single.sh [start_task(1-8)] [end_task(1-8)]
# Optional env:
#   CUDA_VISIBLE_DEVICES, COIN_SINGLE_ROOT (override default single checkpoint root)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

START_TASK=${1:-1}
END_TASK=${2:-8}

COIN_SINGLE_ROOT="${COIN_SINGLE_ROOT:-${REPO_ROOT}/checkpoints/LLaVA/CoIN_single}"
PARSE_ACC_PY="${REPO_ROOT}/scripts/LLaVA/Eval/parse_accuracy.py"

LOG_ROOT="${REPO_ROOT}/logs/LLaVA/only_visual_reeval/MoELoRA_single"
OUT_CSV="${REPO_ROOT}/results/CoIN/LLaVA/metrics/MoELoRA_single_visual_reeval.csv"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
mkdir -p "${LOG_ROOT}"

EVAL_SCRIPTS=(
  "${REPO_ROOT}/scripts/LLaVA/Eval/1_eval_sqa.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/2_eval_textqa.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/3_eval_ImageNet.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/4_eval_gqa.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/5_eval_vizwiz.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/6_eval_grounding.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/7_eval_vqav2.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/8_eval_ocrvqa.sh"
)
TASK_NAMES=(ScienceQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCRVQA)
RESULT_DIRS=(
  "${REPO_ROOT}/results/CoIN/LLaVA/ScienceQA_NoMerge_Visual"
  "${REPO_ROOT}/results/CoIN/LLaVA/Final_MOE_only_vision"
  "${REPO_ROOT}/results/CoIN/LLaVA/Final_ON_ImageNet_MOE_only_vision"
  "${REPO_ROOT}/results/CoIN/LLaVA/GQA_MOE_only_vision"
  "${REPO_ROOT}/results/CoIN/LLaVA/VizWiz"
  "${REPO_ROOT}/results/CoIN/LLaVA/Grounding"
  "${REPO_ROOT}/results/CoIN/LLaVA/VQAv2"
  "${REPO_ROOT}/results/CoIN/LLaVA/OCRVQA"
)

mkdir -p "$(dirname "${OUT_CSV}")"
if [[ ! -f "${OUT_CSV}" ]]; then
  echo "regime,mode,train_task,eval_task,accuracy,result_stage_dir" > "${OUT_CSV}"
fi

export COIN_USE_SDPA_PATCH=1
export COIN_EVAL_BATCH_SIZE=4

cd "${REPO_ROOT}"

for i in $(seq "${START_TASK}" "${END_TASK}"); do
  task_name="${TASK_NAMES[$((i-1))]}"
  eval_script="${EVAL_SCRIPTS[$((i-1))]}"
  result_dir="${RESULT_DIRS[$((i-1))]}"
  ckpt="${COIN_SINGLE_ROOT}/${task_name}_llava_MOE_lora"
  stage="reeval_visual_single_T${i}"
  stage_dir="${result_dir}/${stage}"
  task_log="${LOG_ROOT}/${task_name}_vision_${TIMESTAMP}.log"

  require_path "${ckpt}" "MoELoRA single checkpoint for T${i} (${task_name})"

  echo "[reeval][MoELoRA single] task=T${i} (${task_name}) mode=vision"
  echo "[reeval] ckpt: ${ckpt}"
  echo "[reeval] stage_dir: ${stage_dir}"
  echo "[reeval] log: ${task_log}"

  # Skip if result already exists.
  if python "${PARSE_ACC_PY}" --stage-dir "${stage_dir}" > /dev/null 2>&1; then
    acc=$(python "${PARSE_ACC_PY}" --stage-dir "${stage_dir}")
    if ! grep -q "^single,vision,${i},${i}," "${OUT_CSV}" 2>/dev/null; then
      echo "single,vision,${i},${i},${acc},${stage_dir}" >> "${OUT_CSV}"
    fi
    echo "[reeval] SKIP (done) T${i} (${task_name}) acc: ${acc}"
    continue
  fi

  bash "${eval_script}" "${stage}" "${ckpt}" "vision" 2>&1 | tee "${task_log}"

  acc=$(python "${PARSE_ACC_PY}" --stage-dir "${stage_dir}")
  if ! grep -q "^single,vision,${i},${i}," "${OUT_CSV}" 2>/dev/null; then
    echo "single,vision,${i},${i},${acc},${stage_dir}" >> "${OUT_CSV}"
  fi
  echo "[reeval] T${i} (${task_name}) vision accuracy: ${acc}"
done

echo "[reeval] MoELoRA single vision-only re-eval complete."
echo "[reeval] CSV: ${OUT_CSV}"
echo "[reeval] Logs: ${LOG_ROOT}/"
