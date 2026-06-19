#!/bin/bash
set -euo pipefail

# Re-evaluate MoELoRA continual checkpoints in vision-only mode.
# Evaluation protocol: after training Ti, evaluate on all tasks T1..Ti (online matrix).
# This gives a full 8x8 lower-triangular matrix (36 evals total).
# Results go to new stage subdirs; existing results are NOT overwritten.
#
# Usage:
#   bash scripts/LLaVA/Train_MOE/reeval_visual_cont_T8.sh [start_train(1-8)] [end_train(1-8)]
# Optional env:
#   CUDA_VISIBLE_DEVICES, COIN_CONT_ROOT (override default continual checkpoint root)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

START_TRAIN=${1:-1}
END_TRAIN=${2:-8}

COIN_CONT_ROOT="${COIN_CONT_ROOT:-${REPO_ROOT}/checkpoints/LLaVA/CoIN}"
PARSE_ACC_PY="${REPO_ROOT}/scripts/LLaVA/Eval/parse_accuracy.py"

LOG_ROOT="${REPO_ROOT}/logs/LLaVA/only_visual_reeval/MoELoRA_cont"
OUT_CSV="${REPO_ROOT}/results/CoIN/LLaVA/metrics/MoELoRA_cont_visual_reeval.csv"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
mkdir -p "${LOG_ROOT}"

TASK_NAMES=(ScienceQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCRVQA)
CKPT_NAMES=(
  "ScienceQA_llava_MOE_lora"
  "TextVQA_llava_MOE_lora"
  "ImageNet_llava_MOE_lora"
  "GQA_llava_MOE_lora"
  "VizWiz_llava_MOE_lora"
  "Grounding_llava_MOE_lora"
  "VQAv2_llava_MOE_lora"
  "OCRVQA_llava_MOE_lora"
)
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
# Must match RESULT_DIR hardcoded in each eval script above
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

# Online eval matrix: Tk checkpoint evaluates T1..Tk
for train_i in $(seq "${START_TRAIN}" "${END_TRAIN}"); do
  ckpt="${COIN_CONT_ROOT}/${CKPT_NAMES[$((train_i-1))]}"
  require_path "${ckpt}" "MoELoRA continual T${train_i} checkpoint"

  for eval_j in $(seq 1 "${train_i}"); do
    task_name="${TASK_NAMES[$((eval_j-1))]}"
    eval_script="${EVAL_SCRIPTS[$((eval_j-1))]}"
    result_dir="${RESULT_DIRS[$((eval_j-1))]}"
    stage="reeval_visual_cont_train${train_i}_eval${eval_j}"
    stage_dir="${result_dir}/${stage}"
    task_log="${LOG_ROOT}/train${train_i}_${task_name}_vision_${TIMESTAMP}.log"

    echo "[reeval][MoELoRA cont] train=T${train_i} eval=T${eval_j} (${task_name}) mode=vision"
    echo "[reeval] ckpt:      ${ckpt}"
    echo "[reeval] stage_dir: ${stage_dir}"
    echo "[reeval] log:       ${task_log}"

    # Skip if result already exists and not yet in CSV.
    if python "${PARSE_ACC_PY}" --stage-dir "${stage_dir}" > /dev/null 2>&1; then
      acc=$(python "${PARSE_ACC_PY}" --stage-dir "${stage_dir}")
      if ! grep -q "^continual,vision,${train_i},${eval_j}," "${OUT_CSV}" 2>/dev/null; then
        echo "continual,vision,${train_i},${eval_j},${acc},${stage_dir}" >> "${OUT_CSV}"
      fi
      echo "[reeval] SKIP (done) train=T${train_i} eval=T${eval_j} (${task_name}) acc: ${acc}"
      continue
    fi

    bash "${eval_script}" "${stage}" "${ckpt}" "vision" 2>&1 | tee "${task_log}"

    acc=$(python "${PARSE_ACC_PY}" --stage-dir "${stage_dir}")
    echo "continual,vision,${train_i},${eval_j},${acc},${stage_dir}" >> "${OUT_CSV}"
    echo "[reeval] train=T${train_i} eval=T${eval_j} (${task_name}) vision acc: ${acc}"
  done
done

echo "[reeval] MoELoRA continual vision-only re-eval complete."
echo "[reeval] CSV: ${OUT_CSV}"
echo "[reeval] Logs: ${LOG_ROOT}/"
