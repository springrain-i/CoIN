#!/bin/bash
# T8 MoEMoKA eval runner — launched by wait_and_eval_T8.sh inside the coin tmux session.
# Usage: bash run_evals_T8.sh <CKPT_DIR>

set -euo pipefail

CKPT_DIR="${1:-/hy-tmp/checkpoints/LLaVA/CoIN/OCRVQA_llava_MoEMoKA_lora}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}/eval"
LOG_DIR="${LOG_DIR:-/data4/home/sqx/CoIN/logs/LLaVA/MoEMoKA}"

echo "[$(date)] T8 eval runner started inside tmux coin session."
echo "[$(date)] CKPT_DIR=${CKPT_DIR}"

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[$(date)] ERROR: ${CKPT_DIR} not found. Aborting." >&2
    exit 1
fi

echo "[$(date)] Checkpoint dir: $(ls "${CKPT_DIR}" | tr '\n' ' ')"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EVAL_SCRIPTS=(
    "${EVAL_DIR}/1_eval_sqa.sh"
    "${EVAL_DIR}/2_eval_textqa.sh"
    "${EVAL_DIR}/3_eval_imagenet.sh"
    "${EVAL_DIR}/4_eval_gqa.sh"
    "${EVAL_DIR}/5_eval_vizwiz.sh"
    "${EVAL_DIR}/6_eval_grounding.sh"
    "${EVAL_DIR}/7_eval_vqav2.sh"
    "${EVAL_DIR}/8_eval_ocrvqa.sh"
)
TASK_NAMES=(SciQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCRVQA)

for MODE in all text visual; do
    for i in "${!EVAL_SCRIPTS[@]}"; do
        TASK_NUM=$((i+1))
        STAGE="T8_eval_T${TASK_NUM}_${MODE}"
        SCRIPT="${EVAL_SCRIPTS[$i]}"
        TASK_LOG="${LOG_DIR}/${STAGE}.log"
        echo "[$(date)] >> Mode=${MODE}  Task=${TASK_NAMES[$i]}(${TASK_NUM})  Stage=${STAGE}" | tee -a "${TASK_LOG}"
        bash "${SCRIPT}" "${STAGE}" "${CKPT_DIR}" "${MODE}" 2>&1 | tee -a "${TASK_LOG}"
        echo "[$(date)] << Done: ${STAGE}" | tee -a "${TASK_LOG}"
    done
done

echo "[$(date)] All T8 MoEMoKA evals complete."
