#!/bin/bash
# Wait for continual reeval (PID 2756790) to finish, then run MoELoRA single reeval.

CONT_PID=2756790
REPO_ROOT="/data4/home/sqx/CoIN"
LOG_DIR="${REPO_ROOT}/logs/LLaVA/only_visual_reeval/MoELoRA_single"
MONITOR_LOG="${LOG_DIR}/monitor_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${MONITOR_LOG}") 2>&1

echo "[monitor] $(date) — waiting for continual reeval PID=${CONT_PID}"

while kill -0 "${CONT_PID}" 2>/dev/null; do
    sleep 60
done

echo "[monitor] $(date) — continual reeval done. Starting MoELoRA single reeval..."

conda run -n coin --no-capture-output bash \
    "${REPO_ROOT}/scripts/LLaVA/Train_MOE/reeval_visual_single.sh"

echo "[monitor] $(date) — single reeval complete."
