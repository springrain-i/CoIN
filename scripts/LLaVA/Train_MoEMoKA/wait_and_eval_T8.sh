#!/bin/bash
# Monitor T8 MoEMoKA training; when done, launch all evals in the coin tmux session.
# Usage: nohup bash wait_and_eval_T8.sh &

set -euo pipefail

LOG_FILE="/data4/home/sqx/CoIN/logs/LLaVA/MoEMoKA/train_T8_20260528_010156.log"
CKPT_DIR="/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN_MoEMoKA/OCRVQA_llava_MoEMoKA_lora"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EVAL="${SCRIPT_DIR}/run_evals_T8.sh"
SELF_LOG="/data4/home/sqx/CoIN/logs/LLaVA/MoEMoKA/monitor_T8_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${SELF_LOG}") 2>&1
echo "[$(date)] Monitor started. Watching: ${LOG_FILE}"
echo "[$(date)] Checkpoint expected at: ${CKPT_DIR}"
echo "[$(date)] Monitor log: ${SELF_LOG}"

# ── Wait for training to finish ──────────────────────────────────────
while true; do
    if grep -q '"train_runtime"' "${LOG_FILE}" 2>/dev/null; then
        echo "[$(date)] Detected train_runtime in log — training complete."
        break
    fi
    if [[ -f "${CKPT_DIR}/adapter_model.safetensors" ]] || \
       [[ -f "${CKPT_DIR}/adapter_model.bin" ]]; then
        echo "[$(date)] Checkpoint adapter found in ${CKPT_DIR}."
        break
    fi
    echo "[$(date)] Still training... (sleeping 60s)"
    sleep 60
done

# Extra buffer for checkpoint flush to disk
echo "[$(date)] Waiting 90s for checkpoint flush..."
sleep 90

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[$(date)] ERROR: ${CKPT_DIR} not found. Aborting." >&2
    exit 1
fi

echo "[$(date)] Checkpoint confirmed. Launching eval in coin tmux session..."

# ── Send eval command to coin session window 0 ───────────────────────
tmux send-keys -t coin:0 \
    "bash '${RUN_EVAL}' '${CKPT_DIR}'" Enter

echo "[$(date)] Eval command sent to coin:0. Monitor exiting."
