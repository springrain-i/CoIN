#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Run independent single-task training for CoIN tasks 1..8.
# It reuses each Train_MOE task script but removes previous_task_model_path
# so every task starts from base model instead of continual checkpoint.
#
# Usage:
#   bash scripts/LLaVA/Train_MOE/run_coin_single.sh [start_task(1-8)] [end_task(1-8)]
# Optional env:
#   DRY_RUN=1  # only print transformed script paths

TASK_SCRIPTS=(
  "${SCRIPT_DIR}/1_Science.sh"
  "${SCRIPT_DIR}/2_TextVQA.sh"
  "${SCRIPT_DIR}/3_ImageNet.sh"
  "${SCRIPT_DIR}/4_GQA.sh"
  "${SCRIPT_DIR}/5_VizWiz.sh"
  "${SCRIPT_DIR}/6_Grounding.sh"
  "${SCRIPT_DIR}/7_vqav2.sh"
  "${SCRIPT_DIR}/8_OCRVQA.sh"
)

START_TASK=${1:-1}
END_TASK=${2:-8}
DRY_RUN=${DRY_RUN:-0}

if [[ "$START_TASK" -lt 1 || "$END_TASK" -gt 8 || "$START_TASK" -gt "$END_TASK" ]]; then
  echo "Invalid range. Usage: bash scripts/LLaVA/Train_MOE/run_coin_single.sh [start_task(1-8)] [end_task(1-8)]"
  exit 1
fi

echo "[CoIN] Run single-task training from T${START_TASK} to T${END_TASK}"
cd "${REPO_ROOT}"

# Isolated single-task outputs should not overlap continual outputs.
export COIN_OUTPUT_ROOT="${COIN_OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/LLaVA/CoIN_single}"

for i in $(seq "$START_TASK" "$END_TASK"); do
  src="${TASK_SCRIPTS[$((i-1))]}"
  tmp_script=$(mktemp)

  # Remove continual dependency so every task starts from base model.
  sed '/previous_task_model_path/d' "$src" \
    > "$tmp_script"

  chmod +x "$tmp_script"
  echo "[CoIN] >>> Single-task T${i} via ${src} (tmp=${tmp_script})"

  if [[ "$DRY_RUN" != "1" ]]; then
    bash "$tmp_script"
    echo "[CoIN] <<< Finished single-task T${i}"
  fi

  rm -f "$tmp_script"
done

echo "[CoIN] Single-task training sequence complete."
