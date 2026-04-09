#!/bin/bash
set -euo pipefail

# Run independent single-task training for CoIN tasks 1..8.
# It reuses each Train_MOE task script but removes previous_task_model_path
# so every task starts from base model instead of continual checkpoint.
#
# Usage:
#   bash scripts/LLaVA/Train_MOE/run_coin_single.sh [start_task(1-8)] [end_task(1-8)]
# Optional env:
#   DRY_RUN=1  # only print transformed script paths

TASK_SCRIPTS=(
  "scripts/LLaVA/Train_MOE/1_Science.sh"
  "scripts/LLaVA/Train_MOE/2_TextVQA.sh"
  "scripts/LLaVA/Train_MOE/3_ImageNet.sh"
  "scripts/LLaVA/Train_MOE/4_GQA.sh"
  "scripts/LLaVA/Train_MOE/5_VizWiz.sh"
  "scripts/LLaVA/Train_MOE/6_Grounding.sh"
  "scripts/LLaVA/Train_MOE/7_vqav2.sh"
  "scripts/LLaVA/Train_MOE/8_OCRVQA.sh"
)

START_TASK=${1:-1}
END_TASK=${2:-8}
DRY_RUN=${DRY_RUN:-0}

if [[ "$START_TASK" -lt 1 || "$END_TASK" -gt 8 || "$START_TASK" -gt "$END_TASK" ]]; then
  echo "Invalid range. Usage: bash scripts/LLaVA/Train_MOE/run_coin_single.sh [start_task(1-8)] [end_task(1-8)]"
  exit 1
fi

echo "[CoIN] Run single-task training from T${START_TASK} to T${END_TASK}"

for i in $(seq "$START_TASK" "$END_TASK"); do
  src="${TASK_SCRIPTS[$((i-1))]}"
  tmp_script=$(mktemp)

  # Remove continual dependency and redirect outputs to CoIN_single.
  sed '/previous_task_model_path/d' "$src" \
    | sed 's#\./checkpoints/LLaVA/CoIN/#./checkpoints/LLaVA/CoIN_single/#g' \
    | sed 's#/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN/#/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN_single/#g' \
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
