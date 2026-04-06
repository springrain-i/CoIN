#!/bin/bash
set -euo pipefail

# Fixed CoIN continual order: 1->8
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
  echo "Invalid range. Usage: bash scripts/LLaVA/Train_MOE/run_coin_sequence.sh [start_task(1-8)] [end_task(1-8)]"
  exit 1
fi

echo "[CoIN] Run continual training sequence from T${START_TASK} to T${END_TASK}"

for i in $(seq "$START_TASK" "$END_TASK"); do
  script="${TASK_SCRIPTS[$((i-1))]}"
  echo "[CoIN] >>> Training task T${i} via ${script}"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  bash "$script"
  echo "[CoIN] <<< Finished T${i}"
done

echo "[CoIN] Continual training sequence complete."
