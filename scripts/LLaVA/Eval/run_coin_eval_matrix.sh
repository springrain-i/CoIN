#!/bin/bash
set -euo pipefail

# Usage:
# 1) Continual (evaluate seen tasks after each training stage)
#    bash scripts/LLaVA/Eval/run_coin_eval_matrix.sh continual
# 2) Single-task (evaluate each task checkpoint on itself)
#    bash scripts/LLaVA/Eval/run_coin_eval_matrix.sh single
#
# Optional envs:
#   MODES="all text vision"
#   STAGE_PREFIX="CoIN"
#   OUT_CSV="results/CoIN/LLaVA/metrics/continual_eval_matrix.csv"

REGIME=${1:-continual}
MODES=${MODES:-"all text vision"}
STAGE_PREFIX=${STAGE_PREFIX:-"CoIN"}

if [[ "$REGIME" != "continual" && "$REGIME" != "single" ]]; then
  echo "REGIME must be continual or single"
  exit 1
fi

mkdir -p results/CoIN/LLaVA/metrics
OUT_CSV=${OUT_CSV:-"results/CoIN/LLaVA/metrics/${REGIME}_eval_matrix.csv"}

# Fixed task order and script mapping
TASK_IDS=(1 2 3 4 5 6 7 8)
EVAL_SCRIPTS=(
  "scripts/LLaVA/Eval/1_eval_sqa.sh"
  "scripts/LLaVA/Eval/2_eval_textqa.sh"
  "scripts/LLaVA/Eval/3_eval_ImageNet.sh"
  "scripts/LLaVA/Eval/4_eval_gqa.sh"
  "scripts/LLaVA/Eval/5_eval_vizwiz.sh"
  "scripts/LLaVA/Eval/6_eval_grounding.sh"
  "scripts/LLaVA/Eval/7_eval_vqav2.sh"
  "scripts/LLaVA/Eval/8_eval_ocrvqa.sh"
)
RESULT_DIRS=(
  "results/CoIN/LLaVA/ScienceQA_NoMerge_Visual"
  "results/CoIN/LLaVA/Final_MOE_only_vision"
  "results/CoIN/LLaVA/Final_ON_ImageNet_MOE_only_vision"
  "results/CoIN/LLaVA/GQA_MOE_only_vision"
  "results/CoIN/LLaVA/VizWiz"
  "results/CoIN/LLaVA/Grounding"
  "results/CoIN/LLaVA/VQAv2"
  "results/CoIN/LLaVA/OCRVQA"
)
CKPT_PATHS=(
  "./checkpoints/LLaVA/CoIN/ScienceQA_llava_MOE_lora"
  "./checkpoints/LLaVA/CoIN/TextVQA_llava_MOE_lora"
  "./checkpoints/LLaVA/CoIN/ImageNet_llava_MOE_lora"
  "./checkpoints/LLaVA/CoIN/GQA_llava_MOE_lora"
  "./checkpoints/LLaVA/CoIN/VizWiz_llava_MOE_lora"
  "./checkpoints/LLaVA/CoIN/Grounding_llava_MOE_lora"
  "./checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora"
  "./checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora"
)

if [[ ! -f "$OUT_CSV" ]]; then
  echo "regime,mode,train_task,eval_task,accuracy,result_stage_dir" > "$OUT_CSV"
fi

run_eval_once() {
  local train_task=$1
  local eval_task=$2
  local mode=$3

  local eval_script=${EVAL_SCRIPTS[$((eval_task-1))]}
  local result_dir=${RESULT_DIRS[$((eval_task-1))]}
  local model_path=${CKPT_PATHS[$((train_task-1))]}
  local stage="${STAGE_PREFIX}_${REGIME}_m${mode}_train${train_task}_eval${eval_task}"

  echo "[Eval] mode=${mode}, train=T${train_task}, eval=T${eval_task}"
  bash "$eval_script" "$stage" "$model_path" "$mode"

  local stage_dir="${result_dir}/${stage}"
  local acc
  acc=$(python scripts/LLaVA/Eval/parse_accuracy.py --stage-dir "$stage_dir")
  echo "${REGIME},${mode},${train_task},${eval_task},${acc},${stage_dir}" >> "$OUT_CSV"
}

for mode in $MODES; do
  for train_task in "${TASK_IDS[@]}"; do
    if [[ "$REGIME" == "single" ]]; then
      run_eval_once "$train_task" "$train_task" "$mode"
    else
      for eval_task in $(seq 1 "$train_task"); do
        run_eval_once "$train_task" "$eval_task" "$mode"
      done
    fi
  done
done

echo "[Eval] Matrix evaluation complete -> ${OUT_CSV}"
