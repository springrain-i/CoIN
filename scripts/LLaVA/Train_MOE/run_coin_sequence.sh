#!/bin/bash
set -euo pipefail

# 作用：按 CoIN 8 任务顺序执行持续学习训练，
# 每个任务训练后立即进行 all/text/visual 三模式在线评测，
# 并在全部训练结束后再做一轮全量三模式评测与结果汇总。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

EVAL_SCRIPT_DIR="${REPO_ROOT}/scripts/LLaVA/Eval"
PARSE_ACC_PY="${EVAL_SCRIPT_DIR}/parse_accuracy.py"

# Fixed CoIN continual order: 1->8
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

TASK_NAMES=(
  "ScienceQA"
  "TextVQA"
  "ImageNet"
  "GQA"
  "VizWiz"
  "Grounding"
  "VQAv2"
  "OCRVQA"
)

EVAL_SCRIPTS=(
  "${EVAL_SCRIPT_DIR}/1_eval_sqa.sh"
  "${EVAL_SCRIPT_DIR}/2_eval_textqa.sh"
  "${EVAL_SCRIPT_DIR}/3_eval_ImageNet.sh"
  "${EVAL_SCRIPT_DIR}/4_eval_gqa.sh"
  "${EVAL_SCRIPT_DIR}/5_eval_vizwiz.sh"
  "${EVAL_SCRIPT_DIR}/6_eval_grounding.sh"
  "${EVAL_SCRIPT_DIR}/7_eval_vqav2.sh"
  "${EVAL_SCRIPT_DIR}/8_eval_ocrvqa.sh"
)

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

MODES=${MODES:-"all text visual"}
ENABLE_EVAL=${ENABLE_EVAL:-1}
STAGE_PREFIX=${STAGE_PREFIX:-"CoIN"}
VALID_MODES=(all text visual vision)

START_TASK=${1:-1}
END_TASK=${2:-8}
DRY_RUN=${DRY_RUN:-0}

METRIC_DIR="${REPO_ROOT}/results/CoIN/LLaVA/metrics"
ONLINE_OUT_CSV="${METRIC_DIR}/continual_online_eval.csv"
FINAL_OUT_CSV="${METRIC_DIR}/continual_final_eval.csv"

if [[ "$START_TASK" -lt 1 || "$END_TASK" -gt 8 || "$START_TASK" -gt "$END_TASK" ]]; then
  echo "Invalid range. Usage: bash scripts/LLaVA/Train_MOE/run_coin_sequence.sh [start_task(1-8)] [end_task(1-8)]"
  exit 1
fi

echo "[CoIN] Run continual training sequence from T${START_TASK} to T${END_TASK}"
cd "${REPO_ROOT}"

# Log directory
LOG_ROOT="${REPO_ROOT}/logs/LLaVA/CoIN"
mkdir -p "${LOG_ROOT}"

# Timestamp for this run
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_ROOT}/continual_${TIMESTAMP}.log"

echo "[CoIN] Main log: ${MAIN_LOG}"

if [[ "$ENABLE_EVAL" == "1" ]]; then
  mkdir -p "${METRIC_DIR}"
  if [[ ! -f "$ONLINE_OUT_CSV" ]]; then
    echo "phase,mode,train_task,eval_task,accuracy,result_stage_dir" > "$ONLINE_OUT_CSV"
  fi
  if [[ ! -f "$FINAL_OUT_CSV" ]]; then
    echo "phase,mode,train_task,eval_task,accuracy,result_stage_dir" > "$FINAL_OUT_CSV"
  fi
fi

run_eval_once() {
  local phase="$1"
  local train_task="$2"
  local eval_task="$3"
  local mode="$4"

  local eval_script="${EVAL_SCRIPTS[$((eval_task-1))]}"
  local result_dir="${RESULT_DIRS[$((eval_task-1))]}"
  local model_path="${COIN_OUTPUT_ROOT}/${TASK_NAMES[$((train_task-1))]}_llava_MOE_lora"
  local stage="${STAGE_PREFIX}_${phase}_m${mode}_train${train_task}_eval${eval_task}"
  local stage_dir="${result_dir}/${stage}"
  local out_csv

  if [[ "$phase" == "online" ]]; then
    out_csv="$ONLINE_OUT_CSV"
  else
    out_csv="$FINAL_OUT_CSV"
  fi

  echo "[Eval][$phase] mode=${mode}, train=T${train_task}, eval=T${eval_task}"

  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  require_path "$model_path" "trained checkpoint for T${train_task}"

  bash "$eval_script" "$stage" "$model_path" "$mode"
  local acc
  acc=$(python "$PARSE_ACC_PY" --stage-dir "$stage_dir")
  echo "${phase},${mode},${train_task},${eval_task},${acc},${stage_dir}" >> "$out_csv"
}

validate_modes() {
  local raw_mode
  for raw_mode in $MODES; do
    local ok=0
    local allowed
    for allowed in "${VALID_MODES[@]}"; do
      if [[ "$raw_mode" == "$allowed" ]]; then
        ok=1
        break
      fi
    done
    if [[ "$ok" -ne 1 ]]; then
      echo "Invalid mode in MODES: ${raw_mode}. Allowed: all text visual vision" >&2
      exit 1
    fi
  done
}

validate_modes

for i in $(seq "$START_TASK" "$END_TASK"); do
  script="${TASK_SCRIPTS[$((i-1))]}"
  task_name="${TASK_NAMES[$((i-1))]}"
  task_log="${LOG_ROOT}/${task_name}_${TIMESTAMP}.log"
  echo "[CoIN] >>> Training task T${i} (${task_name}) via ${script}"
  echo "[CoIN] Task log: ${task_log}"

  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$ENABLE_EVAL" == "1" ]]; then
      for mode in $MODES; do
        for eval_task in $(seq 1 "$i"); do
          run_eval_once "online" "$i" "$eval_task" "$mode"
        done
      done
    fi
    continue
  fi

  bash "$script" 2>&1 | tee "$task_log"
  echo "[CoIN] <<< Finished T${i} (${task_name})"
  echo "[CoIN] Log saved to: ${task_log}"

  if [[ "$ENABLE_EVAL" == "1" ]]; then
    for mode in $MODES; do
      for eval_task in $(seq 1 "$i"); do
        run_eval_once "online" "$i" "$eval_task" "$mode"
      done
    done
  fi
done

if [[ "$ENABLE_EVAL" == "1" ]]; then
  echo "[CoIN] >>> Final full eval after continual sequence"
  final_train_task="$END_TASK"
  for mode in $MODES; do
    for eval_task in $(seq 1 "$END_TASK"); do
      run_eval_once "final" "$final_train_task" "$eval_task" "$mode"
    done
  done
  echo "[CoIN] <<< Final full eval complete"
  echo "[CoIN] Online eval metrics -> ${ONLINE_OUT_CSV}"
  echo "[CoIN] Final eval metrics  -> ${FINAL_OUT_CSV}"
fi

echo "[CoIN] Continual training sequence complete."
echo "[CoIN] All task logs saved to: ${LOG_ROOT}/"
