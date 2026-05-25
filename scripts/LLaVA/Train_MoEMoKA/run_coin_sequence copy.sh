#!/bin/bash
set -euo pipefail

# 作用：按 CoIN 8 任务顺序执行持续学习训练，
# 每个任务训练后立即进行 all/text/visual 三模式在线评测，
# 并在全部训练结束后再做一轮全量三模式评测与结果汇总。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

EVAL_SCRIPT_DIR="${SCRIPT_DIR}/eval"
PARSE_ACC_PY="${REPO_ROOT}/scripts/LLaVA/Eval/parse_accuracy.py"

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
  "${EVAL_SCRIPT_DIR}/3_eval_imagenet.sh"
  "${EVAL_SCRIPT_DIR}/4_eval_gqa.sh"
  "${EVAL_SCRIPT_DIR}/5_eval_vizwiz.sh"
  "${EVAL_SCRIPT_DIR}/6_eval_grounding.sh"
  "${EVAL_SCRIPT_DIR}/7_eval_vqav2.sh"
  "${EVAL_SCRIPT_DIR}/8_eval_ocrvqa.sh"
)

# MoEMoKA results live under a single root (set in eval_common.sh via coin_paths.sh)
MOEMOKA_RESULT_ROOT="${COIN_REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA"
RESULT_DIRS=(
  "${MOEMOKA_RESULT_ROOT}/ScienceQA"
  "${MOEMOKA_RESULT_ROOT}/TextVQA"
  "${MOEMOKA_RESULT_ROOT}/ImageNet"
  "${MOEMOKA_RESULT_ROOT}/GQA"
  "${MOEMOKA_RESULT_ROOT}/VizWiz"
  "${MOEMOKA_RESULT_ROOT}/Grounding"
  "${MOEMOKA_RESULT_ROOT}/VQAv2"
  "${MOEMOKA_RESULT_ROOT}/OCRVQA"
)

MODES=${MODES:-"all text visual"}
ENABLE_EVAL=${ENABLE_EVAL:-1}
STAGE_PREFIX=${STAGE_PREFIX:-"CoIN"}
VALID_MODES=(all text visual vision)

START_TASK=${1:-1}
END_TASK=${2:-8}
DRY_RUN=${DRY_RUN:-0}

# Resume / skip controls (set via env vars before calling this script):
#   TRAINED_THROUGH=N   — skip training for tasks 1..N (checkpoint already exists)
#   EVALED_THROUGH=N    — skip ALL online evals for train tasks 1..N
#   SKIP_MODES="a b"    — skip these eval modes entirely (space-separated)
TRAINED_THROUGH=${TRAINED_THROUGH:-0}
EVALED_THROUGH=${EVALED_THROUGH:-0}
SKIP_MODES=${SKIP_MODES:-""}

METRIC_DIR="${REPO_ROOT}/results/CoIN/LLaVA/metrics"
ONLINE_OUT_CSV="${METRIC_DIR}/continual_online_eval.csv"
FINAL_OUT_CSV="${METRIC_DIR}/continual_final_eval.csv"

if [[ "$START_TASK" -lt 1 || "$END_TASK" -gt 8 || "$START_TASK" -gt "$END_TASK" ]]; then
  echo "Invalid range. Usage: bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh [start_task(1-8)] [end_task(1-8)]"
  exit 1
fi

echo "[CoIN] Run continual training sequence from T${START_TASK} to T${END_TASK}"
cd "${REPO_ROOT}"

# Log directory — one timestamped folder per run
LOG_ROOT="${REPO_ROOT}/logs/LLaVA/CoIN"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="${LOG_ROOT}/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/main.log"

echo "[CoIN] Log dir : ${LOG_DIR}"
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
  local model_path="${COIN_OUTPUT_ROOT}/${TASK_NAMES[$((train_task-1))]}_llava_MoEMoKA_lora"
  local stage="${STAGE_PREFIX}_${phase}_m${mode}_train${train_task}_eval${eval_task}"
  local stage_dir="${result_dir}/${stage}"
  local out_csv

  if [[ "$phase" == "online" ]]; then
    out_csv="$ONLINE_OUT_CSV"
  else
    out_csv="$FINAL_OUT_CSV"
  fi

  # Skip if this mode is in SKIP_MODES
  local skip_mode
  for skip_mode in $SKIP_MODES; do
    if [[ "$mode" == "$skip_mode" ]]; then
      echo "[Eval][$phase] mode=${mode} train=T${train_task} eval=T${eval_task} -> SKIPPED (SKIP_MODES)"
      return
    fi
  done

  echo "[Eval][$phase] mode=${mode}, train=T${train_task}, eval=T${eval_task}"

  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  require_path "$model_path" "trained checkpoint for T${train_task}"

  local eval_log="${LOG_DIR}/${phase}_train${train_task}_${TASK_NAMES[$((train_task-1))]}_eval${eval_task}_${TASK_NAMES[$((eval_task-1))]}_${mode}.log"
  bash "$eval_script" "$stage" "$model_path" "$mode" 2>&1 | tee "$eval_log"
  local acc
  acc=$("${COIN_PYTHON}" "$PARSE_ACC_PY" --stage-dir "$stage_dir")
  echo "[Eval][$phase] mode=${mode} train=T${train_task} eval=T${eval_task} -> acc=${acc}"
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
  task_log="${LOG_DIR}/T${i}_${task_name}_train.log"
  echo "[CoIN] >>> Training task T${i} (${task_name}) via ${script}"
  echo "[CoIN] Task log: ${task_log}"

  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$ENABLE_EVAL" == "1" ]] && [[ "$i" -gt "$EVALED_THROUGH" ]]; then
      for mode in $MODES; do
        for eval_task in $(seq 1 "$i"); do
          run_eval_once "online" "$i" "$eval_task" "$mode"
        done
      done
    fi
    continue
  fi

  # --- Training ---
  if [[ "$i" -le "$TRAINED_THROUGH" ]]; then
    echo "[CoIN] T${i} (${task_name}) training SKIPPED (TRAINED_THROUGH=${TRAINED_THROUGH})"
  else
    bash "$script" 2>&1 | tee "$task_log"
    echo "[CoIN] <<< Finished T${i} (${task_name})"
    echo "[CoIN] Log saved to: ${task_log}"
  fi

  # --- Online eval ---
  if [[ "$ENABLE_EVAL" == "1" ]]; then
    if [[ "$i" -le "$EVALED_THROUGH" ]]; then
      echo "[CoIN] T${i} online eval SKIPPED (EVALED_THROUGH=${EVALED_THROUGH})"
    else
      for mode in $MODES; do
        for eval_task in $(seq 1 "$i"); do
          run_eval_once "online" "$i" "$eval_task" "$mode"
        done
      done
    fi
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
