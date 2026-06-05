#!/bin/bash
set -euo pipefail

# Keep user-provided output root if explicitly passed from environment.
USER_COIN_OUTPUT_ROOT="${COIN_OUTPUT_ROOT-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

# Run independent single-task training for CoIN tasks 1..8.
# It reuses each Train_MoEMoKA task script but removes previous_task_model_path
# so every task starts from base model instead of continual checkpoint.
# After each task finishes, the script immediately runs three evals.
#
# Usage:
#   bash scripts/LLaVA/Train_MoEMoKA/run_coin_single.sh [start_task(1-8)] [end_task(1-8)]
# Optional env:
#   DRY_RUN=1
#   RUN_TRAIN=0
#   RUN_EVAL=0
#   SKIP_TRAIN_TASKS="1 3"
#   SKIP_EVAL_TASKS="2"
#   FORCE_TRAIN=1
#   FORCE_EVAL=1

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

EVAL_SCRIPTS=(
  "${SCRIPT_DIR}/eval/1_eval_sqa.sh"
  "${SCRIPT_DIR}/eval/2_eval_textqa.sh"
  "${SCRIPT_DIR}/eval/3_eval_imagenet.sh"
  "${SCRIPT_DIR}/eval/4_eval_gqa.sh"
  "${SCRIPT_DIR}/eval/5_eval_vizwiz.sh"
  "${SCRIPT_DIR}/eval/6_eval_grounding.sh"
  "${SCRIPT_DIR}/eval/7_eval_vqav2.sh"
  "${SCRIPT_DIR}/eval/8_eval_ocrvqa.sh"
)

# Must match RESULT_ROOT/{Task} in scripts/LLaVA/Train_MoEMoKA/eval/eval_common.sh
RESULT_DIRS=(
  "${REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/ScienceQA"
  "${REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/TextVQA"
  "${REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/ImageNet"
  "${REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/GQA"
  "${REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/VizWiz"
  "${REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/Grounding"
  "${REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/VQAv2"
  "${REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/OCRVQA"
)

START_TASK=${1:-1}
END_TASK=${2:-8}
DRY_RUN=${DRY_RUN:-0}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_EVAL=${FORCE_EVAL:-0}
SKIP_TRAIN_TASKS=${SKIP_TRAIN_TASKS:-""}
SKIP_EVAL_TASKS=${SKIP_EVAL_TASKS:-""}
MODES=${MODES:-"all text visual"}
STAGE_PREFIX=${STAGE_PREFIX:-"MoEMoKA_single"}
REGIME=${REGIME:-"single"}
OUT_CSV=${OUT_CSV:-"results/CoIN/LLaVA/metrics/MoEMoKA_single_eval_matrix.csv"}

if [[ "$START_TASK" -lt 1 || "$END_TASK" -gt 8 || "$START_TASK" -gt "$END_TASK" ]]; then
  echo "Invalid range. Usage: bash scripts/LLaVA/Train_MoEMoKA/run_coin_single.sh [start_task(1-8)] [end_task(1-8)]"
  exit 1
fi

echo "[CoIN] Run single-task training from T${START_TASK} to T${END_TASK}"
cd "${REPO_ROOT}"

# Isolated single-task outputs should not overlap continual outputs.
if [[ -n "${USER_COIN_OUTPUT_ROOT}" ]]; then
  export COIN_OUTPUT_ROOT="${USER_COIN_OUTPUT_ROOT}"
else
  export COIN_OUTPUT_ROOT="${REPO_ROOT}/checkpoints/LLaVA/CoIN_MoEMoKA_single"
fi

# Log and state directories for resumable runs.
LOG_ROOT="${REPO_ROOT}/logs/LLaVA/CoIN_MoEMoKA_single"
STATE_DIR="${LOG_ROOT}/.state"
mkdir -p "${LOG_ROOT}" "${STATE_DIR}" "${REPO_ROOT}/results/CoIN/LLaVA/metrics"

if [[ ! -f "${OUT_CSV}" ]]; then
  echo "regime,mode,train_task,eval_task,accuracy,result_stage_dir" > "${OUT_CSV}"
fi

# Timestamp for this run
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

normalize_task_list() {
  echo "$1" | tr ',;' '  '
}

task_in_list() {
  local list
  local task_id="$2"
  list="$(normalize_task_list "$1")"
  [[ " ${list} " == *" ${task_id} "* ]]
}

append_metric_once() {
  local mode="$1"
  local train_task="$2"
  local eval_task="$3"
  local accuracy="$4"
  local stage_dir="$5"
  local line="${REGIME},${mode},${train_task},${eval_task},${accuracy},${stage_dir}"

  if [[ -f "${OUT_CSV}" ]] && grep -Fq ",${stage_dir}" "${OUT_CSV}"; then
    return
  fi

  echo "${line}" >> "${OUT_CSV}"
}

resolve_single_model_path() {
  local task_name="$1"
  local candidate="${COIN_OUTPUT_ROOT}/${task_name}_llava_MoEMoKA_lora"
  if [[ -d "$candidate" ]]; then
    echo "$candidate"
    return
  fi

  # Backward compatibility for legacy lowercase folder name.
  if [[ "$task_name" == "VQAv2" ]]; then
    candidate="${COIN_OUTPUT_ROOT}/vqav2_llava_MoEMoKA_lora"
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  fi

  echo "${COIN_OUTPUT_ROOT}/${task_name}_llava_MoEMoKA_lora"
}

for i in $(seq "$START_TASK" "$END_TASK"); do
  src="${TASK_SCRIPTS[$((i-1))]}"
  task_name="${TASK_NAMES[$((i-1))]}"
  eval_script="${EVAL_SCRIPTS[$((i-1))]}"
  result_dir="${RESULT_DIRS[$((i-1))]}"
  src_dir="$(cd "$(dirname "$src")" && pwd)"
  tmp_script=$(mktemp)
  train_marker="${STATE_DIR}/T${i}.train.done"
  LOG_FILE="${LOG_ROOT}/${task_name}_${TIMESTAMP}.log"

  # Remove continual dependency so every task starts from base model.
  sed '/previous_task_model_path/d' "$src" | \
    sed '/PREVIOUS_TASK_MODEL_PATH/d' | \
    sed '/require_path.*previous task checkpoint/d' | \
    sed "s|^SCRIPT_DIR=.*$|SCRIPT_DIR=\"${src_dir}\"|" \
    > "$tmp_script"

  chmod +x "$tmp_script"
  echo "[CoIN] >>> Single-task T${i} (${task_name}) via ${src} (tmp=${tmp_script})"
  echo "[CoIN] Logging to: ${LOG_FILE}"

  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ "$RUN_TRAIN" == "1" && "$FORCE_TRAIN" != "1" && -f "$train_marker" ]]; then
      echo "[CoIN] Skip training T${i} (${task_name}) because marker exists: ${train_marker}"
    elif [[ "$RUN_TRAIN" == "1" ]]; then
      if task_in_list "$SKIP_TRAIN_TASKS" "$i"; then
        echo "[CoIN] Skip training T${i} (${task_name}) because it is listed in SKIP_TRAIN_TASKS"
      else
        bash "$tmp_script" 2>&1 | tee "$LOG_FILE"
        touch "$train_marker"
        echo "[CoIN] <<< Finished single-task T${i} (${task_name})"
        echo "[CoIN] Log saved to: ${LOG_FILE}"
      fi
    else
      echo "[CoIN] Training disabled for T${i} (${task_name}) via RUN_TRAIN=0"
    fi

    if [[ "$RUN_EVAL" == "1" ]]; then
      model_path="$(resolve_single_model_path "$task_name")"
      require_path "$model_path" "single-task checkpoint for T${i}"

      for mode in $MODES; do
        if task_in_list "$SKIP_EVAL_TASKS" "$i"; then
          echo "[CoIN] Skip eval T${i} (${task_name}) for mode=${mode} because it is listed in SKIP_EVAL_TASKS"
          continue
        fi

        eval_marker="${STATE_DIR}/T${i}.eval.${mode}.done"
        eval_log="${LOG_ROOT}/${task_name}_${mode}_${TIMESTAMP}.eval.log"
        stage="${STAGE_PREFIX}_${REGIME}_m${mode}_train${i}_eval${i}"
        stage_dir="${result_dir}/${stage}"

        if [[ "$FORCE_EVAL" != "1" && -f "$eval_marker" ]]; then
          echo "[CoIN] Skip eval T${i} (${task_name}) mode=${mode} because marker exists: ${eval_marker}"
          continue
        fi

        echo "[CoIN] >>> Eval T${i} (${task_name}) mode=${mode}"
        bash "$eval_script" "$stage" "$model_path" "$mode" 2>&1 | tee "$eval_log"
        acc=$(python "${REPO_ROOT}/scripts/LLaVA/Eval/parse_accuracy.py" --stage-dir "$stage_dir")
        append_metric_once "$mode" "$i" "$i" "$acc" "$stage_dir"
        touch "$eval_marker"
        echo "[CoIN] <<< Eval complete T${i} (${task_name}) mode=${mode} -> ${acc}"
        echo "[CoIN] Eval log saved to: ${eval_log}"
      done
    else
      echo "[CoIN] Eval disabled for T${i} (${task_name}) via RUN_EVAL=0"
    fi
  fi

  rm -f "$tmp_script"
done

echo "[CoIN] Single-task training sequence complete."
echo "[CoIN] All logs saved to: ${LOG_ROOT}/"
