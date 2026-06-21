#!/bin/bash
set -euo pipefail

# Run MoELoRA OCRVQA as an isolated single-task rerun.
# This keeps checkpoints/logs/metrics separate from the historical CoIN_single run.

USER_COIN_OUTPUT_ROOT="${COIN_OUTPUT_ROOT-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RUN_ID="${RUN_ID:-$(date "+%Y%m%d_%H%M%S")}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ -n "${USER_COIN_OUTPUT_ROOT}" ]]; then
  export COIN_OUTPUT_ROOT="${USER_COIN_OUTPUT_ROOT}"
else
  export COIN_OUTPUT_ROOT="${REPO_ROOT}/checkpoints/LLaVA/CoIN_single_rerun_ocrvqa_${RUN_ID}"
fi

source "${SCRIPT_DIR}/coin_paths.sh"

TASK_ID=8
TASK_NAME="OCRVQA"
TRAIN_SCRIPT="${SCRIPT_DIR}/8_OCRVQA.sh"
EVAL_SCRIPT="${REPO_ROOT}/scripts/LLaVA/Eval/8_eval_ocrvqa.sh"
RESULT_DIR="${REPO_ROOT}/results/CoIN/LLaVA/OCRVQA"

DRY_RUN="${DRY_RUN:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
MODES="${MODES:-all text visual}"
REGIME="${REGIME:-single}"
STAGE_PREFIX="${STAGE_PREFIX:-CoIN_single_ocrvqa_rerun_${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/LLaVA/CoIN_single_rerun/OCRVQA_${RUN_ID}}"
OUT_CSV="${OUT_CSV:-${REPO_ROOT}/results/CoIN/LLaVA/metrics/moelora_ocrvqa_single_rerun_${RUN_ID}.csv}"
STATE_DIR="${LOG_ROOT}/.state"

TRAIN_JSON="${COIN_INSTR_ROOT}/OCRVQA/train.json"
TEST_JSON="${COIN_INSTR_ROOT}/OCRVQA/test.json"
MODEL_PATH="${COIN_OUTPUT_ROOT}/OCRVQA_llava_MOE_lora"

cd "${REPO_ROOT}"

current_branch="$(git branch --show-current)"
if [[ "${current_branch}" != "moelora" ]]; then
  echo "[CoIN][Error] Expected branch moelora, got ${current_branch}" >&2
  exit 1
fi

require_path "${COIN_BASE_MODEL}" "base model"
require_path "${COIN_VISION_TOWER}" "vision tower"
require_path "${TRAIN_JSON}" "OCRVQA train json"
require_path "${TEST_JSON}" "OCRVQA test json"
require_path "${TRAIN_SCRIPT}" "OCRVQA train script"
require_path "${EVAL_SCRIPT}" "OCRVQA eval script"

gpu_list="${CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -r -a GPUS <<< "${gpu_list}"
GPU_COUNT="${#GPUS[@]}"
PER_DEVICE_BS="$(awk '/--per_device_train_batch_size/{print $2; exit}' "${TRAIN_SCRIPT}")"
GRAD_ACCUM="$(awk '/--gradient_accumulation_steps/{print $2; exit}' "${TRAIN_SCRIPT}")"
EFFECTIVE_BS=$((PER_DEVICE_BS * GPU_COUNT * GRAD_ACCUM))

if [[ "${GPU_COUNT}" -ne 8 || "${PER_DEVICE_BS}" -ne 4 || "${GRAD_ACCUM}" -ne 8 || "${EFFECTIVE_BS}" -ne 256 ]]; then
  echo "[CoIN][Error] Unexpected OCRVQA effective batch size." >&2
  echo "[CoIN][Error] per_device=${PER_DEVICE_BS}, gpus=${GPU_COUNT}, grad_accum=${GRAD_ACCUM}, effective=${EFFECTIVE_BS}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${STATE_DIR}" "$(dirname "${OUT_CSV}")"

if [[ ! -f "${OUT_CSV}" ]]; then
  echo "regime,mode,train_task,eval_task,accuracy,result_stage_dir" > "${OUT_CSV}"
fi

normalize_task_list() {
  echo "$1" | tr ',;' '  '
}

append_metric_once() {
  local mode="$1"
  local accuracy="$2"
  local stage_dir="$3"
  local line="${REGIME},${mode},${TASK_ID},${TASK_ID},${accuracy},${stage_dir}"

  if [[ -f "${OUT_CSV}" ]] && grep -Fq ",${stage_dir}" "${OUT_CSV}"; then
    return
  fi

  echo "${line}" >> "${OUT_CSV}"
}

tmp_script="$(mktemp)"
cleanup() {
  rm -f "${tmp_script}"
}
trap cleanup EXIT

sed '/previous_task_model_path/d' "${TRAIN_SCRIPT}" | \
  sed '/PREVIOUS_TASK_MODEL_PATH/d' | \
  sed '/require_path.*previous task checkpoint/d' | \
  sed "s|^SCRIPT_DIR=.*$|SCRIPT_DIR=\"${SCRIPT_DIR}\"|" \
  > "${tmp_script}"
chmod +x "${tmp_script}"

train_marker="${STATE_DIR}/T${TASK_ID}.train.done"
train_log="${LOG_ROOT}/${TASK_NAME}_${RUN_ID}.train.log"

echo "[CoIN] OCRVQA single rerun id: ${RUN_ID}"
echo "[CoIN] Branch: ${current_branch}"
echo "[CoIN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[CoIN] Effective bs: ${PER_DEVICE_BS} * ${GPU_COUNT} * ${GRAD_ACCUM} = ${EFFECTIVE_BS}"
echo "[CoIN] Checkpoint root: ${COIN_OUTPUT_ROOT}"
echo "[CoIN] Log root: ${LOG_ROOT}"
echo "[CoIN] Metrics csv: ${OUT_CSV}"
echo "[CoIN] Stage prefix: ${STAGE_PREFIX}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[CoIN] DRY_RUN=1; no train/eval commands will be executed."
  echo "[CoIN] Train command: bash ${tmp_script}"
  for mode in ${MODES}; do
    stage="${STAGE_PREFIX}_${REGIME}_m${mode}_train${TASK_ID}_eval${TASK_ID}"
    echo "[CoIN] Eval command: bash ${EVAL_SCRIPT} ${stage} ${MODEL_PATH} ${mode}"
  done
  exit 0
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if [[ "${FORCE_TRAIN}" != "1" && -f "${train_marker}" ]]; then
    echo "[CoIN] Skip training because marker exists: ${train_marker}"
  elif [[ "${FORCE_TRAIN}" != "1" && -d "${MODEL_PATH}" ]]; then
    echo "[CoIN] Skip training because checkpoint already exists: ${MODEL_PATH}"
    touch "${train_marker}"
  else
    echo "[CoIN] >>> Train T${TASK_ID} (${TASK_NAME}) from base model"
    bash "${tmp_script}" 2>&1 | tee "${train_log}"
    require_path "${MODEL_PATH}" "OCRVQA single rerun checkpoint"
    touch "${train_marker}"
    echo "[CoIN] <<< Train complete T${TASK_ID} (${TASK_NAME})"
    echo "[CoIN] Train log saved to: ${train_log}"
  fi
else
  echo "[CoIN] Training disabled via RUN_TRAIN=0"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  require_path "${MODEL_PATH}" "OCRVQA single rerun checkpoint"

  for mode in ${MODES}; do
    eval_marker="${STATE_DIR}/T${TASK_ID}.eval.${mode}.done"
    eval_log="${LOG_ROOT}/${TASK_NAME}_${mode}_${RUN_ID}.eval.log"
    stage="${STAGE_PREFIX}_${REGIME}_m${mode}_train${TASK_ID}_eval${TASK_ID}"
    stage_dir="${RESULT_DIR}/${stage}"

    if [[ "${FORCE_EVAL}" != "1" && -f "${eval_marker}" ]]; then
      echo "[CoIN] Skip eval mode=${mode} because marker exists: ${eval_marker}"
      continue
    fi

    echo "[CoIN] >>> Eval T${TASK_ID} (${TASK_NAME}) mode=${mode}"
    bash "${EVAL_SCRIPT}" "${stage}" "${MODEL_PATH}" "${mode}" 2>&1 | tee "${eval_log}"
    acc="$(python "${REPO_ROOT}/scripts/LLaVA/Eval/parse_accuracy.py" --stage-dir "${stage_dir}")"
    append_metric_once "${mode}" "${acc}" "${stage_dir}"
    touch "${eval_marker}"
    echo "[CoIN] <<< Eval complete T${TASK_ID} (${TASK_NAME}) mode=${mode} -> ${acc}"
    echo "[CoIN] Eval log saved to: ${eval_log}"
  done
else
  echo "[CoIN] Eval disabled via RUN_EVAL=0"
fi

echo "[CoIN] OCRVQA single rerun complete."
echo "[CoIN] All logs saved to: ${LOG_ROOT}/"
