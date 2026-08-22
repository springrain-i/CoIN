#!/usr/bin/env bash
set -Eeuo pipefail

# Build one standard-LoRA forward projector-swap checkpoint and evaluate its matching task.
# Usage: bash scripts/projector_analysis/run_forward_projector_swap_8tasks.sh EARLY_TASK_ID

PROJECTOR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECTOR_SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/LLaVA/Train_MOE/coin_paths.sh"

EARLY_TASK_ID="${1:-}"
if [[ ! "${EARLY_TASK_ID}" =~ ^[1-7]$ ]]; then
  echo "Usage: bash scripts/projector_analysis/run_forward_projector_swap_8tasks.sh EARLY_TASK_ID(1-7)" >&2
  exit 2
fi

TASK_NAMES=(ScienceQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCRVQA)
EVAL_SCRIPTS=(
  "${REPO_ROOT}/scripts/LLaVA/Eval/1_eval_sqa.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/2_eval_textqa.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/3_eval_ImageNet.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/4_eval_gqa.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/5_eval_vizwiz.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/6_eval_grounding.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/7_eval_vqav2.sh"
  "${REPO_ROOT}/scripts/LLaVA/Eval/8_eval_ocrvqa.sh"
)
QUESTION_FILES=(
  "${COIN_INSTR_ROOT}/ScienceQA/test.json"
  "${COIN_INSTR_ROOT}/TextVQA/val.json"
  "${COIN_INSTR_ROOT}/ImageNet/test.json"
  "${COIN_INSTR_ROOT}/GQA/test.json"
  "${COIN_INSTR_ROOT}/VizWiz/val.json"
  "${COIN_INSTR_ROOT}/Grounding/test.json"
  "${COIN_INSTR_ROOT}/VQAv2/val.json"
  "${COIN_INSTR_ROOT}/OCRVQA/test.json"
)

PYTHON_BIN="${PYTHON_BIN:-/data4/home/sqx/.conda/envs/coin/bin/python}"
# Diagonal protocol is locked: projector Tn is evaluated only on task Tn.
START_EVAL_TASK="${EARLY_TASK_ID}"
END_EVAL_TASK="${EARLY_TASK_ID}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
MATERIALIZE_ADAPTER="${MATERIALIZE_ADAPTER:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
EARLY_TASK_NAME="${TASK_NAMES[$((EARLY_TASK_ID - 1))]}"
ARM_ID="early_T${EARLY_TASK_ID}_${EARLY_TASK_NAME}__final_T8_OCRVQA"
RUN_ID="lora_projector_swap_diagonal_${ARM_ID}_evalT${EARLY_TASK_ID}_all_bs4_${RUN_TIMESTAMP}"
SOURCE_CHECKPOINT_ROOT="${PROJECTOR_SWAP_SOURCE_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/LLaVA/CoIN_coin_lora_zero2_gbs128_seed42_20260820_2110}"
DERIVED_CHECKPOINT_ROOT="${PROJECTOR_SWAP_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/LLaVA/CoIN_lora_projector_swap}"
LOG_BASE="${PROJECTOR_SWAP_LOG_ROOT:-${REPO_ROOT}/logs/LLaVA/lora_projector_swap}"
METRICS_BASE="${PROJECTOR_SWAP_METRICS_ROOT:-${REPO_ROOT}/results/CoIN/LLaVA/metrics/lora_projector_swap}"
RESULTS_BASE="${PROJECTOR_SWAP_RESULTS_ROOT:-${REPO_ROOT}/results/CoIN/LLaVA/lora_projector_swap}"
LOG_ROOT="${LOG_BASE}/${ARM_ID}/${RUN_TIMESTAMP}"
RUN_ROOT="${METRICS_BASE}/${ARM_ID}/${RUN_TIMESTAMP}"
RESULT_ROOT="${RESULTS_BASE}/${ARM_ID}/${RUN_TIMESTAMP}"
METRICS_CSV="${RUN_ROOT}/metrics.csv"
STATUS_JSON="${RUN_ROOT}/status.json"
RUN_MANIFEST="${RUN_ROOT}/run_manifest.json"
PREPARE_SCRIPT="${PROJECTOR_SCRIPT_DIR}/prepare_forward_projector_swap.py"
SUMMARIZE_SCRIPT="${PROJECTOR_SCRIPT_DIR}/summarize_forward_projector_swap.py"
PARSE_ACC_PY="${REPO_ROOT}/scripts/LLaVA/Eval/parse_accuracy.py"

# Formal batch-inference protocol from AGENTS.md/docs/batch_inference_fix.md.
# These are deliberately constants rather than user-overridable defaults.
export COIN_EVAL_BATCH_SIZE=4
export COIN_USE_SDPA_PATCH=1
readonly LORA_MODE="all"
readonly EXPECTED_CHUNKS=8

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "coin Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
if [[ "$(command -v python)" != "${PYTHON_BIN}" ]]; then
  echo "python does not resolve to the locked coin interpreter: $(command -v python)" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}" "${RUN_ROOT}" "${RESULT_ROOT}"
MAIN_LOG="${LOG_ROOT}/main.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

update_status() {
  local task_id="$1"
  local task_name="$2"
  local state="$3"
  local detail="${4:-}"
  "${PYTHON_BIN}" - "${STATUS_JSON}" "${RUN_ID}" "${task_id}" "${task_name}" "${state}" "${detail}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
run_id, task_id, task_name, state, detail = sys.argv[2:]
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
else:
    data = {"run_id": run_id, "tasks": {}}
data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
data["tasks"][task_id] = {
    "task_name": task_name,
    "state": state,
    "detail": detail,
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

check_gpu_idle() {
  if [[ "${DRY_RUN}" == "1" || "${ALLOW_BUSY_GPU}" == "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for the 8-GPU preflight." >&2
    exit 1
  fi
  local gpu_count busy
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  if [[ "${gpu_count}" -lt 8 ]]; then
    echo "Expected at least 8 visible GPUs, found ${gpu_count}." >&2
    exit 1
  fi
  busy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "${busy}" ]]; then
    echo "GPU compute processes are active; refusing to start:" >&2
    echo "${busy}" >&2
    exit 1
  fi
}

prepare_args=(
  --early-task-id "${EARLY_TASK_ID}"
  --repo-root "${REPO_ROOT}"
  --checkpoint-root "${SOURCE_CHECKPOINT_ROOT}"
  --output-root "${DERIVED_CHECKPOINT_ROOT}"
  --reuse-existing
)
if [[ "${MATERIALIZE_ADAPTER}" == "1" ]]; then
  prepare_args+=(--materialize-adapter)
fi

echo "[ProjectorSwap] run_id=${RUN_ID}"
echo "[ProjectorSwap] early=T${EARLY_TASK_ID} ${EARLY_TASK_NAME}"
echo "[ProjectorSwap] final=T8 OCRVQA"
echo "[ProjectorSwap] adaptation_method=standard_lora"
echo "[ProjectorSwap] mode=${LORA_MODE}"
echo "[ProjectorSwap] COIN_EVAL_BATCH_SIZE=${COIN_EVAL_BATCH_SIZE}"
echo "[ProjectorSwap] COIN_USE_SDPA_PATCH=${COIN_USE_SDPA_PATCH}"
echo "[ProjectorSwap] protocol=projector T${EARLY_TASK_ID} -> eval T${EARLY_TASK_ID}"
echo "[ProjectorSwap] eval_target=T${EARLY_TASK_ID} ${EARLY_TASK_NAME}"
echo "[ProjectorSwap] python=${PYTHON_BIN}"
echo "[ProjectorSwap] derived_checkpoint_root=${DERIVED_CHECKPOINT_ROOT}"
echo "[ProjectorSwap] source_checkpoint_root=${SOURCE_CHECKPOINT_ROOT}"
echo "[ProjectorSwap] result_root=${RESULT_ROOT}"
echo "[ProjectorSwap] log_root=${LOG_ROOT}"
echo "[ProjectorSwap] run_root=${RUN_ROOT}"

HYBRID_CHECKPOINT="$("${PYTHON_BIN}" "${PREPARE_SCRIPT}" "${prepare_args[@]}")"
if [[ ! -d "${HYBRID_CHECKPOINT}" ]]; then
  echo "Hybrid checkpoint was not created: ${HYBRID_CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${RUN_MANIFEST}" "${RUN_ID}" "${ARM_ID}" "${EARLY_TASK_ID}" \
  "${EARLY_TASK_NAME}" "${HYBRID_CHECKPOINT}" "${START_EVAL_TASK}" "${END_EVAL_TASK}" \
  "${SOURCE_CHECKPOINT_ROOT}" "${RESULT_ROOT}" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
run_id, arm_id, early_id, early_name, checkpoint, start, end, source_root, result_root = sys.argv[2:]
repo = Path.cwd()
git_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
).stdout.strip()
data = {
    "schema_version": 2,
    "run_id": run_id,
    "arm_id": arm_id,
    "early_task_id": int(early_id),
    "early_task_name": early_name,
    "final_task_id": 8,
    "final_task_name": "OCRVQA",
    "adaptation_method": "standard_lora",
    "source_checkpoint_root": source_root,
    "hybrid_checkpoint": checkpoint,
    "swap_manifest": str(Path(checkpoint) / "swap_manifest.json"),
    "mode": "all",
    "eval_batch_size": 4,
    "use_sdpa_patch": True,
    "expected_chunks": 8,
    "result_root": result_root,
    "start_eval_task": int(start),
    "end_eval_task": int(end),
    "python": sys.executable,
    "git_commit": git_commit,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

if [[ ! -f "${METRICS_CSV}" ]]; then
  echo "run_id,projector_task_id,projector_task_name,final_task_id,eval_task_id,eval_task_name,mode,batch_size,accuracy,expected_predictions,actual_predictions,checkpoint_path,result_dir,log_path,status" > "${METRICS_CSV}"
fi

check_gpu_idle

for eval_task_id in $(seq "${START_EVAL_TASK}" "${END_EVAL_TASK}"); do
  idx=$((eval_task_id - 1))
  eval_task_name="${TASK_NAMES[$idx]}"
  eval_script="${EVAL_SCRIPTS[$idx]}"
  question_file="${QUESTION_FILES[$idx]}"
  stage="predictions"
  stage_dir="${RESULT_ROOT}/${stage}"
  task_log="${LOG_ROOT}/eval_T${eval_task_id}_${eval_task_name}.log"

  require_path "${eval_script}" "eval script for T${eval_task_id}"
  require_path "${question_file}" "question file for T${eval_task_id}"

  echo "[ProjectorSwap][Eval] T${eval_task_id} ${eval_task_name}"
  echo "[ProjectorSwap][Eval] checkpoint=${HYBRID_CHECKPOINT}"
  echo "[ProjectorSwap][Eval] stage_dir=${stage_dir}"
  echo "[ProjectorSwap][Eval] log=${task_log}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[ProjectorSwap][DryRun] COIN_EVAL_RESULT_ROOT=${RESULT_ROOT} bash ${eval_script} ${stage} ${HYBRID_CHECKPOINT} all"
    update_status "${eval_task_id}" "${eval_task_name}" "dry_run" "command validated"
    continue
  fi

  if [[ -e "${stage_dir}" ]]; then
    if [[ -s "${stage_dir}/merge.jsonl" ]] \
      && accuracy="$("${PYTHON_BIN}" "${PARSE_ACC_PY}" --stage-dir "${stage_dir}" 2>/dev/null)"; then
      expected="$("${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${question_file}")"
      actual="$(wc -l < "${stage_dir}/merge.jsonl")"
      if [[ "${actual}" -eq "${expected}" ]]; then
        echo "[ProjectorSwap][Skip] Complete existing result: T${eval_task_id}, acc=${accuracy}"
        update_status "${eval_task_id}" "${eval_task_name}" "complete" "reused complete result"
        if ! awk -F, -v id="${eval_task_id}" '$5 == id {found=1} END {exit !found}' "${METRICS_CSV}"; then
          echo "${RUN_ID},${EARLY_TASK_ID},${EARLY_TASK_NAME},8,${eval_task_id},${eval_task_name},all,4,${accuracy},${expected},${actual},${HYBRID_CHECKPOINT},${stage_dir},${task_log},complete" >> "${METRICS_CSV}"
        fi
        continue
      fi
    fi
    update_status "${eval_task_id}" "${eval_task_name}" "incomplete" "existing stage failed validation"
    echo "Existing stage is incomplete; refusing to overwrite: ${stage_dir}" >&2
    exit 1
  fi

  update_status "${eval_task_id}" "${eval_task_name}" "running" "eval started"
  mkdir -p "${stage_dir}"
  if ! COIN_EVAL_RESULT_ROOT="${RESULT_ROOT}" bash "${eval_script}" "${stage}" "${HYBRID_CHECKPOINT}" "${LORA_MODE}" 2>&1 | tee "${task_log}"; then
    update_status "${eval_task_id}" "${eval_task_name}" "failed" "eval script returned non-zero"
    exit 1
  fi

  for chunk_idx in $(seq 0 $((EXPECTED_CHUNKS - 1))); do
    chunk_file="${stage_dir}/${EXPECTED_CHUNKS}_${chunk_idx}.jsonl"
    if [[ ! -s "${chunk_file}" ]]; then
      update_status "${eval_task_id}" "${eval_task_name}" "failed" "missing chunk ${chunk_file}"
      echo "Missing or empty eval chunk: ${chunk_file}" >&2
      exit 1
    fi
  done

  merged_file="${stage_dir}/merge.jsonl"
  if [[ ! -s "${merged_file}" ]]; then
    update_status "${eval_task_id}" "${eval_task_name}" "failed" "missing merged predictions"
    echo "Missing or empty merged predictions: ${merged_file}" >&2
    exit 1
  fi

  expected="$("${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${question_file}")"
  actual="$(wc -l < "${merged_file}")"
  if [[ "${actual}" -ne "${expected}" ]]; then
    update_status "${eval_task_id}" "${eval_task_name}" "failed" "prediction count ${actual}/${expected}"
    echo "Prediction count mismatch for T${eval_task_id}: expected=${expected}, actual=${actual}" >&2
    exit 1
  fi
  if rg -n "Traceback|CUDA out of memory|RuntimeError" "${task_log}" >/dev/null 2>&1; then
    update_status "${eval_task_id}" "${eval_task_name}" "failed" "error marker found in log"
    echo "Error marker found in task log: ${task_log}" >&2
    exit 1
  fi

  accuracy="$("${PYTHON_BIN}" "${PARSE_ACC_PY}" --stage-dir "${stage_dir}")"
  echo "${RUN_ID},${EARLY_TASK_ID},${EARLY_TASK_NAME},8,${eval_task_id},${eval_task_name},all,4,${accuracy},${expected},${actual},${HYBRID_CHECKPOINT},${stage_dir},${task_log},complete" >> "${METRICS_CSV}"
  update_status "${eval_task_id}" "${eval_task_name}" "complete" "accuracy=${accuracy}, predictions=${actual}/${expected}"
  echo "[ProjectorSwap][Complete] T${eval_task_id} ${eval_task_name}: acc=${accuracy}, predictions=${actual}/${expected}"
done

"${PYTHON_BIN}" "${SUMMARIZE_SCRIPT}" --run-dir "${RUN_ROOT}"
echo "[ProjectorSwap] Finished diagonal eval pair T${EARLY_TASK_ID} -> T${EARLY_TASK_ID}."
echo "[ProjectorSwap] metrics=${METRICS_CSV}"
echo "[ProjectorSwap] summary=${RUN_ROOT}/summary.md"
echo "[ProjectorSwap] logs=${LOG_ROOT}"
