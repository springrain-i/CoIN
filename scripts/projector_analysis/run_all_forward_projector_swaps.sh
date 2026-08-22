#!/usr/bin/env bash
set -Eeuo pipefail

# Run seven standard-LoRA diagonal projector/eval pairs sequentially on all 8 GPUs.

PROJECTOR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECTOR_SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data4/home/sqx/.conda/envs/coin/bin/python}"
START_EARLY_TASK="${START_EARLY_TASK:-1}"
END_EARLY_TASK="${END_EARLY_TASK:-7}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_CHECKPOINT_ROOT="${PROJECTOR_SWAP_SOURCE_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/LLaVA/CoIN_coin_lora_zero2_gbs128_seed42_20260820_2110}"
DERIVED_CHECKPOINT_ROOT="${PROJECTOR_SWAP_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/LLaVA/CoIN_lora_projector_swap}"
LOG_BASE="${PROJECTOR_SWAP_LOG_ROOT:-${REPO_ROOT}/logs/LLaVA/lora_projector_swap}"
METRICS_BASE="${PROJECTOR_SWAP_METRICS_ROOT:-${REPO_ROOT}/results/CoIN/LLaVA/metrics/lora_projector_swap}"
RESULTS_BASE="${PROJECTOR_SWAP_RESULTS_ROOT:-${REPO_ROOT}/results/CoIN/LLaVA/lora_projector_swap}"
ORCHESTRATOR_ROOT="${LOG_BASE}/all_early/${RUN_TIMESTAMP}"
MATRIX_ROOT="${METRICS_BASE}/all_early/${RUN_TIMESTAMP}"
SINGLE_RUNNER="${PROJECTOR_SCRIPT_DIR}/run_forward_projector_swap_8tasks.sh"
MATRIX_SUMMARIZER="${PROJECTOR_SCRIPT_DIR}/summarize_forward_projector_matrix.py"
PREPARE_SCRIPT="${PROJECTOR_SCRIPT_DIR}/prepare_forward_projector_swap.py"

if (( START_EARLY_TASK < 1 || END_EARLY_TASK > 7 || START_EARLY_TASK > END_EARLY_TASK )); then
  echo "START_EARLY_TASK/END_EARLY_TASK must satisfy 1 <= start <= end <= 7." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "coin Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${ORCHESTRATOR_ROOT}" "${MATRIX_ROOT}"
exec > >(tee -a "${ORCHESTRATOR_ROOT}/orchestrator.log") 2>&1

echo "[ProjectorSwapAll] standard-LoRA forward early-projector diagonal sweep"
echo "[ProjectorSwapAll] early_range=T${START_EARLY_TASK}..T${END_EARLY_TASK}"
echo "[ProjectorSwapAll] final=T8 OCRVQA (not re-evaluated as a projector arm)"
echo "[ProjectorSwapAll] protocol=projector Tn -> eval Tn"
echo "[ProjectorSwapAll] execution=sequential diagonal pairs; 8 GPUs per pair"
echo "[ProjectorSwapAll] mode=all"
echo "[ProjectorSwapAll] eval_batch_size=4"
echo "[ProjectorSwapAll] timestamp=${RUN_TIMESTAMP}"
echo "[ProjectorSwapAll] source_checkpoint_root=${SOURCE_CHECKPOINT_ROOT}"

echo "[ProjectorSwapAll] Prebuilding and validating all requested hybrid checkpoints."
for early_task_id in $(seq "${START_EARLY_TASK}" "${END_EARLY_TASK}"); do
  checkpoint_path="$(
    "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
      --early-task-id "${early_task_id}" \
      --repo-root "${REPO_ROOT}" \
      --checkpoint-root "${SOURCE_CHECKPOINT_ROOT}" \
      --output-root "${DERIVED_CHECKPOINT_ROOT}" \
      --reuse-existing
  )"
  echo "[ProjectorSwapAll] checkpoint T${early_task_id}=${checkpoint_path}"
done

for early_task_id in $(seq "${START_EARLY_TASK}" "${END_EARLY_TASK}"); do
  echo "[ProjectorSwapAll] >>> Start projector T${early_task_id} -> eval T${early_task_id}"
  RUN_TIMESTAMP="${RUN_TIMESTAMP}" \
  DRY_RUN="${DRY_RUN}" \
  PROJECTOR_SWAP_SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT}" \
  PROJECTOR_SWAP_CHECKPOINT_ROOT="${DERIVED_CHECKPOINT_ROOT}" \
  PROJECTOR_SWAP_LOG_ROOT="${LOG_BASE}" \
  PROJECTOR_SWAP_METRICS_ROOT="${METRICS_BASE}" \
  PROJECTOR_SWAP_RESULTS_ROOT="${RESULTS_BASE}" \
  bash "${SINGLE_RUNNER}" "${early_task_id}"
  echo "[ProjectorSwapAll] <<< Complete projector T${early_task_id} -> eval T${early_task_id}"
done

summary_args=(
  --metrics-root "${METRICS_BASE}"
  --run-timestamp "${RUN_TIMESTAMP}"
  --start-early-task "${START_EARLY_TASK}"
  --end-early-task "${END_EARLY_TASK}"
  --output-dir "${MATRIX_ROOT}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  summary_args+=(--allow-incomplete)
fi
"${PYTHON_BIN}" "${MATRIX_SUMMARIZER}" "${summary_args[@]}"

echo "[ProjectorSwapAll] Sweep complete."
echo "[ProjectorSwapAll] diagonal=${MATRIX_ROOT}/projector_eval_diagonal.csv"
echo "[ProjectorSwapAll] summary=${MATRIX_ROOT}/summary.md"
echo "[ProjectorSwapAll] orchestrator_log=${ORCHESTRATOR_ROOT}/orchestrator.log"
