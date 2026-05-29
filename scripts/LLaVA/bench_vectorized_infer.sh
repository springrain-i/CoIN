#!/bin/bash
# Benchmark: vectorized vs loop LoRA forward in inference (batch_size=1).
# Isolates the vectorized einsum contribution from eval batch inference.
# Runs on a single GPU to eliminate parallelism as a confound.
#
# Usage: bash scripts/LLaVA/bench_vectorized_infer.sh [num_samples] [gpu_id]
#   num_samples: default 300
#   gpu_id:      default 0
#
# Results: bench_results/vectorized_infer/bench_vectorized_infer_results.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PATH="/data4/home/sqx/.conda/envs/coin/bin:${PATH}"

NUM_SAMPLES="${1:-300}"
GPU_ID="${2:-0}"

MODEL_PATH="/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN/ScienceQA_llava_MOE_lora"
BASE_MODEL="/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5"
QUESTION_FILE="/data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/ScienceQA/test.json"
IMAGE_FOLDER="/data4/wxl/MoBLoRA-backup/CoIN/cl_dataset"

RESULT_DIR="${REPO_ROOT}/bench_results/vectorized_infer"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULT_DIR}/bench_vectorized_infer_${TIMESTAMP}.log"
RESULT_FILE="${RESULT_DIR}/bench_vectorized_infer_results.txt"

mkdir -p "${RESULT_DIR}"

echo "========================================" | tee "${LOG_FILE}"
echo "Vectorized LoRA Inference Benchmark" | tee -a "${LOG_FILE}"
echo "Date: $(date)" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID} (single GPU, batch_size=1)" | tee -a "${LOG_FILE}"
echo "Samples: ${NUM_SAMPLES}" | tee -a "${LOG_FILE}"
echo "Comparing: COIN_USE_VECTORIZED_LORA=0 (loop) vs =1 (einsum)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Create 300-sample subset (reuse if exists)
SUBSET_FILE="${RESULT_DIR}/sqa_subset_${NUM_SAMPLES}.json"
if [[ ! -f "${SUBSET_FILE}" ]]; then
    python3 -c "
import json
d = json.load(open('${QUESTION_FILE}'))
subset = d[:${NUM_SAMPLES}]
json.dump(subset, open('${SUBSET_FILE}', 'w'))
print(f'Created subset: {len(subset)} samples')
with_img = sum(1 for x in subset if 'image' in x)
print(f'  with image: {with_img}, text-only: {len(subset)-with_img}')
"
fi

# ── single-condition runner ────────────────────────────────────────────────
run_infer() {
    local vectorized="$1"   # 0 = loop, 1 = vectorized
    local tag
    [[ "${vectorized}" == "0" ]] && tag="loop" || tag="vectorized"

    local answers_file="${RESULT_DIR}/answers_${tag}.jsonl"
    local t_start t_end elapsed throughput n_out

    echo "" | tee -a "${LOG_FILE}"
    echo "--- lora_mode=${tag} (COIN_USE_VECTORIZED_LORA=${vectorized}) ---" | tee -a "${LOG_FILE}"

    t_start=$(date +%s%3N)

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    COIN_USE_VECTORIZED_LORA="${vectorized}" \
    python3 -m ETrain.Eval.LLaVA.CoIN.model_vqa_science \
        --model-path  "${MODEL_PATH}" \
        --model-base  "${BASE_MODEL}" \
        --question-file "${SUBSET_FILE}" \
        --image-folder  "${IMAGE_FOLDER}" \
        --answers-file  "${answers_file}" \
        --conv-mode vicuna_v1 \
        --temperature 0 \
        --max_new_tokens 10 \
        --merge-lora False \
        --lora-mode all \
        --batch-size 1 \
        2>&1 | grep -v "^$" | tee -a "${LOG_FILE}"

    t_end=$(date +%s%3N)
    elapsed=$(( t_end - t_start ))
    n_out=$(wc -l < "${answers_file}" 2>/dev/null || echo 0)
    throughput=$(python3 -c "print(f'{${n_out} * 1000 / max(1, ${elapsed}):.3f}')")

    echo "" | tee -a "${LOG_FILE}"
    echo "[RESULT] lora_mode=${tag}: ${n_out} samples in ${elapsed}ms → ${throughput} sps" | tee -a "${LOG_FILE}"
    echo "BENCH_INFER mode=${tag} vectorized=${vectorized} gpu=${GPU_ID} n_samples=${n_out} elapsed_ms=${elapsed} throughput_sps=${throughput}" >> "${RESULT_FILE}"
}

# ── write result file header ───────────────────────────────────────────────
{
echo "========================================"
echo "Vectorized LoRA Inference Benchmark"
echo "Date: $(date)"
echo "GPU: ${GPU_ID}, batch_size=1, samples: ${NUM_SAMPLES}"
echo "========================================"
} | tee "${RESULT_FILE}"

cd "${REPO_ROOT}"

# Round 1: baseline loop
run_infer 0

# Round 2: vectorized einsum
run_infer 1

# ── summary ───────────────────────────────────────────────────────────────
echo "" | tee -a "${LOG_FILE}" "${RESULT_FILE}"
echo "======== Final Summary ========" | tee -a "${LOG_FILE}" "${RESULT_FILE}"

python3 - <<'EOF' | tee -a "${LOG_FILE}" "${RESULT_FILE}"
import re, pathlib

lines = pathlib.Path("bench_results/vectorized_infer/bench_vectorized_infer_results.txt").read_text().splitlines()
rows = {}
for l in lines:
    m = re.search(r'mode=(\S+).*throughput_sps=(\S+)', l)
    if m:
        rows[m.group(1)] = float(m.group(2))

if 'loop' in rows and 'vectorized' in rows:
    loop_sps  = rows['loop']
    vec_sps   = rows['vectorized']
    speedup   = vec_sps / loop_sps
    print(f"  loop:       {loop_sps:.3f} sps")
    print(f"  vectorized: {vec_sps:.3f} sps")
    print(f"  speedup:    {speedup:.3f}x")
else:
    print("  (incomplete results)")
EOF

echo ""
echo "Full log:    ${LOG_FILE}"
echo "Results:     ${RESULT_FILE}"
