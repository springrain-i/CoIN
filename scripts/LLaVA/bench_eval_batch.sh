#!/bin/bash
# Benchmark eval batch inference: measures samples/sec for batch_size=1,4,8,16
# Uses GPU 6,7 only; SciQA checkpoint from moelora.
# Usage: bash scripts/LLaVA/bench_eval_batch.sh [num_samples]
#
# Results are printed at the end and appended to bench_eval_results.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_SAMPLES="${1:-200}"   # number of questions to benchmark on (subset)
GPUS="6,7"               # only use GPU 6 for single-GPU eval benchmark
GPU_SINGLE="6"            # single GPU for benchmark (avoids multi-GPU complexity)

MODEL_PATH="/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN/ScienceQA_llava_MOE_lora"
BASE_MODEL="/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5"
QUESTION_FILE="/data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/ScienceQA/test.json"
IMAGE_FOLDER="/data4/wxl/MoBLoRA-backup/CoIN/cl_dataset"
RESULT_DIR="${REPO_ROOT}/bench_results/eval_batch"
LOG_FILE="${RESULT_DIR}/bench_eval_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${RESULT_DIR}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Eval Batch Inference Benchmark" | tee -a "${LOG_FILE}"
echo "Date: $(date)" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_SINGLE} (single GPU)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "Samples: ${NUM_SAMPLES}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Create a small subset question file for fast benchmarking
SUBSET_FILE="${RESULT_DIR}/sqa_subset_${NUM_SAMPLES}.json"
if [[ ! -f "${SUBSET_FILE}" ]]; then
    python3 -c "
import json, sys
d = json.load(open('${QUESTION_FILE}'))
# Take first NUM_SAMPLES items (mix of image and non-image)
subset = d[:${NUM_SAMPLES}]
json.dump(subset, open('${SUBSET_FILE}', 'w'))
print(f'Created subset: {len(subset)} samples')
with_img = sum(1 for x in subset if 'image' in x)
print(f'  with image: {with_img}, text-only: {len(subset)-with_img}')
"
fi

run_bench() {
    local bsz=$1
    local answers_file="${RESULT_DIR}/answers_bs${bsz}.jsonl"
    local t_start t_end elapsed throughput

    echo "" | tee -a "${LOG_FILE}"
    echo "--- batch_size=${bsz} ---" | tee -a "${LOG_FILE}"

    t_start=$(date +%s%3N)

    CUDA_VISIBLE_DEVICES="${GPU_SINGLE}" python3 -m ETrain.Eval.LLaVA.CoIN.model_vqa_science \
        --model-path "${MODEL_PATH}" \
        --model-base "${BASE_MODEL}" \
        --question-file "${SUBSET_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${answers_file}" \
        --conv-mode vicuna_v1 \
        --temperature 0 \
        --max_new_tokens 10 \
        --merge-lora False \
        --lora-mode all \
        --batch-size "${bsz}" \
        2>&1 | grep -v "^$" | tail -5 | tee -a "${LOG_FILE}"

    t_end=$(date +%s%3N)
    elapsed=$(( t_end - t_start ))
    local n_out
    n_out=$(wc -l < "${answers_file}" 2>/dev/null || echo 0)
    throughput=$(python3 -c "print(f'{${n_out} * 1000 / max(1, ${elapsed}):.2f}')")

    echo "[RESULT] batch_size=${bsz}: ${n_out} samples in ${elapsed}ms → ${throughput} samples/sec" | tee -a "${LOG_FILE}"
    echo "BENCH batch_size=${bsz} elapsed_ms=${elapsed} n_samples=${n_out} throughput_sps=${throughput}" >> "${RESULT_DIR}/bench_eval_results.txt"
}

cd "${REPO_ROOT}"

# Warmup with batch_size=1 (also gives baseline)
run_bench 1
run_bench 4
run_bench 8
run_bench 16

echo "" | tee -a "${LOG_FILE}"
echo "======== Summary ========" | tee -a "${LOG_FILE}"
cat "${RESULT_DIR}/bench_eval_results.txt" | grep "^BENCH" | tail -4 | tee -a "${LOG_FILE}"
echo "Full log: ${LOG_FILE}"
