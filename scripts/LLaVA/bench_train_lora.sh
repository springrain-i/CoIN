#!/bin/bash
# Training micro-benchmark: measures training step time with/without vectorized LoRA.
# Runs 50 steps on a tiny SciQA subset. All 8 GPUs (0-7), ZeRO-2.
# Usage: bash scripts/LLaVA/bench_train_lora.sh [num_samples] [max_steps]
#
# Env vars:
#   COIN_USE_VECTORIZED_LORA=0  -> disable vectorized LoRA (test baseline loop)
#   COIN_USE_VECTORIZED_LORA=1  -> enable vectorized LoRA (test optimization)

set -euo pipefail

# Make coin env tools (ninja, deepspeed) available without requiring conda activate
export PATH="/data4/home/sqx/.conda/envs/coin/bin:${PATH}"

# CUDA 11.8 only supports GCC <= 11; system default is GCC 12 which breaks CPUAdam JIT.
# Use GCC 9 which is installed at /usr/bin/gcc-9.
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/Train_MOE/coin_paths.sh"

NUM_SAMPLES="${1:-400}"    # samples in micro-benchmark dataset
MAX_STEPS="${2:-50}"       # training steps to measure

GPUS="${BENCH_GPUS:-4,5,6,7}"
MASTER_PORT=29611

MODEL_PATH="${COIN_BASE_MODEL}"
PRETRAIN_PROJ="${COIN_PRETRAIN_PROJECTOR}"
VISION_TOWER="${COIN_VISION_TOWER}"
DATA_PATH="${COIN_INSTR_ROOT}/ScienceQA/train.json"
IMAGE_ROOT="${COIN_IMAGE_ROOT}"

RESULT_DIR="${REPO_ROOT}/bench_results/train_lora"
LOG_DIR="${REPO_ROOT}/logs/LLaVA/bench_train"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# Create micro dataset subset
SUBSET_FILE="${RESULT_DIR}/sqa_train_subset_${NUM_SAMPLES}.json"
if [[ ! -f "${SUBSET_FILE}" ]]; then
    python3 -c "
import json
d = json.load(open('${DATA_PATH}'))
subset = d[:${NUM_SAMPLES}]
json.dump(subset, open('${SUBSET_FILE}', 'w'))
print(f'Micro dataset: {len(subset)} samples')
"
fi

run_train_bench() {
    local vectorized="${1:-1}"  # 1=vectorized/compiled, 0=loop
    local compiled="${2:-0}"    # 1=torch.compile on top of vectorized
    local tag
    if [[ "${vectorized}" == "0" ]]; then
        tag="loop"
    elif [[ "${compiled}" == "1" ]]; then
        tag="compiled"
    else
        tag="vectorized"
    fi

    local output_dir="${RESULT_DIR}/output_${tag}"
    local log_file="${LOG_DIR}/bench_train_${tag}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${output_dir}"

    echo "" | tee -a "${RESULT_DIR}/bench_train_results.txt"
    echo "=== Training bench: lora_mode=${tag} ===" | tee -a "${RESULT_DIR}/bench_train_results.txt"
    echo "Max steps: ${MAX_STEPS}, GPUs: ${GPUS}" | tee -a "${RESULT_DIR}/bench_train_results.txt"

    local t_start t_end elapsed
    t_start=$(date +%s%3N)

    COIN_GPUS="${GPUS}" \
    COIN_USE_VECTORIZED_LORA="${vectorized}" \
    COIN_USE_COMPILED_LORA="${compiled}" \
    /data4/home/sqx/.conda/envs/coin/bin/deepspeed \
        --include "localhost:${GPUS}" \
        --master_port "${MASTER_PORT}" \
        ETrain/Train/LLaVA/train_mem.py \
        --deepspeed ./scripts/zero3_offload.json \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --expert_num 8 \
        --model_name_or_path "${MODEL_PATH}" \
        --pretrain_mm_mlp_adapter "${PRETRAIN_PROJ}" \
        --version v1 \
        --data_path "${SUBSET_FILE}" \
        --image_folder "${IMAGE_ROOT}" \
        --vision_tower "${VISION_TOWER}" \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir "${output_dir}" \
        --max_steps "${MAX_STEPS}" \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.0 \
        --lr_scheduler_type "cosine" \
        --logging_steps 5 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 2 \
        --lazy_preprocess True \
        --report_to none \
        2>&1 | tee "${log_file}"

    t_end=$(date +%s%3N)
    elapsed=$(( t_end - t_start ))

    # Extract step times from log
    local step_times avg_step_time
    step_times=$(grep -oP "(?<=Step: )[0-9.]+" "${log_file}" 2>/dev/null || \
                 grep -oP "[0-9]+\.[0-9]+s/it" "${log_file}" 2>/dev/null | \
                 grep -oP "[0-9]+\.[0-9]+" | tail -20 || echo "")

    echo "[RESULT] lora_mode=${tag}: total_time=${elapsed}ms for ${MAX_STEPS} steps" | tee -a "${RESULT_DIR}/bench_train_results.txt"

    # Parse training runtime from trainer output (handles both ' and " quoting from HF Trainer)
    local train_runtime
    train_runtime=$(grep "train_runtime" "${log_file}" 2>/dev/null | grep -oP "(?<=['\"]train_runtime['\"]: )[0-9.]+" || echo "N/A")
    local train_sps
    train_sps=$(grep "train_samples_per_second" "${log_file}" 2>/dev/null | grep -oP "(?<=['\"]train_samples_per_second['\"]: )[0-9.]+" || echo "N/A")
    local train_stps
    train_stps=$(grep "train_steps_per_second" "${log_file}" 2>/dev/null | grep -oP "(?<=['\"]train_steps_per_second['\"]: )[0-9.]+" || echo "N/A")

    echo "  train_runtime=${train_runtime}s, samples/sec=${train_sps}, steps/sec=${train_stps}" | tee -a "${RESULT_DIR}/bench_train_results.txt"
    echo "BENCH_TRAIN mode=${tag} runtime_s=${train_runtime} samples_per_sec=${train_sps} steps_per_sec=${train_stps}" >> "${RESULT_DIR}/bench_train_results.txt"
}

cd "${REPO_ROOT}"

echo "========================================" | tee "${RESULT_DIR}/bench_train_results.txt"
echo "Training LoRA Vectorization Benchmark" | tee -a "${RESULT_DIR}/bench_train_results.txt"
echo "Date: $(date)" | tee -a "${RESULT_DIR}/bench_train_results.txt"
echo "GPUs: ${GPUS}, Steps: ${MAX_STEPS}, Subset: ${NUM_SAMPLES} samples" | tee -a "${RESULT_DIR}/bench_train_results.txt"
echo "========================================" | tee -a "${RESULT_DIR}/bench_train_results.txt"

# Baseline: original loop
run_train_bench 0 0

# Optimized: vectorized einsum
run_train_bench 1 0

# Optimized: vectorized + torch.compile (only if explicitly enabled via BENCH_COMPILE=1)
if [[ "${BENCH_COMPILE:-0}" == "1" ]]; then
    run_train_bench 1 1
fi

echo ""
echo "======== Final Summary ========"
grep "^BENCH_TRAIN" "${RESULT_DIR}/bench_train_results.txt"
echo "Full results: ${RESULT_DIR}/bench_train_results.txt"
