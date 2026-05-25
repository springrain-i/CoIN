#!/bin/bash
# Shared setup for all MoEMoKA eval scripts.
# Source this file; do NOT execute it directly.

EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EVAL_SCRIPT_DIR}/../coin_paths.sh"

# --- Args: $1=STAGE  $2=MODELPATH  $3=LORA_MODE ---
STAGE="${1:-Finetune}"
MODELPATH="${2:-}"
LORA_MODE="${3:-all}"

if [[ "$LORA_MODE" == "visual" ]]; then
    LORA_MODE="vision"
fi

if [[ -z "$MODELPATH" ]]; then
    echo "[eval] ERROR: MODELPATH not provided (arg \$2)" >&2
    exit 1
fi

# Use CUDA_VISIBLE_DEVICES if set, else auto-detect via nvidia-smi.
gpu_list="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$gpu_list" ]]; then
    gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 1)
    gpu_list=$(seq -s ',' 0 $((gpu_count-1)))
fi
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

BASE_MODEL_PATH="${COIN_BASE_MODEL}"
VISION_TOWER_PATH="${COIN_VISION_TOWER}"
IMAGE_ROOT="${COIN_IMAGE_ROOT}"
INSTR_ROOT="${COIN_INSTR_ROOT}"
RESULT_ROOT="${COIN_REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA"
PYTHON="${COIN_PYTHON}"
