#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COIN_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

resolve_first_existing_path() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  # Keep behavior predictable even when nothing exists yet.
  echo "$1"
}

DEFAULT_BASE_MODEL="$(resolve_first_existing_path \
  /data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5)"

DEFAULT_VISION_TOWER="$(resolve_first_existing_path \
  /data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/clip-vit-large-patch14-336)"

DEFAULT_PRETRAIN_PROJECTOR="$(resolve_first_existing_path \
  /data4/wxl/MoBLoRA-backup/CoIN/llava_projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin)"

# Core paths (override via env when needed)
COIN_DS_CONFIG="${COIN_DS_CONFIG:-${COIN_REPO_ROOT}/scripts/zero3_offload.json}"
COIN_BASE_MODEL="${COIN_BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
COIN_VISION_TOWER="${COIN_VISION_TOWER:-${DEFAULT_VISION_TOWER}}"
COIN_PRETRAIN_PROJECTOR="${COIN_PRETRAIN_PROJECTOR:-${DEFAULT_PRETRAIN_PROJECTOR}}"
# On 3090 server, instruction jsons and image data are stored separately.
COIN_INSTR_ROOT="${COIN_INSTR_ROOT:-/data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original}"
COIN_IMAGE_ROOT="${COIN_IMAGE_ROOT:-/data4/wxl/MoBLoRA-backup/CoIN/cl_dataset}"
COIN_OUTPUT_ROOT="${COIN_OUTPUT_ROOT:-${COIN_REPO_ROOT}/checkpoints/LLaVA/CoIN}"

build_ds_include() {
  # If user provides explicit include, respect it.
  if [[ -n "${COIN_DS_INCLUDE:-}" ]]; then
    echo "${COIN_DS_INCLUDE}"
    return
  fi

  local gpu_count=0
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local cvd_clean
    cvd_clean=$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')
    IFS=',' read -r -a _cvd_arr <<< "${cvd_clean}"
    gpu_count=${#_cvd_arr[@]}
  elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  fi

  if [[ "${gpu_count}" -le 0 ]]; then
    gpu_count=1
  fi

  local slots=""
  local i
  for ((i=0; i<gpu_count; i++)); do
    if [[ -z "${slots}" ]]; then
      slots="${i}"
    else
      slots="${slots},${i}"
    fi
  done
  echo "localhost:${slots}"
}

COIN_DS_INCLUDE="$(build_ds_include)"

mkdir -p "${COIN_OUTPUT_ROOT}"

require_path() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "$path" ]]; then
    echo "[CoIN][PathError] Missing ${desc}: ${path}" >&2
    exit 1
  fi
}
