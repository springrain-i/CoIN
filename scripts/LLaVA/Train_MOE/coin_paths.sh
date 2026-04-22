#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COIN_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Shared asset root from coin_assets_note.md
COIN_ASSET_ROOT="${COIN_ASSET_ROOT:-/data0/sqx/coin_assets}"

# Core paths (override via env when needed)
COIN_DS_CONFIG="${COIN_DS_CONFIG:-${COIN_REPO_ROOT}/scripts/zero3_offload.json}"
COIN_BASE_MODEL="${COIN_BASE_MODEL:-${COIN_ASSET_ROOT}/Vicuna/vicuna-7b-v1.5}"
COIN_VISION_TOWER="${COIN_VISION_TOWER:-${COIN_ASSET_ROOT}/clip-vit-large-patch14-336}"
COIN_PRETRAIN_PROJECTOR="${COIN_PRETRAIN_PROJECTOR:-${COIN_ASSET_ROOT}/llava_projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin}"
COIN_INSTR_ROOT="${COIN_INSTR_ROOT:-${COIN_ASSET_ROOT}/playground/Instructions_Original}"
COIN_IMAGE_ROOT="${COIN_IMAGE_ROOT:-${COIN_ASSET_ROOT}}"
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
