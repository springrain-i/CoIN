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
  /hy-tmp/Vicuna/vicuna-7b-v1.5 \
  /data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5)"

DEFAULT_VISION_TOWER="$(resolve_first_existing_path \
  /hy-tmp/clip-vit-large-patch14-336 \
  /data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/clip-vit-large-patch14-336)"

DEFAULT_PRETRAIN_PROJECTOR="$(resolve_first_existing_path \
  /hy-tmp/llava_projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
  /data4/wxl/MoBLoRA-backup/CoIN/llava_projectors/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin)"

DEFAULT_INSTR_ROOT="$(resolve_first_existing_path \
  /hy-tmp/playground/Instructions_Original \
  /data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original)"

DEFAULT_IMAGE_ROOT="$(resolve_first_existing_path \
  /hy-tmp \
  /data4/wxl/MoBLoRA-backup/CoIN/cl_dataset)"

# Core paths (override via env when needed)
# Auto-select DeepSpeed config: offload for single-GPU, zero3 for multi-GPU.
# Override with COIN_DS_CONFIG env var when needed.
_auto_gpu_count=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES}"
  _auto_gpu_count=${#_cvd[@]}
else
  _auto_gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 1)
fi
if [[ "${_auto_gpu_count}" -ge 4 ]]; then
  _default_ds_config="${COIN_REPO_ROOT}/scripts/zero3.json"
else
  _default_ds_config="${COIN_REPO_ROOT}/scripts/zero3_offload.json"
fi
COIN_DS_CONFIG="${COIN_DS_CONFIG:-${_default_ds_config}}"
unset _auto_gpu_count _default_ds_config _cvd
COIN_BASE_MODEL="${COIN_BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
COIN_VISION_TOWER="${COIN_VISION_TOWER:-${DEFAULT_VISION_TOWER}}"
COIN_PRETRAIN_PROJECTOR="${COIN_PRETRAIN_PROJECTOR:-${DEFAULT_PRETRAIN_PROJECTOR}}"
COIN_INSTR_ROOT="${COIN_INSTR_ROOT:-${DEFAULT_INSTR_ROOT}}"
COIN_IMAGE_ROOT="${COIN_IMAGE_ROOT:-${DEFAULT_IMAGE_ROOT}}"
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

# Use explicit python to run deepspeed binary — avoids broken shebang when conda env was relocated.
COIN_PYTHON="${COIN_PYTHON:-/hy-tmp/miniconda3/envs/coin/bin/python}"
COIN_DS_BIN="${COIN_DS_BIN:-/hy-tmp/miniconda3/envs/coin/bin/deepspeed}"
COIN_DEEPSPEED="${COIN_PYTHON} ${COIN_DS_BIN}"

# DeepSpeed compilation requirements.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export CC="${CC:-/usr/bin/gcc-9}"
export CXX="${CXX:-/usr/bin/g++-9}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-9}"
export PATH="/hy-tmp/miniconda3/envs/coin/bin:${PATH}"
# Redirect torch JIT extension cache to writable /hy-tmp (avoids overlay-fs issues).
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/hy-tmp/torch_extensions}"

mkdir -p "${COIN_OUTPUT_ROOT}"

require_path() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "$path" ]]; then
    echo "[CoIN][PathError] Missing ${desc}: ${path}" >&2
    exit 1
  fi
}
