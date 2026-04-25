#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Usage:
#   bash CoIN/coin_eval_matrix.sh [continual|single]
# Example:
#   bash CoIN/coin_eval_matrix.sh continual

cd "${ROOT_DIR}"
bash scripts/LLaVA/Eval/run_coin_eval_matrix.sh "$@"
