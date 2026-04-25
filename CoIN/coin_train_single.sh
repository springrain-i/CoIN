#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Usage:
#   bash CoIN/coin_train_single.sh [start_task] [end_task]
# Example:
#   bash CoIN/coin_train_single.sh 1 8

cd "${ROOT_DIR}"
bash scripts/LLaVA/Train_MOE/run_coin_single.sh "$@"
