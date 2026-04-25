#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Usage:
#   bash CoIN/coin_plot_metrics.sh <csv_path> [out_dir]
# Example:
#   bash CoIN/coin_plot_metrics.sh results/CoIN/LLaVA/metrics/continual_eval_matrix.csv

if [[ $# -lt 1 ]]; then
  echo "Usage: bash CoIN/coin_plot_metrics.sh <csv_path> [out_dir]"
  exit 1
fi

CSV_PATH="$1"
OUT_DIR="${2:-results/CoIN/LLaVA/plots}"

cd "${ROOT_DIR}"
python scripts/analysis/plot_coin_metrics.py --csv "${CSV_PATH}" --out-dir "${OUT_DIR}"
