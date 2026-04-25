#!/bin/bash
set -euo pipefail

# Usage:
#   bash coin_plot_metrics.sh <csv_path> [out_dir]
# Example:
#   bash coin_plot_metrics.sh results/CoIN/LLaVA/metrics/continual_eval_matrix.csv

if [[ $# -lt 1 ]]; then
  echo "Usage: bash coin_plot_metrics.sh <csv_path> [out_dir]"
  exit 1
fi

CSV_PATH="$1"
OUT_DIR="${2:-results/CoIN/LLaVA/plots}"

python scripts/analysis/plot_coin_metrics.py --csv "${CSV_PATH}" --out-dir "${OUT_DIR}"
