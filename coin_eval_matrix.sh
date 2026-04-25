#!/bin/bash
set -euo pipefail

# Usage:
#   bash coin_eval_matrix.sh [continual|single]
# Example:
#   bash coin_eval_matrix.sh continual

bash scripts/LLaVA/Eval/run_coin_eval_matrix.sh "$@"
