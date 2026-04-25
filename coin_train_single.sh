#!/bin/bash
set -euo pipefail

# Usage:
#   bash coin_train_single.sh [start_task] [end_task]
# Example:
#   bash coin_train_single.sh 1 8
# Resume helpers:
#   RUN_TRAIN=0 / RUN_EVAL=0 / SKIP_TRAIN_TASKS="1" / SKIP_EVAL_TASKS="1"

bash scripts/LLaVA/Train_MOE/run_coin_single.sh "$@"
