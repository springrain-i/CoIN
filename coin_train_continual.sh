#!/bin/bash
set -euo pipefail

# Usage:
#   bash coin_train_continual.sh [start_task] [end_task]
# Example:
#   bash coin_train_continual.sh 1 8

bash scripts/LLaVA/Train_MOE/run_coin_sequence.sh "$@"
