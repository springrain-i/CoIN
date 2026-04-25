# CoIN Fast Entrypoints

These wrappers are placed under `CoIN/` for quick execution.

## Train (single-task independent)

bash CoIN/coin_train_single.sh 1 8

## Train (continual sequence)

bash CoIN/coin_train_continual.sh 1 8

## Eval matrix

bash CoIN/coin_eval_matrix.sh continual
bash CoIN/coin_eval_matrix.sh single

## Plot metrics

bash CoIN/coin_plot_metrics.sh results/CoIN/LLaVA/metrics/continual_eval_matrix.csv
