# CoIN Fast Entrypoints (Repo Root)

Run these commands directly from repo root `/data4/home/sqx/CoIN`.

## Train (single-task independent)

bash coin_train_single.sh 1 8

## Train (continual sequence)

bash coin_train_continual.sh 1 8

## Eval matrix

bash coin_eval_matrix.sh continual
bash coin_eval_matrix.sh single

## Plot metrics

bash coin_plot_metrics.sh results/CoIN/LLaVA/metrics/continual_eval_matrix.csv
