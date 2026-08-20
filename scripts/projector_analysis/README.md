# Forward mm_projector swap

This experiment keeps the final forward-order T8 OCRVQA MoE-LoRA adapter,
router, model config, Vicuna base model, and CLIP vision tower fixed. It replaces
only the final checkpoint's `mm_projector` tensors with the projector saved
immediately after a selected early continual-learning task. It evaluates each
hybrid only on the matching task: T1 projector on T1, T2 projector on T2, and
so on through T7 projector on T7.

## Fixed protocol

- Task order: ScienceQA, TextVQA, ImageNet, GQA, VizWiz, Grounding, VQAv2, OCRVQA.
- Final checkpoint: `backup/legacy_llava_20260818/checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora`.
- Early checkpoint: T1 through T7 under `backup/legacy_llava_20260818/checkpoints/LLaVA/CoIN`.
- LoRA mode: `all` only.
- Eval batch size: `COIN_EVAL_BATCH_SIZE=4`.
- SDPA batch-inference fix: `COIN_USE_SDPA_PATCH=1`.
- Each pair uses the matching existing script under `scripts/LLaVA/Eval`
  without changing its prompts, decoding defaults, dataset, or scorer.

The training batch-size table in the repository instructions does not apply to
this eval-only experiment. No training command is invoked.

## Build a derived checkpoint

For example, use the projector from T4 GQA:

```bash
/data4/home/sqx/.conda/envs/coin/bin/python \
  scripts/projector_analysis/prepare_forward_projector_swap.py \
  --early-task-id 4
```

The derived checkpoint is written below:

```text
checkpoints/LLaVA/CoIN_projector_swap/
  early_T4_GQA__final_T8_OCRVQA/
    OCRVQA_llava_MOE_lora/
```

The large final adapter is an absolute read-only symlink by default. Pass
`--materialize-adapter` to copy it instead. The projector is always written as
an independent `non_lora_trainables.bin`. `swap_manifest.json` records source
paths, hashes, tensor metadata, Git state, and the exact multimodal config.

## Dry run

Validate the complete T1-through-T7 sweep without starting GPU inference:

```bash
DRY_RUN=1 bash scripts/projector_analysis/run_all_forward_projector_swaps.sh
```

Validate only one projector arm:

```bash
DRY_RUN=1 bash scripts/projector_analysis/run_forward_projector_swap_8tasks.sh 4
```

Dry run builds or validates the derived checkpoint and prints its one matching
formal eval command, but does not start GPU inference.

## Full T1-T7 projector sweep

The formal experiment runs seven diagonal pairs: T1 projector on T1 ScienceQA,
T2 projector on T2 TextVQA, through T7 projector on T7 VQAv2. Each eval uses all
eight GPUs (eight chunks), and the seven pairs run sequentially. T8 is the
unchanged final projector, so it is not re-evaluated.

Activate the `coin` environment, confirm all eight GPUs are idle, and run the
full sweep in the `coin` tmux session:

```bash
conda activate coin
bash scripts/projector_analysis/run_all_forward_projector_swaps.sh
```

For an isolated arm, pass its early task ID to the single-arm runner:

```bash
bash scripts/projector_analysis/run_forward_projector_swap_8tasks.sh 4
```

Resume one diagonal pair with the original `RUN_TIMESTAMP`:

```bash
RUN_TIMESTAMP=YYYYMMDD_HHMMSS \
  bash scripts/projector_analysis/run_forward_projector_swap_8tasks.sh 4
```

Complete existing stages are validated and skipped. An existing incomplete
stage aborts the run and is never deleted or overwritten automatically.

## Outputs

- Derived checkpoint: `checkpoints/LLaVA/CoIN_projector_swap/`.
- Logs: `logs/LLaVA/projector_swap/<arm>/<timestamp>/`.
- Metrics and manifests:
  `results/CoIN/LLaVA/metrics/projector_swap/<arm>/<timestamp>/`.
- Final 7-row diagonal summary:
  `results/CoIN/LLaVA/metrics/projector_swap/all_early/<timestamp>/`.
- Predictions and task scorer outputs remain in the standard result roots used
  by the eight existing eval scripts. Every stage includes a globally unique run
ID, and `result_paths.json` indexes those directories.

For isolated validation, the runner also accepts `PROJECTOR_SWAP_CHECKPOINT_ROOT`,
`PROJECTOR_SWAP_LOG_ROOT`, and `PROJECTOR_SWAP_METRICS_ROOT`. Their defaults are
the standard checkpoint, log, and result locations listed above.
