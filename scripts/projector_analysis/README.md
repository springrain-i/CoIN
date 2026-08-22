# Standard-LoRA `mm_projector` restoration

This directory prepares and evaluates **the completed forward CoIN standard-LoRA
run** `coin_lora_zero2_gbs128_seed42_20260820_2110`. It does not reuse the
historical MoE-LoRA checkpoints or write into their output directories.

For an early task `Tn`, the experiment starts from the final T8 OCRVQA standard
LoRA checkpoint and replaces only its four `mm_projector` tensors with those
saved immediately after `Tn`. It then evaluates that hybrid on the matching
task `Tn`.

## Fixed protocol

- Task order: ScienceQA, TextVQA, ImageNet, GQA, VizWiz, Grounding, VQAv2,
  OCRVQA.
- Source sequence:
  `checkpoints/LLaVA/CoIN_coin_lora_zero2_gbs128_seed42_20260820_2110`.
- Early checkpoints: `T1` through `T7`, named `{Task}_llava_lora`.
- Final checkpoint: `OCRVQA_llava_lora` from T8.
- Adaptation: standard PEFT LoRA, rank 128, alpha 256, targeting `q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Fixed components: Vicuna base model, CLIP vision tower, final T8 LoRA adapter,
  and model/adapter configuration.
- Replaced component: `mm_projector` only, from the matching early checkpoint.
- Evaluation: `--lora-mode all`, eight GPU workers, batch size 4 per worker,
  and SDPA enabled.

T8 is the final task and has no post-training final comparison, so it is not a
projector-restoration arm.

## Derived checkpoints

Build T4's GQA-projector hybrid without GPU inference:

```bash
/data4/home/sqx/.conda/envs/coin/bin/python \
  scripts/projector_analysis/prepare_forward_projector_swap.py \
  --early-task-id 4
```

Derived checkpoints default to:

```text
checkpoints/LLaVA/CoIN_lora_projector_swap/
  early_T4_GQA__final_T8_OCRVQA/
    OCRVQA_llava_lora/
```

The final `adapter_model.bin` is an absolute read-only symlink by default. The
hybrid `non_lora_trainables.bin` is verified tensor-by-tensor: all projector
tensors equal their early source and every non-projector tensor equals the final
source. `swap_manifest.json` records source paths, hashes, projector metadata,
standard-LoRA configuration, model configuration, and Git state.

The builder refuses a checkpoint whose adapter is not standard LoRA or whose
rank, alpha, or target-module set differ from the formal run. To use a different
completed standard-LoRA chain, pass `--checkpoint-root PATH` explicitly.

## Dry run

The runner's dry run builds/validates its derived checkpoint and prints the one
formal evaluation command, but does not start GPU inference:

```bash
DRY_RUN=1 bash scripts/projector_analysis/run_all_forward_projector_swaps.sh
```

For one arm:

```bash
DRY_RUN=1 bash scripts/projector_analysis/run_forward_projector_swap_8tasks.sh 4
```

## Full diagonal sweep

After confirming that the eight GPUs are idle:

```bash
conda activate coin
export CUDA_HOME=$CONDA_PREFIX
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
nvidia-smi
bash scripts/projector_analysis/run_all_forward_projector_swaps.sh
```

The seven arms run sequentially, each occupying all eight GPUs. To resume a
specific arm, retain its timestamp:

```bash
RUN_TIMESTAMP=YYYYMMDD_HHMMSS \
  bash scripts/projector_analysis/run_forward_projector_swap_8tasks.sh 4
```

Existing complete stages are validated and reused. An incomplete stage aborts;
the runner never deletes or overwrites it automatically.

## Outputs

- Derived checkpoints: `checkpoints/LLaVA/CoIN_lora_projector_swap/`.
- Logs: `logs/LLaVA/lora_projector_swap/`.
- Metrics/manifests: `results/CoIN/LLaVA/metrics/lora_projector_swap/`.
- Predictions and task scorer outputs: `results/CoIN/LLaVA/lora_projector_swap/`.
- Sweep summary: `results/CoIN/LLaVA/metrics/lora_projector_swap/all_early/<timestamp>/`.

All results produced before sample-cohort annotation are descriptive full-test
outputs. They must not be presented as an `essential_forgetting` causal result
until the frozen two-stage labeling/cohort procedure has been applied.

## Historical MoE-LoRA artifacts

`EXPERIMENT_REPORT_20260818.md` documents a prior MoE-LoRA run and is retained
solely as historical implementation context. Its checkpoints, outputs, and
numbers do not belong to the standard-LoRA experiment above.
