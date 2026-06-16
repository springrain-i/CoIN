# Grad-Attn Logging Policy

This document records the formal logging policy for `grad_attn` mechanism-analysis runs.

## Goal

The mechanism analysis measures whether text and visual tokens contribute unevenly to LoRA parameter updates during MCIT. The logger should preserve interpretable optimizer-step statistics while keeping single-A100 runs tractable.

## Default Training Policy

The formal single-GPU entrypoint is:

```bash
bash scripts/LLaVA/Train_MOE/run_grad_attn_analysis.sh 1 8
```

The script defaults to:

```text
per_device_train_batch_size = 4
COIN_GRAD_STEP_SCHEDULE = staged
COIN_GRAD_LAYER_BLOCKS = 0,4,8,12,16,20,24,28,31
COIN_GRAD_MICROBATCH_SAMPLE = 0.25
```

The script computes task-specific gradient accumulation so the single-GPU effective batch size matches the original MoELoRA 8-GPU setting.

| Task | Dataset | Effective batch | Single-GPU bs | Accumulation |
|---:|---|---:|---:|---:|
| T1 | ScienceQA | 512 | 4 | 128 |
| T2 | TextVQA | 1024 | 4 | 256 |
| T3 | ImageNet | 512 | 4 | 128 |
| T4 | GQA | 384 | 4 | 96 |
| T5 | VizWiz | 256 | 4 | 64 |
| T6 | Grounding | 256 | 4 | 64 |
| T7 | VQAv2 | 256 | 4 | 64 |
| T8 | OCRVQA | 256 | 4 | 64 |

## Step Sampling

Use task-length-aware optimizer-step sampling. Short tasks are logged densely; long tasks are logged sparsely.

| Total optimizer steps | Logged step ratio |
|---:|---:|
| `steps <= 50` | 100% |
| `50 < steps <= 100` | 30% |
| `100 < steps <= 200` | 20% |
| `200 < steps <= 600` | 10% |
| `steps > 600` | 8% |

Within each task, selected steps are staged:

- first 20% of training progress: 50% of logging points,
- 20%-60% progress: 30% of logging points,
- 60%-100% progress: 20% of logging points.

This gives denser coverage early in training, where gradients are expected to change faster.

## Layer Sampling

By default, log LoRA modules and attention layers in these transformer blocks:

```text
0, 4, 8, 12, 16, 20, 24, 28, 31
```

This samples every four blocks and always includes the final block. Inside selected blocks, all LoRA injection points are logged.

## Micro-Batch Sampling

Within a selected optimizer step, log 25% of gradient-accumulation micro-batches by default:

```text
COIN_GRAD_MICROBATCH_SAMPLE=0.25
```

The selected micro-batches are spread uniformly over the accumulation window using rounded equally spaced points. This preserves coverage across the beginning, middle, and end of the effective batch while avoiding the cost of logging every micro-batch.

Rationale: full micro-batch logging inside selected optimizer steps is too slow on a single A100 because hooks fire once per accumulation micro-batch. Uniform 25% sampling is the current formal compromise between representativeness and runtime.

## Gradient Metrics

`ModalGradientLogger` records per selected step, micro-batch, and layer:

```text
G_text_A, G_vis_A, R_A, R_A_tok
G_text_B, G_vis_B, R_B, R_B_tok
G_text_dW, G_vis_dW, R_dW, R_dW_tok
n_text, n_vis
```

Current modality buckets are:

```text
raw_mask == 2 -> text token
raw_mask == 1 -> visual token
```

Important caveat: answer tokens are currently included in the text-token bucket. If answer-token-specific analysis is needed, future logger versions should split text into `answer_text` and `non_answer_text` using `labels != IGNORE_INDEX`.

## Attention Metrics

`AttentionLogger` uses the same step, layer, and micro-batch sampling policy as gradient logging.

It records post-image text-query attention and information-flow metrics:

```text
A_tt, A_tv, R_att_text
U_vis, U_text, R_info
n_post_text
```

The logger streams CSV rows as records are produced rather than waiting until the end of training.

## Output Locations

```text
analysis/gradient_dominance/
analysis/attn_dominance/
logs/LLaVA/grad_attn_analysis/
```

These output directories are experiment artifacts and should not be committed.
