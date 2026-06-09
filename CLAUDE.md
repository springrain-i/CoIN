# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Research Goals

This workspace has three sequential objectives, each building on the previous:

1. **Run MoKA on CoIN datasets** — implement MoKA's modality-specific adapter within the CoIN/LLaVA pipeline and verify it trains and evaluates correctly across all 8 tasks.
2. **Build MoE-MoKA** — extend MoKA with mixture-of-experts routing, combining per-expert modality-specific A matrices with MoE soft routing.
3. **Analyze over-textualization in continual learning** — test whether MoKA's fix for over-textualization degrades across the 8-task sequence, using partial modality inference (`all` / `text` / `visual` masking). The hypothesis is that continual learning re-introduces or worsens the text bias that MoKA suppresses in the single-task setting.

---

## Environment

```bash
conda activate coin
```

DeepSpeed CPUAdam requires these environment variables (set before running any training in a new shell):
```bash
export CUDA_HOME=$CONDA_PREFIX
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
```

Proxy (required for network access on this server):
```bash
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7891
```

All GPU work runs in tmux session `copilot`. Check `nvidia-smi` before starting GPU jobs.

---

## Entry Points

```bash
# Training
bash coin_train_single.sh 1 8        # All 8 tasks independently (single-task baseline)
bash coin_train_continual.sh 1 8     # Sequential continual learning (task 1→8)

# Evaluation
bash coin_eval_matrix.sh continual   # Full eval matrix after continual training
bash coin_eval_matrix.sh single      # Full eval matrix after single-task training

# Visualization
bash coin_plot_metrics.sh results/CoIN/LLaVA/metrics/continual_eval_matrix.csv
```

Orchestration scripts: `scripts/LLaVA/Train_MOE/run_coin_sequence.sh` (continual) and `run_coin_single.sh` (single-task).

Per-task training scripts: `scripts/LLaVA/Train_MOE/[1-8]_*.sh`, each called by the orchestrators.

---

## Fixed CoIN Task Order

All continual learning experiments use this fixed 8-task sequence:

| # | Task | Dataset |
|---|------|---------|
| 1 | ScienceQA | Multi-choice science VQA |
| 2 | TextVQA | Text reading in images |
| 3 | ImageNet | Image classification |
| 4 | GQA | Compositional VQA |
| 5 | VizWiz | VQA with real-world images |
| 6 | Grounding | Region grounding |
| 7 | VQAv2 | General VQA |
| 8 | OCRVQA | OCR-based VQA |

---

## Path Configuration

All paths are centralized in `scripts/LLaVA/Train_MOE/coin_paths.sh` and sourced by every training/eval script. Override via environment variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `COIN_BASE_MODEL` | `/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5` | Foundation model |
| `COIN_VISION_TOWER` | `/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/clip-vit-large-patch14-336` | CLIP encoder |
| `COIN_PRETRAIN_PROJECTOR` | `/data4/wxl/MoBLoRA-backup/CoIN/llava_projectors/.../mm_projector.bin` | LLaVA v1.5 projector |
| `COIN_INSTR_ROOT` | `/data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original` | Instruction JSON files |
| `COIN_IMAGE_ROOT` | `/data4/wxl/MoBLoRA-backup/CoIN/cl_dataset` | Image datasets |
| `COIN_OUTPUT_ROOT` | `<repo>/checkpoints/LLaVA/CoIN` | Training outputs |

GPU allocation is auto-detected from `CUDA_VISIBLE_DEVICES` and passed to DeepSpeed via `${COIN_DS_INCLUDE}`.

---

## Code Architecture

### Package Layout

- **`ETrain/`** — main training/evaluation framework
  - `Train/LLaVA/train.py` — training entry; defines `ModelArguments` (includes `expert_num`, `task_embedding_dim`, `lora_enable`), data/model loading, PEFT wrapping
  - `Train/LLaVA/train_mem.py` — thin wrapper that monkey-patches flash attention then calls `train.py`
  - `Models/LLaVA/llava_arch.py` — multimodal model mixin; `prepare_inputs_labels_for_multimodal()` builds interleaved text+vision embeddings and assigns the `current_lora_mask` tensor
  - `Models/LLaVA/language_model/llava_llama.py` — `_apply_lora_token_mask()` propagates `current_lora_mask` and `lora_mode` to every LoRA module before each forward pass
  - `Eval/LLaVA/CoIN/model_vqa.py` — evaluation loop; sets `model.lora_mode` from CLI `--lora-mode {all,text,vision}`

- **`CoIN/peft/tuners/`** — custom PEFT (forked from `peft==0.4.0`)
  - `lora.py` — standard LoRA extended with `token_mask` and `lora_mode`; `_get_effective_token_mask()` converts the mask tensor to a boolean mask per mode
  - `coinmoelora.py` — MoE-LoRA: `CoINMOELoraConfig` (`expert_num`, `task_embedding_dim`), `CoINMOELoraLinear` with soft-routing (softmax-weighted sum over experts); fixed expert count, not task-expandable

- **`scripts/LLaVA/`** — bash orchestration only (no Python logic here)

### Partial Modality Masking (lora_mode)

Token mask encoding — assigned in `llava_arch.py` during multimodal input preparation:
- `2` = text token
- `1` = vision token  
- `0` = padding (always excluded)

The mask is stored as `self.current_lora_mask` on the model and pushed to each LoRA module via `_apply_lora_token_mask()`. At each LoRA linear forward, `_get_effective_token_mask()` converts it to a boolean based on `lora_mode`:
- `"all"` → `mask > 0` (text + vision)
- `"text"` → `mask == 2`
- `"vision"` → `mask == 1`

### MoE-LoRA

Uses soft routing (softmax over expert weights), not hard per-task routing. Expert count is set by `--expert_num` at training time and stays fixed throughout continual learning. Continual training loads the previous task's LoRA checkpoint via `--previous_task_model_path` and continues fine-tuning the same adapter.

---

## MoKA: Multimodal low-rank Adaptation (NeurIPS 2025)

Paper: `/data4/home/sqx/MokA/Wei 等 - MokA Multimodal Low-Rank Adaptation for MLLMs.pdf`

### The problem MoKA solves

Standard LoRA uses a single shared A matrix for all modality tokens. Because text tokens dominate training (more tokens, stronger gradient signal), the shared A gets over-optimized for text. Partial modality inference — running only vision or only text tokens through the LoRA path at inference — shows a large gap: text-only is nearly as good as full modality, but vision-only degrades sharply. This is the "over-textualization" phenomenon.

### MoKA architecture

MoKA replaces the single LoRA `A` with three components, keeping `B` shared:

```
h = W_0 x + B·Ax
```

**1. Unimodal matrix A (modality-specific):**
Each modality gets its own A matrix `{A^text, A^visual}`. Tokens are routed to their own A independently:
```
Ax = [A^text · x^text ;  A^visual · x^visual]
```
This prevents text gradient from contaminating the visual compression subspace.

**2. Task-centric cross-attention (cross-modal interaction):**
After unimodal compression, non-text (visual) tokens attend to text tokens as keys/values. This injects task-description context into visual representations. Text tokens themselves are left unchanged.
```
Att(A^v x^v, A^t x^t, A^t x^t) = softmax( (A^v x^v)(A^t x^t)^T / sqrt(r_per) ) · A^t x^t
# Scaling note: MoKA paper formula writes sqrt(N_t) (text token count); the original MoKA
# repo code uses sqrt(r) (full lora rank, d_k of the projection); our MoEMoKA uses sqrt(r_per)
# (= r/N, per-expert rank d_k). All are valid scaled dot-product attention; sqrt(N_t) varies
# with sequence length causing unstable softmax sharpness, so we follow the code convention.
Enhanced visual: A^v x^v + Att(...)
```
No extra linear projections W_q/W_k/W_v — the A matrices already serve as projections.

**3. Shared multimodal B (single-LoRA MoKA only):**
In the original single-LoRA MoKA, one B projects all enhanced representations into the output space, enforcing cross-modal alignment. This "shared B" description applies to the base MoKA paper which has exactly one LoRA adapter per linear layer.

Final forward (MoKA, single LoRA):
```
h = W_0 x + [B·A^t x^t  ;  B·(A^v x^v + Att_{v,t,t})  ]
             ↑ unimodal   ↑ unimodal + cross-modal
```

**Initialization:** A matrices use Kaiming uniform; B is initialized to zero (so ΔW=0 at start, same as LoRA).

**Parameter cost:** ~1.33% of total model (vs 1.27% for standard LoRA) — negligible overhead.

### Why MoKA is relevant to the continual learning hypothesis

MoKA fixes over-textualization in the **single-task** setting. The open question is: does continual learning across 8 tasks re-introduce or worsen text bias even with MoKA? The modality-specific A matrices could still drift toward text-dominant directions as more text-heavy tasks are seen sequentially. This is the core research hypothesis.

### Implementing MoKA in CoIN

The existing token mask infrastructure (`text=2, vision=1, pad=0` in `llava_arch.py`, propagated via `_apply_lora_token_mask()`) provides exactly the routing signal MoKA needs. Implementation requires:

1. **New PEFT layer** `CoIN/peft/tuners/mokalora.py`:
   - Replace single `lora_A` dict with `lora_A_text` and `lora_A_visual` dicts
   - Add a cross-attention module operating on low-rank representations
   - Shared `lora_B` stays unchanged
   - Forward: split input by `token_mask`, apply respective A, run cross-attention on visual tokens using text as K/V, concatenate, apply B

2. **New config** `MoKALoraConfig` (subclass of `LoraConfig`): no new hyperparameters needed beyond standard rank/alpha.

3. **Training integration** (`ETrain/Train/LLaVA/train.py`): add `--moka_enable` flag alongside existing `--lora_enable`.

4. **Eval integration** (`ETrain/Eval/LLaVA/CoIN/model_vqa.py`): MoKA is transparent to eval since `lora_mode` masking is handled in the layer forward — no changes needed.

### MoE-MoKA design

Extend MoKA with N experts. Each expert holds its own `(A^text_i, A^visual_i, B_i)` triple — **per-expert B is intentional**, following the same convention as the original CoIN MoE-LoRA baseline where each expert is fully independent. This differs from the single-LoRA MoKA paper (which has one shared B) because in a mixture-of-experts setting there is no single "shared" projection. Routing is soft (softmax-weighted sum). Cross-attention can be per-expert or shared; current implementation uses per-expert for consistency.

```
ΔW·x = Σ_i  w_i · B_i · [A^text_i · x^text ;  A^visual_i · x^visual + Att_i(...)]
```

where `w_i` is the soft routing weight for expert i, computed from the original (pre-dropout) input `x` via a learned linear router, then softmax. Expert A and B matrices receive dropout-regularized input (same as CoIN baseline).

---

## Results Layout

```
results/CoIN/LLaVA/metrics/
├── continual_online_eval.csv    # Per-task accuracy after each training step (continual)
├── continual_final_eval.csv     # Full matrix eval after all 8 tasks (continual)
└── single_eval_matrix.csv       # Per-task accuracy for single-task baseline
```

---

## Git Conventions

One commit per experiment phase. Commit message style:
```
baseline: snapshot before new experiment
exp-config: <what changed in config>
exp-run: <experiment description>
analysis: <metrics/plotting changes>
report: <findings summary>
```

Never merge multiple experiment phases into one commit.

---

## Installation

```bash
conda create -n coin python=3.10 -y
conda activate coin
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
