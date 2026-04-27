# MoE-MoKA Implementation Guide

**Purpose:** Human-review reference for the MoE-MoKA (Mixture-of-Experts Multimodal LoRA Adaptation) implementation in the CoIN continual learning framework.

---

## 1. What Is MoE-MoKA

MoE-MoKA extends MoKA (NeurIPS 2025) with a mixture-of-experts routing layer. Each expert holds modality-specific A matrices `(A_text_i, A_vis_i)` for i=1..N. A soft router (softmax) blends expert outputs before multiplying by a shared B matrix.

**Core forward for each LoRA linear layer:**

```
ΔW·x = B · Σ_i  w_i · [ A_text_i · x_text  ;  A_vis_i · x_vis + CrossAttn(A_vis_i·x_vis, A_text_i·x_text) ]
```

where:
- `x_text`, `x_vis` = tokens sliced by `token_mask` (2=text, 1=visual, 0=pad)
- `w_i` = router softmax weight for expert i (router input = mean of all token A-outputs)
- Cross-attention: visual tokens (Q) attend to text tokens (K, V) in rank-r space, scaled by `lora_attn` (learnable scalar)
- B is shared across all experts

---

## 2. File Map

| File | Role |
|------|------|
| `CoIN/peft/tuners/mokamoelora.py` | **Core MoE-MoKA layer** — config, linear module, forward |
| `CoIN/peft/mapping.py` | Registers `MoEMoKALoraConfig` and `PeftModelForCausalLMLORAMOE` |
| `CoIN/peft/utils/save_and_load.py` | PEFT save/load — **bypassed** for MoE-MoKA (see §5) |
| `ETrain/Train/LLaVA/train.py` | Training entry; `ModelArguments.moe_moka_enable` flag |
| `ETrain/Train/LLaVA/llava_trainer.py` | `save_trained_model` + `load_model_from_previous_task` |
| `ETrain/Models/LLaVA/builder.py` | Eval-time model loading (`load_pretrained_model`) |
| `ETrain/Models/LLaVA/llava_arch.py` | Sets `current_lora_mask` (token_mask) on model |
| `ETrain/Models/LLaVA/language_model/llava_llama.py` | `_apply_lora_token_mask()` propagates mask to each layer |

---

## 3. Core Module: `mokamoelora.py`

**Path:** `CoIN/peft/tuners/mokamoelora.py`

### 3.1 Config — `MoEMoKALoraConfig`

```python
@dataclass
class MoEMoKALoraConfig(LoraConfig):
    peft_type: PeftType = field(default=PeftType.MOE_MOKA_CoIN)
    expert_num: int = field(default=4)
```

Inherits all standard LoRA hyperparameters (`r`, `lora_alpha`, `target_modules`, etc.).

### 3.2 Layer — `MoEMoKALoraLinear`

**Key attributes:**

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `lora_A_text[adapter]` | `nn.ModuleList` of N `nn.Linear(in, r, bias=False)` | Per-expert text A matrices |
| `lora_A_vis[adapter]` | `nn.ModuleList` of N `nn.Linear(in, r, bias=False)` | Per-expert visual A matrices |
| `lora_B[adapter]` | `nn.Linear(r, out, bias=False)` | Shared B matrix |
| `lora_router[adapter]` | `nn.Linear(r, N, bias=False)` | Soft router |
| `lora_attn[adapter]` | `nn.ParameterDict` → scalar | Cross-attention gate |

**Init:** A matrices use Kaiming uniform, B is zero-initialized (so ΔW=0 at start).

### 3.3 Forward Pass

**File:** `mokamoelora.py` → `MoEMoKALoraLinear.forward()`

Step-by-step:

1. **Token mask slicing** (lines ~180-210): If `token_mask.shape[1] != S` (generation with KV-cache), trim mask to `token_mask[:, -S:]`. Fall back to all-text mask if still mismatched.
2. **Unconditional A calls** (lines ~220-250): Each expert's A_text and A_vis are always called (even if `has_text=False` or `has_vis=False`), then conditionally written to `a_out`. This is required for ZeRO-3 trace consistency.
3. **Cross-attention** (lines ~255-275): Visual A-outputs (Q) attend to text A-outputs (K, V) via `_cross_attention()`. Output scaled by `lora_attn` scalar.
4. **Router** (lines ~280-300): Mean-pool `a_out`, pass through `lora_router`, softmax → weights `w`.
5. **Expert blend** (lines ~300-320): `Σ_i w_i * per_expert_a_out_i`, then `lora_B(blended)`.
6. **Residual** (lines ~320-330): Add LoRA delta to base linear output, scaled by `lora_alpha / r`.

### 3.4 Cross-Attention — `_cross_attention()`

```python
def _cross_attention(self, q, k):
    # q: (n_vis, r), k: (n_text, r)
    scale = math.sqrt(k.shape[-1])
    scores = (q @ k.T) / scale          # (n_vis, n_text)
    weights = torch.softmax(scores, -1)
    return weights @ k                  # (n_vis, r)
```

No learned projections — A matrices serve as Q/K/V projectors directly (faithful to MoKA paper).

---

## 4. Token Mask Infrastructure

**Set in:** `ETrain/Models/LLaVA/llava_arch.py` → `prepare_inputs_labels_for_multimodal()`

```
token_mask value:  2 = text token,  1 = visual token,  0 = pad
```

**Propagated by:** `ETrain/Models/LLaVA/language_model/llava_llama.py` → `_apply_lora_token_mask()`
Called once per forward pass; sets `token_mask` attribute on every LoRA module.

**Used in:** `MoEMoKALoraLinear.forward()` to route tokens to correct A matrices.

---

## 5. Save / Load

### 5.1 Why Standard PEFT Save/Load Is Bypassed

`CoIN/peft/utils/save_and_load.py`:
- `get_peft_model_state_dict()` has a type whitelist that does **not** include `MOE_MOKA_CoIN` → raises `NotImplementedError`.
- `set_peft_model_state_dict()` same issue, plus key rewriting breaks for nested `lora_` keys.

**Decision:** bypass both for MoE-MoKA, use direct `torch.save` / `model.load_state_dict(strict=False)`.

### 5.2 Training Save — `save_trained_model`

**File:** `ETrain/Train/LLaVA/llava_trainer.py` lines ~444-470

```python
if getattr(training_args, 'moe_moka_enable', False):
    # Collect ZeRO-3 shards via get_peft_state_maybe_zero_3
    state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), training_args.lora_bias)
    if training_args.local_rank in (0, -1):
        self.model.config.save_pretrained(training_args.output_dir)
        torch.save(state_dict, os.path.join(training_args.output_dir, WEIGHTS_NAME))  # adapter_model.bin
        self.model.peft_config['default'].save_pretrained(training_args.output_dir)   # adapter_config.json
        torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
```

**Files written per task:**
```
<output_dir>/
├── adapter_model.bin          # MoE-MoKA weights (all lora_ params)
├── adapter_config.json        # MoEMoKALoraConfig (reconstructed by from_pretrained)
├── non_lora_trainables.bin    # mm_projector + embed_tokens
└── config.json                # LLaVA model config
```

### 5.3 Training Load — `load_model_from_previous_task`

**File:** `ETrain/Train/LLaVA/llava_trainer.py` lines ~295-330

```python
if moe_moka_enable:
    # Keys in adapter_model.bin include ".default." — matches model state dict directly
    model.load_state_dict(adapters_weights, strict=False)
else:
    set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
```

Called at the start of each task (task 2 through 8) to load the previous task's adapter before continuing training.

### 5.4 Eval Load — `load_pretrained_model`

**File:** `ETrain/Models/LLaVA/builder.py` lines ~78-104

```python
is_moe_moka = 'moka' in model_name.lower()
if is_moe_moka:
    adapter_config = MoEMoKALoraConfig.from_pretrained(model_path)
    model = get_peft_model(model, adapter_config)
    weights = torch.load(os.path.join(model_path, WEIGHTS_NAME), map_location='cpu')
    model.load_state_dict(weights, strict=False)
    # MoE-MoKA merge is not implemented; always runs unmerged.
```

The model name must contain `"moka"` (case-insensitive) to trigger this path. All MoEMoKA training scripts output to directories named `*_MoEMoKA_lora` which satisfies this requirement.

---

## 6. Training Entry Points

### 6.1 `ModelArguments.moe_moka_enable`

**File:** `ETrain/Train/LLaVA/train.py` line ~64

```python
@dataclass
class ModelArguments:
    ...
    moe_moka_enable: bool = field(default=False)
    expert_num: int = field(default=4)
```

Propagated to `training_args.moe_moka_enable` at line ~119 before `Trainer` construction.

### 6.2 PEFT Wrapping

**File:** `ETrain/Train/LLaVA/train.py` lines ~200-250 (approximate)

```python
if model_args.moe_moka_enable:
    lora_config = MoEMoKALoraConfig(
        r=training_args.lora_r,
        expert_num=model_args.expert_num,
        ...
    )
    model = get_peft_model(model, lora_config)
```

---

## 7. ZeRO-3 Compatibility

### Problem

DeepSpeed ZeRO-3 partitions parameters across GPUs and rebuilds them only when a module's `__call__` is traced at launch. Conditional `if has_text: expert.lora_A_text(flat_text)` breaks the trace because one branch may never be called on one rank.

### Fix

All A matrices are called unconditionally (dummy tensor if the modality is absent):

```python
out_text = expert.lora_A_text(flat_text)   # always called
out_vis  = expert.lora_A_vis(flat_vis)     # always called
if has_text:
    a_out[text_mask] = out_text
if has_vis:
    a_out[vis_mask]  = out_vis
```

**File:** `CoIN/peft/tuners/mokamoelora.py` lines ~220-260

---

## 8. Continual Learning Sequence

### Script Orchestration

```
scripts/LLaVA/Train_MoEMoKA/
├── coin_paths.sh             # Centralized paths; auto-detects GPUs
├── 1_Science.sh              # Task 1 training (no previous task)
├── 2_TextVQA.sh              # Task 2 training (loads Task 1 adapter)
├── ...
├── 8_OCRVQA.sh               # Task 8 training (loads Task 7 adapter)
├── run_coin_sequence.sh      # Orchestrates all 8 tasks + online eval
├── mini_pipeline_test.sh     # 3-step smoke test for all 8 tasks
└── eval/
    ├── eval_common.sh        # Shared eval setup (sources coin_paths.sh)
    ├── 1_eval_sqa.sh         # ScienceQA eval
    ├── 2_eval_textqa.sh      # TextVQA eval
    ├── ...
    └── 8_eval_ocrvqa.sh      # OCRVQA eval
```

### Task-to-Task Checkpoint Flow

```
Task 1: base model → [train] → ScienceQA_llava_MoEMoKA_lora/
Task 2: ScienceQA_llava_MoEMoKA_lora/ → [load] → [train] → TextVQA_llava_MoEMoKA_lora/
Task 3: TextVQA_llava_MoEMoKA_lora/ → [load] → [train] → ImageNet_llava_MoEMoKA_lora/
...
Task 8: VQAv2_llava_MoEMoKA_lora/ → [load] → [train] → OCRVQA_llava_MoEMoKA_lora/
```

### How to Run

**Full continual training (4 or 8 GPU):**
```bash
# From repo root, with conda env active:
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh
# Or specific task range:
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh 1 8
```

**Mini smoke test (any GPU count, ~3 steps per task):**
```bash
bash scripts/LLaVA/Train_MoEMoKA/mini_pipeline_test.sh
# Skip eval:
SKIP_EVAL=1 bash scripts/LLaVA/Train_MoEMoKA/mini_pipeline_test.sh
```

**Eval only (after training):**
```bash
# Args: STAGE  MODEL_PATH  LORA_MODE
bash scripts/LLaVA/Train_MoEMoKA/eval/1_eval_sqa.sh \
    Task1 \
    checkpoints/LLaVA/CoIN/ScienceQA_llava_MoEMoKA_lora \
    all
```

---

## 9. Known Deviations From MoKA Paper

| Item | Paper | Implementation | Reason |
|------|-------|----------------|--------|
| Cross-attn scale | `sqrt(N_t)` (number of text tokens) | `sqrt(d_k)` (rank r) | Standard attention scale; N_t varies per sample making it unstable |
| Cross-attn gate | None | `lora_attn` learnable scalar | Extra expressivity; initialized to small value |
| Continual learning | Not covered | Load previous task adapter | CoIN requirement |

---

## 10. Quick Verification

After training a task, verify the checkpoint:
```bash
ls checkpoints/LLaVA/CoIN/ScienceQA_llava_MoEMoKA_lora/
# Expected:
# adapter_config.json   adapter_model.bin   config.json   non_lora_trainables.bin
```

After eval, check result:
```bash
# ScienceQA: acc in output_result.jsonl
cat results/CoIN/LLaVA/MoEMoKA/ScienceQA/<STAGE>/output_result.jsonl | python -c "import json,sys; print(json.load(sys.stdin)['acc'])"
```
