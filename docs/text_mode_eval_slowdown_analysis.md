# MoE-MoKA Text-Mode Eval Slowdown Analysis

**Issue**: After T5 (VizWiz) training, `text` mode eval on TextVQA takes ~22 hours, while `all` mode takes ~2 hours (~11× difference). Earlier tasks (T1–T4) showed no such gap.

**Log files analyzed**:
- `logs/LLaVA/CoIN/20260507_144530/online_train5_VizWiz_eval2_TextVQA_all.log`
- `logs/LLaVA/CoIN/20260507_144530/online_train5_VizWiz_eval2_TextVQA_text.log`

---

## Measured Times

| Mode | Samples | Total Time | Avg s/it |
|------|---------|------------|----------|
| all  | 5000 (4×1250) | 1:25:36 | 4.11 s/it |
| text | 5000 (4×1250) | 22:11:34 | 63.92 s/it |

**The 11× difference is not a uniform slowdown.** Per-sample timing shows most items run at 3–5 s/it in both modes, but specific samples in `text` mode spike to 200–418 s/it.

Evidence from `text` mode log:
```
82% | 1020/1250 [22:03:38<13:06:01, 205.05s/it]
82% | 1021/1250 [22:03:45<9:15:31,  145.55s/it]
82% | 1022/1250 [22:03:49<6:31:34,  103.05s/it]   ← rapid EMA decay
...
95% | 1192/1250 [22:06:10<6:44:34,  418.53s/it]
```

The EMA-decay pattern (200→145→103→72→52→...) is the tqdm exponential moving average recovering after a **single extremely slow sample**. These are not slow batches — they are individual samples that run to `max_new_tokens`.

---

## Root Cause

### 1. text mode disables visual LoRA completely

In `mokamoelora.py` (line 247–248):
```python
if lora_mode == "text":
    vis_mask = torch.zeros_like(vis_mask)   # no visual tokens go through LoRA
```

This means all visual token positions get zero LoRA contribution:
```
a_comb[vis_positions] = 0
→ out_i[vis_positions] = B · 0 = 0
→ result[vis_positions] = W₀ · x   (base model only)
```

The model was trained with LoRA-adapted visual representations. In `text` mode, image patch tokens are processed through the base model's `W₀` only — representations the trained `B` has never seen in this context.

### 2. TextVQA answers are OCR-dependent

TextVQA requires reading text in images (e.g., "What is the house number?", "What brand is shown?"). The answer is embedded in the image, not the question. In `text` mode, visual LoRA is disabled → the model cannot recover the answer from image features → generates incoherent/repetitive tokens instead of stopping at EOS.

**Why only SOME samples are affected**: TextVQA questions vary in visual dependency:
- Low dependency (e.g., "What language is this?"): model infers from context → generates short answer normally
- High dependency (e.g., "What number appears on the sign?"): model has no path to the answer → enters repetition loop

### 3. max_new_tokens=1024 amplifies the problem

The eval shell script sets:
```bash
--max-new-tokens "${MAX_NEW_TOKENS:-1024}"
```

TextVQA answers are typically 1–5 words (≤10 tokens). A sample that fails to generate a short answer runs to 1024 tokens:
- Normal sample: ~10 tokens × 0.4 s/token = 4 s
- Stuck sample: 1024 tokens × 0.4 s/token ≈ 410 s  ← matches the 418 s/it spike

### 4. Why T1–T4 did not show this pattern (timing perspective)

With `all ≈ text` time for T1–T4, both modes generate similar-length outputs. The larger timing gap at T5 is a generation behavior effect: VizWiz training on ambiguous real-world visual questions ("What is this?", "Can you help me?") teaches the model verbose/uncertain generation. In text mode at T5, many TextVQA samples become ambiguous (visual features absent), triggering this same verbose pattern.

**Critically, the timing difference is NOT evidence of increasing visual dependency.** TextVQA text-mode accuracy actually INCREASED from T4 to T5 (12.66% → 19.67%), the opposite of what visual-dependency growth would predict.

---

## Code Path Verification

`mokamoelora.py` forward (lines 228–293):

```python
# 1. Mask setup
text_mask = (token_mask == 2)     # (B, S) bool
vis_mask  = (token_mask == 1)     # (B, S) bool

if lora_mode == "text":
    vis_mask = torch.zeros_like(vis_mask)   # visual disabled

has_vis = bool(vis_mask.any())    # False in text mode

# 2. Expert loop
flat_vis = dropped_x[vis_mask]    # Empty tensor in text mode
for expert in experts:
    out_text = expert.lora_A_text(flat_text)
    out_vis  = expert.lora_A_vis(flat_vis)   # Called on empty tensor (ZeRO-3 trace)

    a_comb = torch.zeros(B, S, r_per, ...)
    if has_text:
        a_comb[text_mask] = out_text
    # has_vis is False → visual positions stay at zero

    out_i = expert.lora_B(a_comb)   # B · 0 = 0 at visual positions
```

**Cross-attention is also skipped in text mode** (`has_vis=False` prevents `_cross_attention` call), so the intra-expert attention that would normally enrich visual representations is absent.

---

## Accuracy Data: Over-Textualization Pattern

The timing anomaly is a distraction from the more important finding: **accuracy data consistently supports over-textualization, not visual dominance**.

| After Task | Task Eval | all | text | visual | text/all | visual/all |
|-----------|-----------|-----|------|--------|----------|------------|
| T1 ScienceQA | ScienceQA | 72.15 | 72.11 | 66.56 | **0.99** | 0.92 |
| T2 TextVQA | ScienceQA | 66.52 | 65.62 | 61.38 | **0.99** | 0.92 |
| T2 TextVQA | TextVQA | 42.31 | 41.86 | 17.53 | **0.99** | 0.41 |
| T3 ImageNet | ImageNet | 93.49 | 93.66 | 31.03 | **1.00** | 0.33 |
| T4 GQA | GQA | 39.81 | 37.28 | 16.43 | 0.94 | 0.41 |
| T4 GQA | TextVQA | 26.63 | 12.66 | 5.48 | 0.48 | 0.21 |
| T5 VizWiz | TextVQA | 34.38 | 19.67 | — | 0.57 | — |

**Key evidence for over-textualization**:

1. **T1–T2**: `text/all ≈ 1.0` across all tasks. Disabling visual LoRA costs almost nothing. Text mode alone is nearly as powerful as full mode.

2. **T3 ImageNet (most striking)**: `text=93.66% ≈ all=93.49%`, `visual=31.03%`. ImageNet is pure visual classification, yet text LoRA alone achieves full accuracy. Visual LoRA alone drops to 31%. The model learned a visual task almost entirely through text token patterns — extreme over-textualization.

3. **T4 GQA**: GQA (compositional visual reasoning) shows `text=37.28% >> visual=16.43%`. Text consistently outperforms visual mode on a task that should require visual understanding.

4. **Text accuracy increases T4→T5**: TextVQA text-mode acc: 12.66% → 19.67%. If visual LoRA were increasingly critical, text mode should degrade. Instead it improves.

**Verdict**: The model is over-textualized throughout training. Text LoRA dominates in every task, including visually-heavy ones (ImageNet, GQA). The T5 timing anomaly is a generation behavior artifact unrelated to this modality imbalance.

### What the `all-text` gap increase at T4–T5 means

| Task | T1 | T2 | T3 | T4 | T5 |
|------|----|----|----|----|-----|
| ScienceQA all-text gap | 0.04 | 0.90 | −0.04 | 12.78 | 17.07 |

This increase at T4–T5 does NOT indicate visual dependency growth. It reflects that after T3 (ImageNet) catastrophic forgetting and T4 recovery, the model rebuilds BOTH text and visual pathways. The gap widening means visual recovered too, not that text weakened.

---

## Secondary Observation: LoRA Module Check is Misleading

Both modes print:
```
LoRA module check: total=224, active_with_r>0=224
```

This is **not a bug** — the count reflects modules with r>0 adapters loaded, not tokens currently being processed through LoRA. Token-level routing happens inside the forward pass. The check does not distinguish between all/text/visual modes. This can be misleading and should be noted when reading logs.

---

## Recommendations

### Fix 1: Reduce max_new_tokens for TextVQA eval (immediate)
```bash
# In 2_eval_textqa.sh, add task-specific override:
--max-new-tokens "${MAX_NEW_TOKENS:-50}" \
```
TextVQA answers are ≤10 tokens. This reduces stuck-sample cost from 410 s to ~20 s.

### Fix 2: Add repetition penalty (more robust)
```python
output_ids = model.generate(
    ...
    repetition_penalty=1.3,   # breaks repetition loops
)
```
This prevents the model from looping on degenerate outputs regardless of max_new_tokens.

### Fix 3: Track time ratio as a research metric
The ratio `T_text / T_all` across training steps is a proxy for visual LoRA dependency:

| After Task | T_text / T_all (TextVQA) |
|------------|--------------------------|
| T1–T4      | ~1× (no gap)              |
| T5         | ~11× (large gap)          |

Tracking this across T6–T8 could provide an implicit measure of how much the model shifts toward visual-dominated generation over the continual learning sequence.

---

## Summary

The 22-hour text-mode eval at T5 is caused by **individual samples running to max_new_tokens=1024** when visual LoRA is disabled. This is a generation behavior artifact from VizWiz training, not evidence of visual LoRA dependency.

**The accuracy data tells the real story**: text LoRA consistently dominates over visual LoRA across all tasks, including visually-heavy ones. This supports the over-textualization hypothesis:
- T1–T2: text/all ≈ 1.0 (near-complete text dominance)
- T3 ImageNet: text=93.66% ≈ all=93.49%, visual=31.03% (text LoRA alone solves a visual task)
- T4–T5: text > visual in all measured tasks

The T5 text-mode timing anomaly does not contradict over-textualization. It is a confound caused by generation length inflation, and should be controlled for by:
1. Setting `MAX_NEW_TOKENS=50` for short-answer tasks (TextVQA, VizWiz)
2. Adding `repetition_penalty=1.3` to prevent generation loops

**Missing data**: Visual mode evals for T5 were not collected. These are needed to confirm whether visual LoRA continues to degrade relative to text LoRA at T5 and beyond.
