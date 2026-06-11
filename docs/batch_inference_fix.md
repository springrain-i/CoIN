# Batch 推理修复记录

`COIN_EVAL_BATCH_SIZE=4` 加速 eval，但曾有三个分层 bug 导致精度从 62% 降至 28%。以下修复已全部应用，不可回退。

## Bug 1：混合 batch 图像处理（`model_vqa_science.py`）

`prepare_inputs_labels_for_multimodal` 对每个 sample 都会递增 `cur_image_idx`，包括纯文本 sample。混合 batch 必须为纯文本 slot 提供 dummy 零图像：

```python
ref_img = next(s["image_tensor"] for s in samples if s["image_tensor"] is not None)
images = torch.stack([
    s["image_tensor"] if s["image_tensor"] is not None
    else torch.zeros_like(ref_img)
    for s in samples
])
```

HF `generate()` 返回 `output_ids` shape 为 `[B, original_input_ids_len + new_tokens]`，解码 offset 始终为 `input_ids.shape[1]`。

## Bug 2：tokenizer_padding_side 未传递给 model.config

`prepare_inputs_labels_for_multimodal` 读取 `model.config.tokenizer_padding_side` 决定图像展开后的 padding 方向。batch_size > 1 时必须设置：

```python
model.config.tokenizer_padding_side = "left"
```

已在所有 eval 脚本中设置（`model_vqa_science.py`、`model_vqa.py`、`model_text_vqa.py` 等）。

## Bug 3：SDPA monkey patch 中左 padding 的 NaN 传播

padding query position 的 attention row 全为 -inf → softmax → NaN → 通过残差连接传播。修复：在 `ETrain/Train/LLaVA/llama_sdpa_monkey_patch.py` 的 SDPA 调用后，将 padding query position 的输出置零：

```python
if attention_mask is not None and not is_decode:
    q_pad = (attention_mask[:, :q_len] == 0)
    attn_output = attn_output.masked_fill(q_pad[:, None, :, None], 0.0)
```

**eval 时必须设置 `COIN_USE_SDPA_PATCH=1`**（在 `eval_common.sh` 中已设置）。
