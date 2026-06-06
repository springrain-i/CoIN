# ETrain/Train/LLaVA — 文件说明

## 训练入口

| 文件 | 用途 |
|------|------|
| `train_mem.py` | **MoEMoKA / MoELoRA 训练入口**。在 import 阶段调用 Flash Attention monkey-patch，然后调用 `train.py` 的 `train()` |
| `train.py` | 核心训练逻辑：解析参数、构建模型、调用 LLaVATrainer |
| `llava_trainer.py` | 自定义 Trainer：包含 `save_trained_model()`（保存 adapter 权重）、`_save_checkpoint()` 等 |
| `train_xformers.py` | xFormers attention 版本，MoEMoKA 中未使用 |

---

## Attention Monkey-Patch 文件

训练和 eval 使用不同的 attention 实现，各有原因：

### `llama_flash_attn_monkey_patch.py` — 训练专用

**何时激活**：`train_mem.py` 在启动时无条件 import 并执行 `replace_llama_attn_with_flash_attn()`

**作用**：把 LLaMA 的标准 attention 替换为 Flash Attention 2 的 CUDA kernel

**为什么训练用它**：
- 比 PyTorch 原生 attention 快 2-4×
- 显存从 O(N²) 降到 O(N)，支持更长序列
- 训练时序列是 right-padded（无左 padding 问题），不需要特殊处理

**依赖**：`flash-attn` 包（需单独安装）

---

### `llama_sdpa_monkey_patch.py` — Eval 专用

**何时激活**：`eval_common.sh` 设置 `COIN_USE_SDPA_PATCH=1` 环境变量，eval 脚本在启动时 import 并执行

**作用**：把 LLaMA 的 attention 替换为 `F.scaled_dot_product_attention`（PyTorch 2.0 内置）

**为什么 eval 不能用 Flash Attention**：

Batch eval 时使用 **left padding**（让所有样本对齐到右侧）。Flash Attention 对 padding query 位置的 attention row 处理会产生 NaN：

```
padding query → attention score 全为 -inf → softmax → NaN → 残差传播 → 整个输出变 NaN
```

SDPA 版本在 attention 计算后加了一行修复：

```python
# 将 padding query 位置的输出置零，阻断 NaN 传播
q_pad = (attention_mask[:, :q_len] == 0)
attn_output = attn_output.masked_fill(q_pad[..., None], 0.0)
```

Flash Attention 是黑箱 CUDA kernel，无法在其内部插入这个修复，所以 eval 改用 SDPA。

**依赖**：PyTorch 2.0+（内置，无需额外安装）

---

## 总结

```
训练：train_mem.py → llama_flash_attn_monkey_patch.py（速度优先）
Eval：COIN_USE_SDPA_PATCH=1  → llama_sdpa_monkey_patch.py（正确性优先，修复 left-padding NaN）
```
