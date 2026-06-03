# CoIN 项目说明

本文件为 Claude Code 在此仓库工作时提供上下文和指导。

---

## 研究目标

### 核心研究问题

MoKA（NeurIPS 2025）在**单任务多模态**设置下已证明能解决"过文本化"问题——标准 LoRA 的共享 A 矩阵因文本 token 梯度占优，导致视觉模态表征退化。

**本项目的问题是：持续学习（8 任务顺序训练）会不会重新引入或加剧过文本化？**

即使 MoKA 在单任务上有效，顺序学习更多文字密集型任务后，模态特定 A 矩阵仍可能向文本方向漂移。

### 实验设计

| 模型 | 结构 | 角色 |
|------|------|------|
| **MoELoRA** | 共享 A 矩阵 + MoE 路由 | 基线（预期有文本偏置） |
| **MoEMoKA** | 模态特定 A 矩阵 + MoE 路由 | 提出方法（预期减少文本偏置） |

**测量过文本化的方式**：使用 `all` / `text` / `visual` 三种部分模态推理模式分别评估精度。若 `text` 精度 ≈ `all` 而 `visual` 精度显著低于 `all`，则说明过文本化严重。

### 三个阶段

1. **在 CoIN 上跑 MoKA** — 实现并验证 MoKA 在 CoIN/LLaVA pipeline 8 个任务上能正确训练和评估。✅ 完成
2. **构建 MoE-MoKA** — 每个专家持有各自的模态特定 `(A^text, A^visual)` 矩阵，加上 MoE 软路由。✅ 完成
3. **分析持续学习中的过文本化** — 对比 MoELoRA 和 MoEMoKA 在 T8 checkpoint 下三种推理模式的精度差异。🔄 进行中

---

## 当前实验状态

| 实验 | 状态 | 说明 |
|------|------|------|
| MoELoRA T8 continual training | ✅ 完成 | 检查点：`checkpoints/LLaVA/CoIN/` |
| MoELoRA T8 全量 eval（8×3模式） | ✅ 完成 | 总耗时 35.5h，日志：`logs/LLaVA/CoIN/` |
| MoEMoKA T8 continual training | ✅ 完成 | 检查点：`checkpoints/LLaVA/CoIN_MoEMoKA/OCRVQA_llava_MoEMoKA_lora` |
| MoEMoKA T8 eval（all 模式） | 🔄 进行中 | T7_all 运行中，日志：`logs/LLaVA/MoEMoKA/` |
| MoEMoKA T8 eval（text/visual 模式） | ⏳ 待跑 | 等 all 模式完成后开始 |

**已有初步结果（T1-T6 all 模式）**：MoEMoKA 比 MoELoRA 平均快 **2.4×**，Grounding 任务最显著 **2.9×**。

---

## 环境配置

```bash
conda activate coin
```

DeepSpeed CPUAdam 需要以下环境变量（每个新 shell 训练前设置）：
```bash
export CUDA_HOME=$CONDA_PREFIX
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
```

网络代理（此服务器访问网络必需）：
```bash
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7891
```

所有 GPU 工作在 tmux session `coin` 中运行。启动 GPU 任务前先检查 `nvidia-smi`。

---

## 入口脚本

### MoEMoKA（当前主要实验）

```bash
# 持续训练（T1→T8 顺序）
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh

# T8 eval（在 tmux coin 中运行）
bash scripts/LLaVA/Train_MoEMoKA/run_evals_T8.sh <CKPT_DIR>

# 等待 T8 训练完成后自动触发 eval
bash scripts/LLaVA/Train_MoEMoKA/wait_and_eval_T8.sh
```

### MoELoRA（基线，已完成）

```bash
# 持续训练
bash scripts/LLaVA/Train_MOE/run_coin_sequence.sh

# 单任务基线
bash scripts/LLaVA/Train_MOE/run_coin_single.sh
```

### 工具脚本

```bash
# GPU 运行时切换 watchdog（后台运行）
bash scripts/watch_and_switch_gpus.sh <main_log_file>

# Batch 推理修复验证
COIN_USE_SDPA_PATCH=1 python scripts/test_batch_fix.py
COIN_USE_SDPA_PATCH=1 python scripts/test_image_batch_fix.py
```

---

## 固定 CoIN 任务顺序

所有持续学习实验使用此固定 8 任务序列：

| # | 任务 | 数据集 |
|---|------|--------|
| 1 | ScienceQA | 多选科学 VQA |
| 2 | TextVQA | 图像中的文字阅读 |
| 3 | ImageNet | 图像分类 |
| 4 | GQA | 组合推理 VQA |
| 5 | VizWiz | 真实场景 VQA |
| 6 | Grounding | 区域定位 |
| 7 | VQAv2 | 通用 VQA |
| 8 | OCRVQA | OCR 类 VQA |

---

## 路径配置

所有路径集中在对应的 `coin_paths.sh` 中，每个训练/eval 脚本都会 source 它。

### 公共路径

| 变量 | 默认值 |
|------|--------|
| `COIN_BASE_MODEL` | `/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5` |
| `COIN_VISION_TOWER` | `/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/clip-vit-large-patch14-336` |
| `COIN_INSTR_ROOT` | `/data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original` |
| `COIN_IMAGE_ROOT` | `/data4/wxl/MoBLoRA-backup/CoIN/cl_dataset` |

### 检查点路径

| 实验 | 路径 |
|------|------|
| MoEMoKA T8（OCRVQA，最终） | `checkpoints/LLaVA/CoIN_MoEMoKA/OCRVQA_llava_MoEMoKA_lora` |
| MoEMoKA T7（VQAv2） | `checkpoints/LLaVA/CoIN_MoEMoKA/VQAv2_llava_MoEMoKA_lora` |
| MoELoRA T8 | `checkpoints/LLaVA/CoIN/`（各任务子目录） |

### 日志路径

| 实验 | 路径 |
|------|------|
| MoEMoKA eval | `logs/LLaVA/MoEMoKA/T8_eval_T{N}_{mode}.log` |
| MoELoRA eval | `logs/LLaVA/CoIN/eval_online_trainT8_evalT{N}_{mode}_*.log` |

### 结果路径

```
results/CoIN/LLaVA/
├── MoEMoKA/          # MoEMoKA eval 结果（各任务子目录）
├── metrics/          # CSV 汇总表
└── CoIN/             # MoELoRA eval 结果
```

---

## 代码架构

### 包结构

- **`ETrain/`** — 主训练/评估框架
  - `Train/LLaVA/train.py` — 训练入口；`ModelArguments` 含 `expert_num`、`task_embedding_dim`、`lora_enable`、`moka_enable`
  - `Train/LLaVA/train_mem.py` — 薄封装，monkey-patch flash attention 后调用 `train.py`
  - `Train/LLaVA/llama_sdpa_monkey_patch.py` — 用 `F.scaled_dot_product_attention` 替代 flash_attn；含 NaN 修复（见下方）
  - `Models/LLaVA/llava_arch.py` — 多模态模型 mixin；`prepare_inputs_labels_for_multimodal()` 构建交错的文本+视觉 embedding，并赋值 `current_lora_mask`
  - `Models/LLaVA/language_model/llava_llama.py` — `_apply_lora_token_mask()` 在每次 forward 前将 `current_lora_mask` 和 `lora_mode` 传递给每个 LoRA 模块
  - `Eval/LLaVA/CoIN/model_vqa_science.py` 等 — 评估循环；从 CLI `--lora-mode {all,text,vision}` 设置 `model.lora_mode`

- **`CoIN/peft/tuners/`** — 自定义 PEFT（fork 自 `peft==0.4.0`）
  - `lora.py` — 标准 LoRA 扩展了 `token_mask` 和 `lora_mode`
  - `coinmoelora.py` — MoE-LoRA：`CoINMOELoraConfig`，软路由（softmax 加权求和）
  - `mokamoelora.py` — **MoE-MoKA**：每个专家持有 `(lora_A_text_i, lora_A_visual_i)`，共享 `lora_B`；含跨模态注意力模块

### 部分模态掩码（lora_mode）

Token mask 编码（在 `llava_arch.py` 中赋值）：
- `2` = 文本 token
- `1` = 视觉 token
- `0` = padding（始终排除）

`_get_effective_token_mask()` 按 `lora_mode` 转换：
- `"all"` → `mask > 0`（文本 + 视觉）
- `"text"` → `mask == 2`
- `"vision"` → `mask == 1`

### MoEMoKA 配置

```json
{
  "peft_type": "MOE_MOKA_CoIN",
  "lora_experts": 8,
  "r": 32,
  "lora_alpha": 64
}
```

---

## Batch 推理修复（勿破坏）

`COIN_EVAL_BATCH_SIZE=4` 加速 eval，但曾有三个分层 bug 导致精度从 62% 降至 28%。**以下修复已全部应用，不可回退。**

### Bug 1：混合 batch 图像处理（`model_vqa_science.py`）

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

### Bug 2：tokenizer_padding_side 未传递给 model.config

`prepare_inputs_labels_for_multimodal` 读取 `model.config.tokenizer_padding_side` 决定图像展开后的 padding 方向。batch_size > 1 时必须设置：

```python
model.config.tokenizer_padding_side = "left"
```

已在所有 eval 脚本中设置（`model_vqa_science.py`、`model_vqa.py`、`model_text_vqa.py` 等）。

### Bug 3：SDPA monkey patch 中左 padding 的 NaN 传播

padding query position 的 attention row 全为 -inf → softmax → NaN → 通过残差连接传播。修复：在 `ETrain/Train/LLaVA/llama_sdpa_monkey_patch.py` 的 SDPA 调用后，将 padding query position 的输出置零：

```python
if attention_mask is not None and not is_decode:
    q_pad = (attention_mask[:, :q_len] == 0)
    attn_output = attn_output.masked_fill(q_pad[:, None, :, None], 0.0)
```

**eval 时必须设置 `COIN_USE_SDPA_PATCH=1`**（在 `eval_common.sh` 中已设置）。

---

## GPU 运行时切换机制

eval 过程中可以不重启进程切换 GPU：

```bash
# 写入 override 文件（立即生效于下一个任务）
echo "0,1,2,3" > /tmp/coin_gpu_override

# 或使用 watchdog 脚本自动切换（在后台运行）
bash scripts/watch_and_switch_gpus.sh <main_log_file> &
```

`eval_common.sh` 在每个任务开始时读取 `/tmp/coin_gpu_override`，若存在则覆盖 `CUDA_VISIBLE_DEVICES`。

---

## MoKA 原理（NeurIPS 2025）

论文：`/data4/home/sqx/MokA/Wei 等 - MokA Multimodal Low-Rank Adaptation for MLLMs.pdf`

### 过文本化问题

标准 LoRA 的单一共享 A 矩阵因文本 token 梯度更强，导致 A 被过度优化为文本方向。部分模态推理时，`text-only` 精度接近 `all`，但 `vision-only` 精度显著下降。

### MoKA 架构

用三个组件替换单一 A 矩阵，保持 B 共享：

1. **模态特定 A 矩阵**：`A^text` 和 `A^visual` 各自独立，防止文本梯度污染视觉压缩子空间
2. **任务中心跨注意力**：视觉 token 以文本 token 为 K/V 进行注意力，注入任务描述上下文
3. **共享多模态 B**：将增强后的单模态表征投影到输出空间

```
h = W_0·x + B·[A^t·x^t  ;  A^v·x^v + Att(A^v·x^v, A^t·x^t, A^t·x^t)]
```

### MoE-MoKA 扩展

N 个专家各持有 `(A^text_i, A^visual_i)`，共享 B，软路由：

```
ΔW·x = B · Σ_i  w_i · [A^text_i·x^text ;  A^visual_i·x^visual + Att_i(...)]
```

实现文件：`CoIN/peft/tuners/mokamoelora.py`

---

## 安装

```bash
conda create -n coin python=3.10 -y
conda activate coin
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

---

## Git 规范

每个实验阶段一个 commit，提交信息格式：

```
baseline: 新实验前的快照
exp-config: <配置变更说明>
exp-run: <实验描述>
analysis: <指标/绘图变更>
report: <发现总结>
```

不同实验阶段不合并为一个 commit。
