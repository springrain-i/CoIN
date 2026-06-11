# CoIN 项目说明


## 研究目标

### 核心研究问题

MokA（NeurIPS 2025）已在单任务多模态fine-tuning场景下证明，共享LoRA参数被优势模态token所主导，导致非优势模态信息利用不足。本实验旨在验证这一现象在MCIT场景下同样成立：在多模态持续学习的设定中，对每个任务单独微调时，LoRA参数是否依然系统性地对优势模态知识编码更充分，而非优势模态始终处于弱势？

### 实验设计

| 模型 | 结构 | 角色 |
|------|------|------|
| **MoELoRA** | 每专家独立 (A_i, B_i)，无模态区分 + MoE 软路由 | 基线（预期有文本偏置） |
| **MoEMoKA** | 每专家独立 (A_text_i, A_vis_i, B_i)，专家内跨模态注意力 + MoE 软路由 | 提出方法（预期减少文本偏置） |

**测量过文本化的方式**：使用 `all` / `text` / `visual` 三种部分模态推理模式分别评估精度。若 `text` 精度 ≈ `all` 而 `visual` 精度显著低于 `all`，则说明过文本化严重。

### 三个阶段

1. **在 CoIN 上实现并验证 MoELoRA 和 MoEMoKA** — 完成两个模型在 CoIN/LLaVA pipeline 8 个任务上的训练与三模式 eval。✅ 完成

2. **预实验：验证 MCIT 场景下模态不平衡遗忘的存在性** — 对比两模型在 T8 checkpoint 下 `all` / `text` / `visual` 三种推理模式的精度。✅ 完成，结论如下：
   - visual-only 精度系统性低于 text-only 精度
   - text-only 与 all 的差距远小于 visual-only 与 all 的差距
   - 该规律在 8 个任务上普遍成立，与 MoKA 原始发现一致
   - MoEMoKA 与 MoELoRA 呈现高度相似的三线分化模式（text ≈ all ≫ visual），说明单纯将参数替换为 MoKA 结构不能消除 MCIT 场景下视觉模态被系统性边缘化的问题

3. **不平衡遗忘的成因分析** — 通过梯度和注意力分析，揭示 MCIT 下视觉模态持续边缘化的机制。🔄 进行中

---

## 当前实验状态

| 实验 | 状态 | 说明 |
|------|------|------|
| MoELoRA T8 continual training | ✅ 完成 | 检查点：`checkpoints/LLaVA/CoIN/` |
| MoELoRA T8 全量 eval（8×3模式） | ✅ 完成 | 总耗时 35.5h，日志：`logs/LLaVA/CoIN/` |
| MoEMoKA T8 continual training | ✅ 完成 | 检查点：`checkpoints/LLaVA/CoIN_MoEMoKA/OCRVQA_llava_MoEMoKA_lora` |
| MoEMoKA T8 全量 eval（8×3模式） | ✅ 完成 | 日志：`logs/LLaVA/MoEMoKA/` |
| 预实验：模态不平衡遗忘验证 | ✅ 完成 | 两模型均呈现 text ≈ all ≫ visual 三线分化 |
| 成因分析（grad / attn） | 🔄 进行中 | — |

备注：MOEMOKA的T1~T7的实验在另一服务器进行，log请去对应恒源云查看
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

## ⚠️ 训练 Batch Size 配置（启动前必须核对）

**规则：每次启动训练脚本前，必须确认对应任务的 `per_device_train_batch_size` 已按下表设置。MoEMoKA 的 bs 为 MoELoRA 对应值的一半。**

| # | 任务 | MoELoRA bs | MoEMoKA bs |
|---|------|-----------|-----------|
| 1 | ScienceQA | 512 | 256 |
| 2 | TextVQA | 1024 | 512 |
| 3 | ImageNet | 512 | 256 |
| 4 | GQA | 384 | 192 |
| 5 | VizWiz | 256 | 128 |
| 6 | Grounding | 256 | 128 |
| 7 | VQAv2 | 256 | 128 |
| 8 | OCRVQA | 256 | 128 |

以上 bs 对 single-task 和 continual 训练均适用。

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

所有路径集中在 `scripts/LLaVA/Train_MoEMoKA/coin_paths.sh`（MoEMoKA）和 `scripts/LLaVA/Train_MOE/coin_paths.sh`（MoELoRA）中，每个训练/eval 脚本都会 source 它。**换服务器时只需修改对应 `coin_paths.sh`，不要在此处记录具体路径。**

关键变量：`COIN_BASE_MODEL`、`COIN_VISION_TOWER`、`COIN_INSTR_ROOT`、`COIN_IMAGE_ROOT`、`COIN_OUTPUT_ROOT`

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
  - `Train/LLaVA/train.py` — 训练入口；`ModelArguments` 含 `expert_num`、`task_embedding_dim`、`lora_enable`、`moe_moka_enable`
  - `Train/LLaVA/train_mem.py` — 薄封装，monkey-patch flash attention 后调用 `train.py`
  - `Train/LLaVA/llama_sdpa_monkey_patch.py` — 用 `F.scaled_dot_product_attention` 替代 flash_attn；含 NaN 修复（见下方）
  - `Models/LLaVA/llava_arch.py` — 多模态模型 mixin；`prepare_inputs_labels_for_multimodal()` 构建交错的文本+视觉 embedding，并赋值 `current_lora_mask`
  - `Models/LLaVA/language_model/llava_llama.py` — `_apply_lora_token_mask()` 在每次 forward 前将 `current_lora_mask` 和 `lora_mode` 传递给每个 LoRA 模块
  - `Eval/LLaVA/CoIN/model_vqa_science.py` 等 — 评估循环；从 CLI `--lora-mode {all,text,vision}` 设置 `model.lora_mode`

- **`CoIN/peft/tuners/`** — 自定义 PEFT（fork 自 `peft==0.4.0`）
  - `lora.py` — 标准 LoRA 扩展了 `token_mask` 和 `lora_mode`
  - `coinmoelora.py` — MoE-LoRA：`CoINMOELoraConfig`，软路由（softmax 加权求和）
  - `mokamoelora.py` — **MoE-MoKA**：每个专家持有 `(lora_A_text_i, lora_A_vis_i, lora_B_i)`；含跨模态注意力模块

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

## 成因分析工具（`grad_attn` 分支）

分析代码在独立分支 `grad_attn`，worktree 挂载在 `.worktrees/grad/`。

### 核心文件

| 文件 | 作用 |
|------|------|
| `ETrain/Train/LLaVA/gradient_logger.py` | `ModalGradientLogger`：逐层记录文本/视觉梯度范数及比值 |
| `ETrain/Train/LLaVA/attention_logger.py` | `AttentionLogger`：记录 post-image text token 对 text/vis key 的注意力分布及信息流 |
| `ETrain/Train/LLaVA/train_grad.py` | 训练入口，注入两个 logger；需 `COIN_USE_SDPA_PATCH=1` |
| `scripts/LLaVA/Train_MOE/run_grad_analysis.sh` | MoELoRA T1→T8 全流程（GPU 6,7）带 grad+attn 日志 |
| `scripts/analysis/merge_grad_csvs.py` | 合并多卡 CSV |
| `scripts/analysis/plot_coin_metrics.py` | 绘图 |

### 关键指标

**梯度主导度**（`gradient_logger.py`）：
- `R_A`、`R_B`、`R_dW`：文本/视觉梯度 Frobenius 范数比（> 1 = 文本主导）
- `R_*_tok`：按 token 数归一化后的比值（剔除 token 数量差异）

**注意力分布**（`attention_logger.py`）：
- `A_tv`：post-image text query 对 visual key 的注意力比例（核心指标）
- `U_vis`、`U_text`：信息流范数（Wu et al., CoLM 2025）
- `R_info = U_vis / (U_vis + U_text)`：视觉信息流占比

### 输出路径

```
analysis/gradient_dominance/   # 梯度 CSV（每任务每层）
analysis/attn_dominance/       # 注意力 CSV（每任务每层）
logs/LLaVA/grad_analysis/      # 训练日志
```

### 运行入口

```bash
# 需先切换到 grad_attn 分支或进入 .worktrees/grad/
COIN_USE_SDPA_PATCH=1 bash scripts/LLaVA/Train_MOE/run_grad_analysis.sh 1 8
```

---

## Batch 推理修复（勿破坏）

`COIN_EVAL_BATCH_SIZE=4` 已修复三个分层 bug（精度曾从 62% 降至 28%），**修复已全部应用，不可回退**。详见 [`docs/batch_inference_fix.md`](docs/batch_inference_fix.md)。

eval 时必须设置 `COIN_USE_SDPA_PATCH=1`（`eval_common.sh` 中已设置）。

---

## 参考文档

- MoKA 论文：`/data4/home/sqx/MokA/Wei 等 - MokA Multimodal Low-Rank Adaptation for MLLMs.pdf`
- MoE-MoKA 实现说明：[`docs/MoEMoKA_Implementation.md`](docs/MoEMoKA_Implementation.md)
- Batch 推理修复：[`docs/batch_inference_fix.md`](docs/batch_inference_fix.md)

---

## 安装

见 `README.md` 安装章节（conda env `coin`，Python 3.10，需 flash-attn）。

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
