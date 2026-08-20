# CoIN 仓库规范

## 研究主线

本仓库服务于 **“Where Does Multimodal Continual Learning Forget?”**：沿
LLaVA 的完整前向计算路径，定位**样本级的本质 MCIT 遗忘**。不得预设所有遗忘都
来自 projector 或 LLM。

对一个旧任务样本，刚完成该任务时的 checkpoint 是 **early checkpoint**；之后继续
学习若干任务得到的 checkpoint 是 **final checkpoint**。主要分析 cohort 的构造规则为：

1. 排除 ill-defined samples：图像并非回答所必需、答案/评分存在实质歧义，或 final
   答案在语义上实际可接受；
2. 保留 `early-correct -> final-wrong` 的样本；
3. 将仅有输出格式或回答风格变化的样本标为 `superficial_forgetting` 并排除；
4. 对剩余依赖视觉证据的语义错误，标为 `essential_forgetting` 并进行机制分析。

没有完成对应实验与控制组时，不得将统计结果、因果归因或保护效果表述为研究发现。
尤其是，表示漂移和 attention 漂移仅是观察性信号，不能单独证明故障位置。

## 新实验的标准训练协议

除非某个实验明确说明并论证了偏离原因，新的 MoELoRA CoIN discovery/confirmation
实验必须使用以下设置。

| 项目 | 必须设置 |
| --- | --- |
| 骨干模型 | LLaVA-1.5-7B：Vicuna-7B-v1.5 + CLIP ViT-L/14-336px |
| 可训练参数 | 仅 `mm_projector`、MoELoRA experts 和 routers；冻结视觉编码器与 LLM 基础权重 |
| MoELoRA 目标层 | 全部 32 层的 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` |
| experts / rank / alpha | 8 experts、总 rank 128（每个 expert rank 16）、`lora_alpha=256` |
| 训练时长 | 每个任务 1 epoch |
| 有效全局 batch | 每个任务均为 **128** |
| 优化 | MoELoRA/router LR `2e-4`；projector LR `2e-5`；warm-up ratio `0.03`；cosine decay |
| 任务顺序 | ScienceQA → TextVQA → ImageNet → GQA → VizWiz → Grounding → VQAv2 → OCRVQA |

有效全局 batch 为
`per_device_train_batch_size * gradient_accumulation_steps * world_size`。
不得将 global batch 写成 `per_device_train_batch_size`。启动训练前必须记录三项因子，
并验证其乘积为 128。

`scripts/zero3_offload.json` 不得定义 DeepSpeed 的 `optimizer` 或 `scheduler`：必须由
Transformers 创建 optimizer 并执行启动脚本指定的 `--lr_scheduler_type cosine`。
`LLaVATrainer.create_optimizer()` 必须保留 `mm_projector_lr` 的独立 projector 参数组，
不能将其并入主 `learning_rate` 参数组。

## 因果定位方案

沿完整前向路径分析。所有 component restoration 均以 final checkpoint 为基础，并从
匹配的 early checkpoint 恢复对应模块。

| 计算阶段 | 干预接口 | 解释规则 |
| --- | --- | --- |
| Cross-modal projection (P) | 恢复 early `mm_projector` | 仅部分恢复只支持 projector 介导部分遗忘，不能声称 projector 是普遍原因 |
| Attention routing (QK) | 联合恢复 early `q_proj` + `k_proj` 的 MoELoRA experts **和 routers** | 因果检验 routing；不能由 RAPT 直接推断 |
| Evidence transmission (VO) | 联合恢复 early `v_proj` + `o_proj` 的 experts 和 routers | 检验被选择证据的提取、聚合与 residual 写回 |
| Feature transformation (F) | 联合恢复 early `gate_proj` + `up_proj` + `down_proj` 的 experts 和 routers | 检验 gated MLP 的特征变换 |

每个 essential-forgetting 样本均须报告完整 causal restoration profile：
`CRP = (R_P, R_QK, R_VO, R_F)`。之后再扫描 early/middle/late layer blocks；对单一
阶段无法恢复的样本，继续测试 QK+VO、QK+F、VO+F 及完整 LLM MoELoRA restoration。
任意交换都必须保持冻结的 LLM base weights 不变。

每个因果主张都应按需包含匹配对照：non-forgotten 与 positive-transfer、
wrong-checkpoint swap、known corruption、matched-norm/equal-budget、counterfactual
image pairs 以及方向性或 factor-exchange 对照。进入 confirmatory run 前，冻结 cohort
准则、阶段划分、阈值和 where-to-protect 规则。

保护实验应比较 matched-stage protection（projector、QK、VO 或 F）与等预算的
wrong-stage、random-stage、uniform、no-protection。确认实验优先覆盖独立 seed、反向或
随机任务顺序，并在可行时使用不同模型家族。

## 环境与 GPU 安全

所有 GPU 工作前，执行：

```bash
conda activate coin
export CUDA_HOME=$CONDA_PREFIX
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
nvidia-smi
```

服务器通常使用 tmux session `coin`。路径配置集中于
`scripts/LLaVA/Train_MOE/coin_paths.sh`；不得在 launcher 中硬编码服务器相关的数据、
模型或输出路径。每个新 GPU run 必须有唯一 run ID，并保存启动命令、checkpoint 来源、
seed、任务顺序、代码 commit 和输出/日志目录。

### Projector 继承是强制要求

从任务 2 开始，持续训练必须从上一任务的 `non_lora_trainables.bin` 加载 projector。
加载器必须提取 `mm_projector` 张量，并直接载入
`model.get_model().mm_projector`，且使用 `strict=True`。缺失、重复或多余的 projector
键均必须立刻失败，并输出：

```text
[ProjectorLoad] source=..., matched=4, missing=0, unexpected=0
```

若没有此检查或检查失败，不得继续该训练链。

## LLaVA 评测安全规则（强制）

- 默认使用 `--lora-mode all`、batch size 4 和 SDPA；`COIN_EVAL_BATCH_SIZE` 仅用于
  明确提出的 override。
- 所有支持 `--batch-size > 1` 的 evaluator，都必须在加载模型**之前**调用
  `replace_llama_attn_with_sdpa()`。只有显式设置 `COIN_USE_SDPA_PATCH=0` 才能在受控
  诊断中关闭。
- 必须保留生成上限：T1=10、T2=40、T3=20、T4=20、T5=50、T6=50、T7=50、T8=150。
  添加 `--batch-size` 时不得替换 `--max_new_tokens`。
- 大量空答案或突然出现数小时 ETA 视为验证失败：立即停止任务、诊断后再恢复。
- 训练与评测都必须遵从 `model.config.image_aspect_ratio`。使用
  `process_images(..., model.config)`，不要直接调用 CLIP preprocess。Grounding eval 必须
  按顺序对齐预测与标注、检查长度和 ID，且不得盲目对 bbox 使用 `[1:-1]`。

## 关键代码与产物

- `ETrain/Train/LLaVA/train.py`、`train_mem.py`：训练入口。
- `ETrain/Train/LLaVA/llava_trainer.py`：上一任务加载与 optimizer 参数组。
- `ETrain/Models/LLaVA/utils.py`：LLaVA 构建与 MoELoRA 注入。
- `CoIN/peft/tuners/coinmoelora.py`：experts 与 routers。
- `ETrain/Eval/LLaVA/CoIN/`：任务 evaluator；`eval_grounding.py` 包含对齐与 bbox
  evaluator。
- `scripts/projector_analysis/`：projector-restoration 分析脚本与证据。

## Git 纪律

每个 coherent stage 单独提交；不要把 baseline snapshot、protocol 修改、实验运行和分析
混入同一 commit。使用清晰前缀：

```text
baseline: ...
protocol: ...
exp-config: ...
exp-run: ...
analysis: ...
report: ...
```

任何偏离本文件的设置，都必须写入 commit message 和 experiment report。
