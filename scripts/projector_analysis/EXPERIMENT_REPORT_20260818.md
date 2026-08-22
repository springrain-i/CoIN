# 历史 MoE-LoRA：Early `mm_projector` 替换实验汇报

> 历史记录：本报告使用的是旧的 MoE-LoRA checkpoint，与当前
> `coin_lora_zero2_gbs128_seed42_20260820_2110` 标准 LoRA 实验无关。本文数值不得
> 用作当前标准 LoRA projector swap 的结果或结论；当前实验协议见 `README.md`。

## 1. 实验摘要

本实验研究多模态持续学习过程中，视觉特征到语言模型特征空间的映射模块
`mm_projector` 是否发生了任务相关遗忘。具体做法是保留正序持续学习完成 T8
（OCRVQA）后的最终 MoE-LoRA adapter、router 和模型配置，仅将最终 checkpoint
中的 `mm_projector` 替换为对应早期任务刚训练完成时保存的 projector，然后在该
projector 对应的任务上重新评估。

实验采用对角线评估协议：

- T1 projector → T1 ScienceQA
- T2 projector → T2 TextVQA
- T3 projector → T3 ImageNet
- T4 projector → T4 GQA
- T5 projector → T5 VizWiz
- T6 projector → T6 Grounding
- T7 projector → T7 VQAv2
- T8 early projector 与 T8 final projector 相同，因此不重复评估

全部 7 个实验均已完成，预测数量与数据集预期数量完全一致。与旧 T8 final
checkpoint 回测结果相比，early projector 在 5/7 个任务上取得更高准确率，
T1–T7 非加权平均准确率从 29.12% 提高到 41.31%，增加 12.19 个百分点。不过，
对应任务刚训练完成时的完整 early checkpoint 平均准确率为 57.39%，说明只恢复
projector 通常只能恢复部分能力，不能还原 early checkpoint 的全部任务性能。

## 2. 研究问题与实验假设

### 2.1 `mm_projector` 的功能

LLaVA 中的 `mm_projector` 是视觉编码器与大语言模型之间的特征映射模块。本实验
所用配置为 `mlp2x_gelu`：它把 CLIP vision tower 输出的 1024 维视觉特征映射到
Vicuna/LLaMA 的 4096 维 hidden space，使视觉 token 能与文本 token 一起进入语言
模型。

本实验中的 projector 包含四个可训练张量：

| 张量 | 形状 |
| --- | --- |
| `mm_projector.0.weight` | `[4096, 1024]` |
| `mm_projector.0.bias` | `[4096]` |
| `mm_projector.2.weight` | `[4096, 4096]` |
| `mm_projector.2.bias` | `[4096]` |

这些参数保存在 checkpoint 的 `non_lora_trainables.bin` 中。当前 checkpoint 的该
文件不含需要保留的其他非-projector 参数，因此构建混合 checkpoint 时可以只替换
该文件，并保持 T8 的 adapter、router 和配置不变。

### 2.2 实验假设

如果 T8 final projector 在持续学习过程中遗忘了早期任务所需的视觉—语言映射，
那么把任务 Tn 刚训练完成时的 projector 放回 T8 final 模型后，Tn 的准确率应当
回升。反之，如果准确率不升或下降，则可能说明：

1. 该任务的遗忘主要发生在 adapter/router 等其他模块；
2. 最终 projector 学到了更具迁移性的映射；
3. early projector 与最终 adapter/router 存在参数协同不匹配。

## 3. 实验设计

### 3.1 固定任务顺序

| Task ID | 数据集 | 任务类型 |
| ---: | --- | --- |
| T1 | ScienceQA | 多选科学 VQA |
| T2 | TextVQA | 图像文字理解 |
| T3 | ImageNet | 图像分类 |
| T4 | GQA | 组合推理 VQA |
| T5 | VizWiz | 真实场景 VQA |
| T6 | Grounding | 区域定位 |
| T7 | VQAv2 | 通用 VQA |
| T8 | OCRVQA | OCR 类 VQA |

### 3.2 被固定和被替换的参数

| 组成部分 | 来源 | 是否变化 |
| --- | --- | --- |
| Vicuna base model | 固定 base checkpoint | 否 |
| CLIP vision tower | 固定 vision checkpoint | 否 |
| MoE-LoRA adapter/router | T8 OCRVQA final checkpoint | 否 |
| 模型与 adapter 配置 | T8 OCRVQA final checkpoint | 否 |
| `mm_projector` | 对应 T1–T7 early checkpoint | **是** |
| prompt、数据集、scorer | 各任务原有 eval 脚本 | 否 |

最终模型 checkpoint 为：

```text
checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
```

其中基线评估实际使用的主要文件为：

```text
adapter_model.bin          # T8 final MoE-LoRA adapter/router
non_lora_trainables.bin    # T8 final mm_projector
adapter_config.json
config.json
```

本实验没有使用其内部的 `checkpoint-646/` 子目录，而是使用训练正常结束后写入的
顶层最终输出文件。

### 3.3 混合 checkpoint 构建

每个混合 checkpoint 以 T8 final checkpoint 为主体，保留 T8 的
`adapter_model.bin`，并用 Tn checkpoint 的 `non_lora_trainables.bin` 替换 T8 的
对应文件。大体路径如下：

```text
checkpoints/LLaVA/CoIN_projector_swap/
├── early_T1_ScienceQA__final_T8_OCRVQA/OCRVQA_llava_MOE_lora/
├── early_T2_TextVQA__final_T8_OCRVQA/OCRVQA_llava_MOE_lora/
├── early_T3_ImageNet__final_T8_OCRVQA/OCRVQA_llava_MOE_lora/
├── early_T4_GQA__final_T8_OCRVQA/OCRVQA_llava_MOE_lora/
├── early_T5_VizWiz__final_T8_OCRVQA/OCRVQA_llava_MOE_lora/
├── early_T6_Grounding__final_T8_OCRVQA/OCRVQA_llava_MOE_lora/
└── early_T7_VQAv2__final_T8_OCRVQA/OCRVQA_llava_MOE_lora/
```

每个目录中的 `swap_manifest.json` 记录 early/final checkpoint 来源、文件 SHA256、
projector 张量名称和形状、模型配置、Git 状态与构建时间，可用于复核实验输入。
T1–T7 混合 checkpoint 的 `adapter_model.bin` 均与 T8 final adapter 的 SHA256
`f5129f1fdd90258e1b8235531fdfb5c1772cb9558f477c30c61a32ae463a272d`
一致。

### 3.4 评估配置

| 配置项 | 当前 projector 实验 |
| --- | --- |
| 模式 | `all` |
| GPU | 8 张卡 |
| 调度方式 | 一次一个任务，每个任务占用 8 卡；T1→T7 顺序运行 |
| 数据切分 | 8 chunks |
| eval batch size | 每个 GPU worker 为 4 |
| SDPA | 默认启用 |
| 训练 | 不进行训练，纯 eval |

任务级生成上限保持为仓库规定值：ScienceQA=10、TextVQA=40、ImageNet=20、
GQA=20、VizWiz=50、Grounding=50、VQAv2=50。完整运行前已针对 BS=1 与 BS=4
进行回归检查；正式结果中没有异常空答案、NaN、OOM 或预测缺失。

运行 ID：

```text
20260818_020720
```

启动命令：

```bash
conda activate coin
bash scripts/projector_analysis/run_all_forward_projector_swaps.sh
```

## 4. 实验结果

### 4.1 准确率主结果

表中加入三个模型状态：

1. **Early 完整 checkpoint**：任务 Tn 刚训练结束时的 projector、adapter 和 router；
2. **Projector 替换模型**：Tn early projector 与 T8 final adapter/router 的组合；
3. **T8 final 完整 checkpoint**：未经替换的 T8 final projector、adapter 和 router。

所有数值均为 `all` 模式准确率，变化量单位为百分点（percentage points, pp）。

| 任务 | Early 完整 checkpoint：Tn projector + Tn adapter | Projector 替换模型：Tn projector + T8 adapter | T8 final 完整 checkpoint：T8 projector + T8 adapter | 替换模型 vs. Early | 替换模型 vs. T8 final |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 ScienceQA | 78.50 | 66.42 | 62.70 | **-12.07 pp** | **+3.73 pp** |
| T2 TextVQA | 47.48 | 30.44 | 39.20 | **-17.04 pp** | **-8.76 pp** |
| T3 ImageNet | 96.42 | 64.87 | 3.80 | **-31.55 pp** | **+61.07 pp** |
| T4 GQA | 48.11 | 30.90 | 33.34 | **-17.21 pp** | **-2.44 pp** |
| T5 VizWiz | 54.50 | 31.95 | 24.38 | **-22.55 pp** | **+7.57 pp** |
| T6 Grounding | 11.33 | 8.99 | 0.52 | **-2.34 pp** | **+8.47 pp** |
| T7 VQAv2 | 65.40 | 55.63 | 39.92 | **-9.77 pp** | **+15.71 pp** |
| **T1–T7 非加权平均** | **57.39** | **41.31** | **29.12** | **-16.08 pp** | **+12.19 pp** |

这三列的比较含义不同：

- “替换模型 vs. T8 final”主要观察在最终 adapter/router 固定时，换回 early
  projector 是否改善任务性能；
- “替换模型 vs. Early”观察只保留 early projector、但使用 T8 adapter/router 后，
  与原始任务刚训练完成状态还相差多少；
- “Early vs. T8 final”反映完整模型从任务 Tn 训练结束到 T8 结束后的总体遗忘，
  但不能单独定位遗忘发生在哪个模块。

补充统计：

- 7 个任务中 5 个提升、2 个下降；
- 变化量中位数为 **+7.57 pp**；
- 最大提升为 ImageNet 的 **+61.07 pp**；
- 排除 ImageNet 这一极大变化后，其余 6 个任务平均仍提升约 **+4.05 pp**；
- 替换模型在 7/7 个任务上仍低于对应的完整 early checkpoint，平均低
  **16.08 pp**，说明 projector 之外的参数变化或模块协同同样重要；
- Grounding 虽提高 8.47 pp，但绝对准确率仍只有 8.99%，不能视为任务已经恢复。

### 4.2 预测完整性

| 任务 | 预期预测数 | 实际预测数 | 状态 |
| --- | ---: | ---: | --- |
| ScienceQA | 4,241 | 4,241 | 完整 |
| TextVQA | 5,000 | 5,000 | 完整 |
| ImageNet | 5,050 | 5,050 | 完整 |
| GQA | 12,578 | 12,578 | 完整 |
| VizWiz | 4,319 | 4,319 | 完整 |
| Grounding | 30,969 | 30,969 | 完整 |
| VQAv2 | 214,354 | 214,354 | 完整 |

### 4.3 耗时

当前实验从 2026-08-18 02:07:22 到 09:25:32，总墙钟时间为
**7 小时 18 分 10 秒**，其中包含 checkpoint 预验证、任务切换和结果汇总时间。

| 任务 | 当前实验耗时 | 旧 T8 final 回测耗时 |
| --- | ---: | ---: |
| T1 ScienceQA | 00:04:52 | 00:14:13 |
| T2 TextVQA | 00:17:48 | 00:36:05 |
| T3 ImageNet | 00:09:39 | 00:24:14 |
| T4 GQA | 00:17:27 | 00:41:19 |
| T5 VizWiz | 00:09:24 | 00:30:02 |
| T6 Grounding | 01:48:34 | 09:38:36 |
| T7 VQAv2 | 04:29:12 | 11:35:15 |

严格按共同的 T1–T7 范围比较：

| 指标 | 当前实验 | 旧回测 |
| --- | ---: | ---: |
| T1–T7 墙钟时间 | 07:18:10 | 23:39:46 |
| 节省时间 | — | **16:21:36** |
| 加速比 | — | 当前约为旧评估的 **3.24×** |
| 时间降幅 | — | **69.15%** |

旧实验完整 T1–T8 `all` 回测耗时为 **35:27:14**；当前实验没有重复评估 T8，
因此不能把 07:18:10 与该 8 任务总耗时作为严格同范围比较。

## 5. 结果分析

### 5.1 支持的观察

1. **Projector 遗忘在多个任务上是显著因素。** ScienceQA、ImageNet、VizWiz、
   Grounding 和 VQAv2 在恢复对应 early projector 后均提高，说明最终 projector
   不再完整保留这些任务在其训练阶段形成的视觉—语言映射。
2. **ImageNet 对 projector 漂移最敏感。** ImageNet 从 3.80% 恢复到 64.87%，
   表明其分类能力的大幅遗忘与 projector 变化高度相关。
3. **Projector 遗忘不是统一、单调的。** TextVQA 和 GQA 使用 early projector 后
   分别下降 8.76 和 2.44 个百分点，说明最终 projector 对部分任务可能具有更好的
   迁移性，或 early projector 与 T8 adapter/router 存在协同不匹配。
4. **只恢复 projector 不能完全恢复所有任务。** Grounding 虽明显提高，但绝对值
   仍低；而且 7 个替换结果均未达到完整 early checkpoint 的原始准确率。这提示
   遗忘还涉及 adapter/router、语言建模、模块协同或输出格式等其他部分。

### 5.2 当前实验不能直接证明的内容

- 本实验仅测量对角线组合 `projector Tn → task Tn`，没有完整的
  `projector source × eval task` 交叉矩阵，因此不能判断 projector 是任务专用还是
  普遍随时间退化。
- 当前实验把 early projector 与 T8 final adapter/router 组合，结果同时受到
  projector 本身能力和模块间协同匹配程度影响。
- 各任务指标和数据规模不同，7 任务非加权平均仅用于总体概览，不能替代逐任务分析。

## 6. 基线可比性与限制

准确率表中的 Early 完整 checkpoint 和 T8 final 完整 checkpoint 两组参照值来自
2026-05 的原始在线评估，当前 projector 实验使用了修复后的批量评估路径。实验采用
相同 checkpoint 系列、任务数据和 scorer，且 T1–T7 样本数一致，但当前替换实验与
两组历史参照的运行配置并不完全相同：

| 项目 | 2026-05 Early/T8 历史参照 | 当前 projector 实验 |
| --- | --- | --- |
| checkpoint | Tn early 完整 checkpoint 或 T8 final 完整 checkpoint | T8 final adapter + Tn early projector |
| 数据 chunks | 6 | 8 |
| eval batch size | 旧日志未显式记录 | 4 |
| SDPA 批量修复 | 旧评估实现 | 明确启用并通过 BS=1/BS=4 回归 |
| max token 配置 | 旧脚本版本 | 当前任务级固定上限 |

因此，当前结果足以作为“early projector 可能显著缓解部分任务遗忘”的趋势性证据，
但不能把全部准确率差异严格归因于 projector 替换。表中“替换模型 vs. Early”的
差距也同时包含评估实现差异。特别是 ImageNet 的巨大提升需要在完全一致的当前评估
配置下重跑未替换的 T8 final checkpoint 后再确认；若要严格衡量恢复到 early 水平
的程度，还应按当前配置重评 T1–T7 完整 early checkpoint。

建议的严格对照实验是：保持当前 8 GPU、BS=4、SDPA、任务级 max token、数据与
scorer 全部不变，直接用原始 T8 final checkpoint 重评 T1–T7。该结果应作为最终
论文或正式汇报中的主 baseline；本报告中的 2026-05 结果可保留为 legacy baseline。

另需注意：旧 T8 OCRVQA 日志同时出现 `Samples: 0` 和 `Accuracy: 60.31%`，属于旧
scorer 输出格式异常。由于当前协议没有重评 T8，OCRVQA 不纳入本报告的主准确率
比较或均值计算。

## 7. 当前结论

当前实验表明，在 CoIN 正序持续学习中，`mm_projector` 的阶段性变化与早期任务遗忘
存在明显关联。恢复任务训练完成时的 projector 后，5/7 个任务相对 T8 final 得到
改善，其中 ImageNet、VQAv2、Grounding 和 VizWiz 的改善最突出。然而，所有替换
模型仍低于对应的完整 early checkpoint；同时 TextVQA 和 GQA 相对 T8 final 下降。
这说明 projector 漂移可能是部分遗忘的重要来源，但模型能力还受到 projector 与
最终 adapter/router 协同关系以及其他参数遗忘的共同影响。

现阶段建议将结论表述为：

> 初步对角线替换实验显示，持续学习后的 projector 漂移可能是部分早期任务遗忘的
> 重要来源，但不是唯一来源；在 7 个匹配任务中，恢复阶段性 projector 可改善 5 个
> 任务，但均未完全恢复到对应 early 完整 checkpoint 的水平。严格归因仍需在统一
> 评估配置下补跑未经替换的 T8 final baseline。

## 8. 关键文件路径

### 8.1 实验代码

```text
scripts/projector_analysis/prepare_forward_projector_swap.py
scripts/projector_analysis/run_forward_projector_swap_8tasks.sh
scripts/projector_analysis/run_all_forward_projector_swaps.sh
scripts/projector_analysis/summarize_forward_projector_swap.py
scripts/projector_analysis/summarize_forward_projector_matrix.py
scripts/projector_analysis/README.md
```

### 8.2 输入 checkpoint

```text
# T1–T7 early projector 来源
checkpoints/LLaVA/CoIN/ScienceQA_llava_MOE_lora/
checkpoints/LLaVA/CoIN/TextVQA_llava_MOE_lora/
checkpoints/LLaVA/CoIN/ImageNet_llava_MOE_lora/
checkpoints/LLaVA/CoIN/GQA_llava_MOE_lora/
checkpoints/LLaVA/CoIN/VizWiz_llava_MOE_lora/
checkpoints/LLaVA/CoIN/Grounding_llava_MOE_lora/
checkpoints/LLaVA/CoIN/VQAv2_llava_MOE_lora/

# 最终 adapter/router/config 与旧基线来源
checkpoints/LLaVA/CoIN/OCRVQA_llava_MOE_lora/
```

### 8.3 当前实验汇总结果

```text
results/CoIN/LLaVA/metrics/projector_swap/all_early/20260818_020720/
├── combined_metrics.csv
├── experiment_manifest.json
├── projector_eval_diagonal.csv
└── summary.md
```

其中 `combined_metrics.csv` 包含每个任务的 checkpoint、预测结果目录、日志路径、
准确率、预期/实际预测数和完成状态，是最完整的机器可读索引。

### 8.4 当前实验日志

```text
# 总调度日志
logs/LLaVA/projector_swap/all_early/20260818_020720/orchestrator.log

# 各任务日志（路径模式）
logs/LLaVA/projector_swap/
  early_T{n}_{task}__final_T8_OCRVQA/
  20260818_020720/eval_T{n}_{task}.log
```

### 8.5 当前预测与 scorer 输出

预测文件仍保存在各任务原有 result root 下，具体绝对路径已逐行记录在
`combined_metrics.csv` 的 `result_dir` 列中。各结果目录通常包含 8 个分片、
`merge.jsonl` 以及任务 scorer 的输出文件。

### 8.6 旧 T8 final 基线

```text
# 汇总准确率
results/CoIN/LLaVA/metrics/continual_online_eval.csv

# all 模式日志路径模式
logs/LLaVA/CoIN/
  eval_online_trainT8_evalT{1..8}_all_20260515_114414.log

# 旧预测结果目录路径模式
results/CoIN/LLaVA/<task-result-root>/CoIN_online_mall_train8_eval{1..8}/
```

### 8.7 Early 完整 checkpoint 的在线评估结果

Early checkpoint 的准确率同样记录在：

```text
results/CoIN/LLaVA/metrics/continual_online_eval.csv
```

使用其中满足 `phase=online, mode=all, train_task=eval_task=Tn` 的 T1–T7 行。对应日志
路径模式为：

```text
logs/LLaVA/CoIN/eval_online_trainT{n}_evalT{n}_all_<timestamp>.log
```

对应预测结果目录记录在 CSV 的 `result_stage_dir` 列中，目录名模式为：

```text
results/CoIN/LLaVA/<task-result-root>/CoIN_online_mall_train{n}_eval{n}/
```

## 9. 推荐的后续实验

1. **优先补齐同配置 T8 final baseline。** 使用当前 BS=4、8 GPU、SDPA 和任务级
   max token 设置，直接评估未替换的 T8 final checkpoint；这是确认 projector
   因果贡献的必要对照。
2. **计算统一配置下的 retained gain。** 对每个任务报告
   `Acc(early projector + final adapter) - Acc(final projector + final adapter)`。
3. **选择关键任务做交叉 projector 矩阵。** 至少对 ImageNet、TextVQA、GQA 和
   VQAv2 测试多个 projector 来源，区分任务专用 projector 与一般时间漂移。
4. **增加模块消融。** 对比仅替换 projector、仅恢复 task adapter/router、以及二者
   同时恢复，以定位遗忘来源及模块协同效应。
5. **分析 projector 参数漂移。** 计算各阶段 projector 的权重距离、输出特征余弦
   相似度，并与准确率变化关联。
