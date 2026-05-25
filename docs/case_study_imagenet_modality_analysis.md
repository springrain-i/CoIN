# 案例分析：ImageNet 评估 —— 模态贡献分析

**分析焦点**：ImageNet 任务（T3）分别在 T3 训练结束后和 T4（GQA）训练结束后进行评估，采用三种模态模式：`all`、`text`、`visual`。

本案例呈现两个独立现象：
1. **过度文本化（Over-textualization）**（T3 后）：text LoRA 单独使用即可完成纯视觉分类任务。
2. **灾难性遗忘（Catastrophic Forgetting）**（T4 后）：GQA 训练彻底抹除 ImageNet 知识，两条模态路径以截然不同的方式退化。

---

## 数据来源

| 数据内容 | 路径 |
|---------|------|
| T3 → ImageNet 评估日志 | `logs/LLaVA/CoIN/20260502_150901/online_train3_ImageNet_eval3_ImageNet_{all,text,visual}.log` |
| T4 → ImageNet 评估日志 | `logs/LLaVA/CoIN/20260507_144530/online_train4_GQA_eval3_ImageNet_{all,text,visual}.log` |
| T3 输出结果文件 | `results/CoIN/LLaVA/MoEMoKA/ImageNet/CoIN_online_m{all,text,visual}_train3_eval3/merge.jsonl` |
| T4 输出结果文件 | `results/CoIN/LLaVA/MoEMoKA/ImageNet/CoIN_online_m{all,text,visual}_train4_eval3/merge.jsonl` |

---

## 第一部分：T3 训练后的过度文本化

### 精度对比

| 模式 | 精度 | 数据来源 |
|------|------|---------|
| all | **93.49%** | `online_train3_ImageNet_eval3_ImageNet_all.log` |
| text | **93.66%** | `online_train3_ImageNet_eval3_ImageNet_text.log` |
| visual | **31.03%** | `online_train3_ImageNet_eval3_ImageNet_visual.log` |

- `text / all` 比值 = **1.00**（text 模式与完整模式效果相当）
- `visual / all` 比值 = **0.33**（visual 模式损失 62.5 个百分点）

### 输出一致性验证

```
text 输出与 all 输出完全一致：4995 / 5050 = 98.9%
visual 输出与 all 输出完全一致：1388 / 5050 = 27.5%
```

*验证方法：对 `merge.jsonl` 文件按 `question_id` 对齐后逐条比较输出字符串*

text 模式与 all 模式对 **98.9%** 的样本生成了完全相同的输出序列。模型的决策几乎完全由 text LoRA 路径决定。

### 样本级输出对比（T3，前 5 条）

| 问题 | all 输出 | text 输出 | visual 输出 |
|------|---------|-----------|------------|
| "图中是什么物体？" | Recreational vehicle | Recreational vehicle | Recreational vehicle |
| 同上 | Crane2 | Crane2 | `.` |
| 同上 | Cabbage butterfly | Cabbage butterfly | `A.` |
| 同上 | Garden spider | Garden spider | `Cater.` |
| 同上 | Pickup | Pickup | `Driving car.` |

*数据来源：`CoIN_online_m{all,text,visual}_train3_eval3/merge.jsonl`，前 5 条记录*

text 模式准确复现了正确类别名。visual 模式则输出截断符号（`.`、`A.`、`Cater.`）或不相关的短语，无法识别正确类别。

### 现象解读

ImageNet 分类要求从照片中识别物体类别，本质是纯视觉任务。但 T3 训练后，模型通过 **text LoRA 路径** 学会了这一任务：

- 任务问题（`"What is the object in the image? Answer with a single word."`）是固定的文本 token 序列。
- text LoRA 编码了"问题模式 → 类别名"的映射关系。
- 关闭 visual LoRA 对精度几乎无影响（93.49% → 93.66%，反而略有提升）。
- 关闭 text LoRA（即 visual 模式）导致精度从 93.49% 崩溃至 31.03%。

**结论**：这是典型的过度文本化——视觉任务通过文本 token 适配完成，visual LoRA 路径对性能几乎没有贡献。

---

## 第二部分：T4 训练后的灾难性遗忘

### 精度对比

| 模式 | 精度 | 数据来源 |
|------|------|---------|
| all | **0.00%** | `online_train4_GQA_eval3_ImageNet_all.log` |
| text | **0.00%** | `online_train4_GQA_eval3_ImageNet_text.log` |
| visual | **0.00%** | `online_train4_GQA_eval3_ImageNet_visual.log` |

经过一轮 GQA 微调后，三种模式全部降至 0%，ImageNet 知识被彻底抹除。

### 输出长度分布

| 模式 | 中位数（词数） | p95 | 数据来源 |
|------|-------------|-----|---------|
| all | 1 | 1 | `CoIN_online_mall_train4_eval3/merge.jsonl` |
| text | 1 | 1 | `CoIN_online_mtext_train4_eval3/merge.jsonl` |
| visual | **8** | **8** | `CoIN_online_mvisual_train4_eval3/merge.jsonl` |

### 样本级输出对比（T4，前 5–8 条）

**all / text 模式**（短词，错误类别名）：
```
all 模式：  Motorcycle / Chair / Motorcycle / Pole / Motorcycle
text 模式： Chair / Chair / Chair / Chair / Computer
```

**visual 模式**（固定句子模板）：
```
The object in the image is a tree.
The object in the image is a book.
The object in the image is a tree.
The object in the image is a tree.
The object in the image is a tree.
```

*数据来源：`CoIN_online_m{all,text,visual}_train4_eval3/merge.jsonl`，前 5–8 条记录*

### 两条路径的退化方式

**text 模式**退化为少数高频短词。"Chair" 在几乎所有样本中重复出现。text LoRA 被 GQA 的答案分布覆盖（GQA 答案通常为短名词或形容词），残留模式收敛到少数几个均值词，不再对应任何具体类别。

**visual 模式**退化为固定描述模板：`"The object in the image is a [泛称名词]."`，输出长度固定为 8 个词。visual LoRA 在 T3 阶段从未真正承担分类功能（如第一部分所示），遗忘后回归到 base model 的通用描述模式，与图像实际内容无关。

**all 模式**行为与 text 模式一致（短词），而非与 visual 模式一致（长句）。这说明即便在遗忘后，text LoRA 仍然主导 all 模式的输出，**过度文本化的结构特征在遗忘后依然保留**。

---

## 综合对比

| 问题 | 证据 |
|------|------|
| T3 训练后 text LoRA 是否主导输出？ | 是。text/all 输出完全一致率 98.9%；精度差仅 0.17 pp。 |
| 视觉任务能否在无 visual LoRA 的情况下完成？ | 是。T3 text 模式在纯视觉分类任务 ImageNet 上达到 93.66%。 |
| 灾难性遗忘后两条路径如何退化？ | text 路径退化为高频短词；visual 路径退化为通用描述模板；方式截然不同。 |
| 过度文本化的结构是否在遗忘后保留？ | 是。T4 的 all 模式退化方向与 text 一致，而非 visual。 |

---

## 对过度文本化假说的意义

T3 ImageNet 结果是本实验中过度文本化最强的单条证据：

- ImageNet 本质是视觉任务（从照片识别类别）。
- MoE-MoKA 的设计目标之一正是为视觉 token 提供独立的 LoRA 路径（`A_vis`），避免文本梯度污染。
- 然而 T3 训练后，text LoRA 单独即可达到 93.66% 精度，与 all 模式输出 98.9% 逐字相同。
- visual LoRA 单独仅达 31.03%。

MoKA 的设计在单任务场景下能够减轻过度文本化，但在持续学习的多任务序列中，文本 token 路径仍积累了足够的梯度信号，使其能够独立编码原本应由视觉路径承担的分类响应。
