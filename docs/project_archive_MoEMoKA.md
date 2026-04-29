# MoE-MoKA on CoIN — 项目存档

> 存档时间：2026-04-28  
> 仓库：`/root/CoIN`，分支：`MoEMoKA`  
> 实验服务器：`/hy-tmp/` 存放所有大文件/模型

---

## 一、研究背景与目标

### 问题
标准 LoRA 使用单一共享 A 矩阵处理所有 token。由于文本 token 数量更多、梯度信号更强，A 矩阵会被过度优化为文本方向——这就是"**过度文本化（over-textualization）**"现象。表现为：partial modality 推理时，only-text 精度接近 all，但 only-vision 精度显著下降。

### MoKA 的解法（NeurIPS 2025）
将单一 A 矩阵替换为**模态专属双路 A**：
- `A_text`：只处理文本 token
- `A_visual`：只处理视觉 token，并通过 cross-attention 吸收文本语义
- 共享 `B`：将低秩表示投影回输出空间

### 本项目的研究假设
MoKA 在单任务场景下修复了过度文本化。**假设：在 8 任务持续学习场景中，即使用 MoKA，文本偏差仍会随任务序列加深。**

MoE-MoKA 是拟提出的解决方案：为每个专家配置独立的 `(A_text_i, A_visual_i, B_i)`，加入软路由机制。

### 三阶段计划

| 阶段 | 目标 | 状态 |
|------|------|------|
| 1. MoKA on CoIN | MoKA 在 8 任务持续学习验证 | ⚠️ 已放弃（直接跳到 MoE-MoKA） |
| 2. MoE-MoKA | 实现并验证 MoE-MoKA 全流程 | ✅ 完成 |
| 3. 过度文本化分析 | all/text/vision 三模式对比实验 | ⏳ 待正式实验 |

---

## 二、MoE-MoKA 架构设计

### 核心文件
`CoIN/peft/tuners/mokamoelora.py`

### 设计规格

| 参数 | 值 | 说明 |
|------|-----|------|
| `expert_num` | 8 | 与 CoIN MoE-LoRA 对齐 |
| `r` | 32 | 总 rank |
| `r_per` | 4 | 每专家 rank（r // expert_num） |
| `lora_alpha` | 64 | scaling = alpha/r = 2.0 |

### 每个专家结构
```
expert_i:
  lora_A_text_i : (d_in → r_per)   仅处理文本 token
  lora_A_vis_i  : (d_in → r_per)   仅处理视觉 token
  lora_B_i      : (r_per → d_out)  投影回输出空间
```

### Forward 流程
```
x → W₀·x（基础线性）
  → router(x) → softmax → weights (B, S, N)
  → 对每个专家 i：
      flat_text → A_text_i → out_text
      flat_vis  → A_vis_i  → out_vis
      if has_text and has_vis:
          out_vis += cross_attn(query=out_vis, key=out_text, val=out_text)
      a_comb = [out_text at text positions ; out_vis at vis positions]
      out_i = B_i · a_comb
  → expert_sum = Σ weights_i · out_i
  → result = W₀·x + expert_sum × scaling
```

### Modality Masking（lora_mode）
CoIN token 约定：`2=文本, 1=视觉, 0=padding`

| lora_mode | 效果 | cross-attn |
|-----------|------|-----------|
| `all` | 文本+视觉都走 LoRA | ✅ 执行 |
| `text` | 仅文本走 LoRA，vis_mask 清零 | ❌ 跳过 |
| `vision` | 仅视觉走 LoRA，text_mask 清零 | ❌ 跳过 |

cross-attn 由 `if has_vis: if has_text:` 双重门控，only-text/only-vision 时自动跳过。

---

## 三、代码架构

### 新增文件

| 文件 | 说明 |
|------|------|
| `CoIN/peft/tuners/mokamoelora.py` | MoE-MoKA adapter 实现 |
| `CoIN/peft/utils/config.py` | 新增 `PeftType.MOE_MOKA_CoIN`、`TaskType.CAUSAL_LM_MoEMoKA` |
| `CoIN/peft/mapping.py` | 注册 MoE-MoKA 到 PEFT mapping |
| `scripts/LLaVA/Train_MoEMoKA/[1-8]_*.sh` | 各任务训练脚本 |
| `scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh` | 持续学习全流程 orchestrator |
| `scripts/LLaVA/Train_MoEMoKA/eval/[1-8]_eval_*.sh` | 各任务 eval 脚本 |
| `scripts/LLaVA/Train_MoEMoKA/coin_paths.sh` | 路径/环境变量统一配置 |
| `scripts/LLaVA/Train_MoEMoKA/full_pipeline_test.sh` | mini 测试脚本（3步训练+50问eval）|

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `ETrain/Eval/LLaVA/CoIN/model_gqa.py` | `--max_new_tokens` → `--max-new-tokens`；LoRA 诊断增加 `lora_experts` 检测 |
| `ETrain/Eval/LLaVA/CoIN/model_text_vqa.py` | 同上 |
| `ETrain/Eval/LLaVA/CoIN/model_vqa.py` | LoRA 诊断增加 `lora_experts` 检测 |
| `ETrain/Eval/LLaVA/CoIN/model_vizwiz.py` | `--max_new_tokens` → `--max-new-tokens`；`model.generate(input_ids, ...)` → `model.generate(input_ids=input_ids, ...)` |
| `ETrain/Eval/LLaVA/CoIN/model_ocr_vqa.py` | 同 model_vizwiz.py |
| `ETrain/Eval/LLaVA/CoIN/eval_gqa.py` | 新增 `--questions-dir` 参数，替换硬编码 `/data4/...` 路径 |
| `ETrain/Models/LLaVA/builder.py` | 通过 `adapter_config.json` 的 `peft_type` 判断是否加载 MoE-MoKA |

---

## 四、Checkpoint 存储逻辑

链式结构，每个任务独立保存：

```
base_model（Vicuna-7B）
  └→ T1 训练 → ScienceQA_llava_MoEMoKA_lora/
       └→ T2 训练 → TextVQA_llava_MoEMoKA_lora/
            └→ T3 训练 → ImageNet_llava_MoEMoKA_lora/
                 ...
                    └→ T8 训练 → OCRVQA_llava_MoEMoKA_lora/
```

- `--save_strategy "epoch"`：每任务 1 epoch，结束时保存
- 8 个 checkpoint 目录独立保存，不覆盖
- `--previous_task_model_path`：T(n) 脚本指向 T(n-1) 的 checkpoint

### Checkpoint 目录位置
`/root/CoIN/checkpoints/LLaVA/CoIN/{TaskName}_llava_MoEMoKA_lora/`

---

## 五、Bug 修复记录

| Bug | 原因 | 修复 |
|-----|------|------|
| `--max-new-tokens` 无法识别 | eval scripts 传 hyphen，Python 定义 underscore，argparse 不自动转换 | 统一改为 hyphen |
| `PeftModelForCausalLM.generate()` 报错 | `model.generate(input_ids, ...)` 以 positional arg 传入，PEFT 只接受 keyword | 改为 `input_ids=input_ids` |
| LoRA 诊断 `active_with_r>0=0` | 诊断只检查 `lora_A/lora_B`，MoE-MoKA 用 `lora_experts` | 诊断逻辑优先检查 `lora_experts` |
| eval_gqa 硬编码 `/data4/...` 路径 | 原始路径在本机不存在 | 新增 `--questions-dir` 参数，eval 脚本传 `${IMAGE_ROOT}/GQA` |
| 单 GPU 下 ZeRO-3 无 offload OOM | 默认用 `zero3.json` 不支持单 GPU CPU offload | `coin_paths.sh` 自动检测 GPU 数量，单卡用 `zero3_offload.json` |

---

## 六、全流程验证结果（2026-04-28）

### 测试配置
- 训练：每任务 3 steps（`MAX_STEPS=3`）
- Eval：每任务 50 问（动态生成 mini JSON）
- 目的：验证流程疏通，非正式精度

### 训练阶段
| 任务 | 结果 |
|------|------|
| T1 ScienceQA | ✅ PASS |
| T2 TextVQA | ✅ PASS |
| T3 ImageNet | ✅ PASS |
| T4 GQA | ✅ PASS |
| T5 VizWiz | ✅ PASS |
| T6 Grounding | ✅ PASS |
| T7 VQAv2 | ✅ PASS |
| T8 OCRVQA | ✅ PASS |

### Online Eval（训练各任务后立即 eval 该任务，mode=all）
| 任务 | 结果 | 备注 |
|------|------|------|
| T1 ScienceQA | ✅ PASS 0.59% | |
| T2 TextVQA | ⚠️ WARN | 运行时 bug 修复前跑，修复后 final eval 正常 |
| T3 ImageNet | ✅ PASS 4.00% | |
| T4 GQA | ✅ PASS 0.00% | |
| T5 VizWiz | ⚠️ WARN | `generate()` positional arg bug 修复前跑 |
| T6 Grounding | ✅ PASS 0.00% | |
| T7 VQAv2 | ✅ PASS 0.00% | |
| T8 OCRVQA | ✅ PASS 0.00% | |

### Final Eval Matrix（T8 checkpoint → 全 8 任务）
| 任务 | Accuracy |
|------|---------|
| T1 ScienceQA | 0.59% |
| T2 TextVQA | 0.00% |
| T3 ImageNet | 4.00% |
| T4 GQA | 0.00% |
| T5 VizWiz | 0.00% |
| T6 Grounding | 0.00% |
| T7 VQAv2 | 0.00% |
| T8 OCRVQA | 0.00% |

> 精度极低属正常：3步训练不足以学习任何任务，验证目的是流程疏通，非精度。  
> `[pipeline] PASS` — 全流程正常退出。

---

## 七、正式实验入口

```bash
# 在 tmux copilot 会话中执行
cd /root/CoIN
source scripts/LLaVA/Train_MoEMoKA/coin_paths.sh

# 持续学习全流程（T1→T8）
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh

# 指定范围（如从 T3 开始）
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh 3 8

# 强制指定 DeepSpeed 配置
COIN_DS_CONFIG=./scripts/zero3.json bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh
```

### 关键路径（`coin_paths.sh` 管理）
| 变量 | 路径 |
|------|------|
| `COIN_BASE_MODEL` | `/hy-tmp/Vicuna/vicuna-7b-v1.5` |
| `COIN_VISION_TOWER` | `/hy-tmp/clip-vit-large-patch14-336` |
| `COIN_INSTR_ROOT` | `/hy-tmp/playground/Instructions_Original` |
| `COIN_IMAGE_ROOT` | `/hy-tmp` |
| `COIN_OUTPUT_ROOT` | `/root/CoIN/checkpoints/LLaVA/CoIN` |

---

## 八、结果输出位置

| 内容 | 路径 |
|------|------|
| 各任务训练 log | `logs/LLaVA/CoIN/{TaskName}_{timestamp}.log` |
| Online eval 精度 CSV | `results/CoIN/LLaVA/metrics/continual_online_eval.csv` |
| Final eval 精度 CSV | `results/CoIN/LLaVA/metrics/continual_final_eval.csv` |
| Eval 原始输出 | `results/CoIN/LLaVA/MoEMoKA/{TaskName}/{stage}/` |

---

## 九、下一步

1. **正式 8 卡持续学习实验**：全量数据，`model_max_length=2048`，`num_train_epochs=1`
2. **三模式 eval 对比**：`MODES="all text vision"` 已内置于 `run_coin_sequence.sh`，自动输出三模式精度到 CSV
3. **过度文本化分析**：从 CSV 对比每个任务下 text-only vs vision-only vs all 精度，验证研究假设
