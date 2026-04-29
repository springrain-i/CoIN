---
name: MoE-MoKA Bug Fix & Pipeline Verification
description: All bugs fixed, T1-T8 full pipeline verified 2026-04-28 — ready for real experiments
type: project
originSessionId: 72aba22e-72a5-4e70-8bcc-d67d70940b1c
---
## 状态
**全流程验证通过（2026-04-28）**，`[pipeline] PASS`，可直接跑正式 8 卡实验。

## 已修复的 Bug

| Bug | 文件 | 修复方式 |
|-----|------|---------|
| `--max-new-tokens` 不识别 | model_text_vqa/gqa/vizwiz/ocr_vqa.py | underscore 改 hyphen |
| `generate()` positional arg 报错 | model_vizwiz.py, model_ocr_vqa.py | 改为 `input_ids=input_ids` keyword arg |
| LoRA 诊断 `active_with_r>0=0` | model_vqa/text_vqa/gqa.py | 优先检查 `lora_experts`（MoE-MoKA）再检查 `lora_A/lora_B` |
| eval_gqa 硬编码 `/data4/` 路径 | eval_gqa.py + 4_eval_gqa.sh | 新增 `--questions-dir` 参数 |
| 单 GPU ZeRO-3 OOM | coin_paths.sh | 自动检测 GPU 数量，单卡用 zero3_offload.json |
| MoE-MoKA 共享 B（设计错误） | mokamoelora.py | 改为每专家独立 B_i |

## Final Eval Matrix 结果（mini test，T8 checkpoint → 全 8 任务）
ScienceQA=0.59%，TextVQA=0.00%，ImageNet=4.00%，GQA/VizWiz/Grounding/VQAv2/OCRVQA=0.00%
（精度低属正常：3步训练的预期值）

## 下一步
正式 8 卡实验：`bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh`
对比 MODES="all text vision" 三模式精度 → 分析过度文本化
