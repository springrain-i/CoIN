---
name: CoIN Research Project — MoE-MoKA
description: MoE-MoKA on CoIN 8-task continual learning — architecture implemented, full pipeline verified 2026-04-28
type: project
originSessionId: 72aba22e-72a5-4e70-8bcc-d67d70940b1c
---
## 仓库与分支
- Repo: `/root/CoIN`，Branch: `MoEMoKA`
- 详细存档：`/root/CoIN/docs/project_archive_MoEMoKA.md`

## 研究目标
三阶段：MoKA on CoIN（已跳过） → MoE-MoKA（✅ 完成） → 过度文本化分析（⏳ 待正式实验）

**核心假设**：持续学习场景下，即使用 MoKA，文本偏差仍会随任务序列加深。

## MoE-MoKA 架构要点
- 每专家独立 `(A_text_i, A_vis_i, B_i)`，expert_num=8，r=32，r_per=4
- Visual tokens 做 cross-attn（attend text tokens），only-text/only-vision 时自动跳过
- lora_mode = all / text / vision，三模式完整支持

## 全流程验证状态（2026-04-28）
- T1-T8 训练：全部 PASS
- Final eval matrix（T8 checkpoint → 全 8 任务）：全部 PASS，`[pipeline] PASS`
- 精度极低正常（3步 mini 训练）

## 正式实验入口
```bash
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh
```

## 关键路径
- Base model: `/hy-tmp/Vicuna/vicuna-7b-v1.5`
- Vision tower: `/hy-tmp/clip-vit-large-patch14-336`
- Data: `/hy-tmp/playground/Instructions_Original/`
- Checkpoints: `/root/CoIN/checkpoints/LLaVA/CoIN/{TaskName}_llava_MoEMoKA_lora/`

## Why
- expert_num=8 对标 CoIN MoE-LoRA 原始设计
- `coin_paths.sh` 自动 GPU 检测：单卡用 zero3_offload.json，多卡用 zero3.json
