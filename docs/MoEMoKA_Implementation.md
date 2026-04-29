# MoE-MoKA 实现说明文档

**日期：** 2026-04-28  
**分支：** `MoEMoKA`  
**状态：** T1-T8 mini pipeline 全流程通过，可直接在 4/8 卡运行正式实验

---

## 1. 设计目标与架构

### 1.1 背景

- **MoKA**（NeurIPS 2025）：用模态专用 A 矩阵解决 over-textualization 问题。单任务设置下每层有 (A_text, A_vis, B)，视觉 token 在 rank-r 空间内做 cross-attention 关注文本 token。
- **CoIN MoE-LoRA**（CoIN 原始）：N 个专家，每专家持有独立 (A_i, B_i)，软路由加权输出，rank 按专家平分（r_per = r//N）。
- **MoE-MoKA（本实现）**：将上述两者结合。每个专家是完整的 MoKA 单元，N=8 个专家各自持有 (A_text_i, A_vis_i, B_i)，专家内部做 cross-attention，软路由加权各专家输出之和。

### 1.2 核心公式

每个替换后的 `nn.Linear` 层，前向传播为：

```
ΔW · x = Σᵢ  wᵢ · Bᵢ · [A_text_i · x_text  ;  A_vis_i · x_vis + CrossAttn_i(vis, text)]

其中：
  x_text = x[token_mask == 2]         # 文本 token
  x_vis  = x[token_mask == 1]         # 视觉 token
  wᵢ     = softmax(W_router · x)[i]  # 每个 token 的专家路由权重
  CrossAttn_i(vis, text):
    Q = A_vis_i · x_vis    (n_vis, r_per)
    K = V = A_text_i · x_text  (n_text, r_per)
    output = softmax(Q·Kᵀ / √r_per) · V + Q  (残差连接)

最终: result = W₀·x + ΔW·x · (lora_alpha / r)
```

### 1.3 参数规模（对比标准 LoRA）

| 配置 | 新增参数 |
|------|---------|
| 标准 LoRA (r=32) | d_in×r + r×d_out |
| MoE-MoKA (r=32, N=8) | N×(2×d_in×r_per + r_per×d_out) + d_in×N (路由器) |
| r_per = r//N = 4 | 与标准 LoRA 参数量相近（总 rank budget 相同） |

---

## 2. 文件地图

| 文件 | 作用 |
|------|------|
| `CoIN/peft/tuners/mokamoelora.py` | **核心实现**：MoEMoKALoraConfig、MoEMoKAExpert、MoEMoKALoraLinear、MoEMoKALoraModel |
| `CoIN/peft/mapping.py` | 注册 peft_type → 模型类的映射 |
| `ETrain/Train/LLaVA/train.py` | 训练入口：`ModelArguments.moe_moka_enable`、`expert_num` |
| `ETrain/Train/LLaVA/llava_trainer.py` | `save_trained_model`（torch.save 路径）、`load_model_from_previous_task` |
| `ETrain/Models/LLaVA/builder.py` | eval 时加载 checkpoint（通过 adapter_config.json 检测类型） |
| `ETrain/Models/LLaVA/llava_arch.py` | 设置 `current_lora_mask`（2=text, 1=vis, 0=pad） |
| `ETrain/Models/LLaVA/language_model/llava_llama.py` | `_apply_lora_token_mask()` 将 mask 传给每层 |

---

## 3. 核心实现：`mokamoelora.py`

### 3.1 配置

```python
@dataclass
class MoEMoKALoraConfig(LoraConfig):
    expert_num: int = field(default=8)   # 与 CoIN MoE-LoRA 对齐
    def __post_init__(self):
        self.peft_type = PeftType.MOE_MOKA_CoIN
        # adapter_config.json 中 task_type = "CAUSAL_LM_MoEMoKA"
```

**重要约束：** `r % expert_num == 0`（否则 rank 无法平分）。默认配置 r=32, N=8 → r_per=4。

### 3.2 专家单元

```python
class MoEMoKAExpert(nn.Module):
    def __init__(self, in_features, out_features, r_per):
        self.lora_A_text = nn.Linear(in_features, r_per, bias=False)  # d_in → r_per
        self.lora_A_vis  = nn.Linear(in_features, r_per, bias=False)  # d_in → r_per
        self.lora_B      = nn.Linear(r_per, out_features, bias=False) # r_per → d_out
```

每个专家有**独立的 B 矩阵**（与 CoIN MoE-LoRA 设计对齐，不同于原 MoKA 的共享 B）。

### 3.3 层存储结构

```python
class MoEMoKALoraLayer:
    lora_experts: ModuleDict  # adapter_name → ModuleList[MoEMoKAExpert × N]
    lora_router:  ModuleDict  # adapter_name → nn.Linear(d_in, N)
```

### 3.4 前向传播（关键逻辑）

```python
def forward(self, x):
    result = F.linear(x, W0, bias)          # 基础线性层
    
    # 解析 token 模态
    text_mask = (token_mask == 2)           # (B, S)
    vis_mask  = (token_mask == 1)
    
    # 路由权重
    router_weights = softmax(lora_router(x), dim=-1)  # (B, S, N)
    
    # 提前提取，避免每个专家重复 indexing
    flat_text = x[text_mask]   # (n_text, d_in)
    flat_vis  = x[vis_mask]    # (n_vis, d_in)
    
    expert_sum = zeros(B, S, d_out)
    for i, expert in enumerate(experts):
        r_per = expert.lora_A_text.out_features
        
        # 无条件调用 A 矩阵（ZeRO-3 trace 一致性要求）
        out_text = expert.lora_A_text(flat_text)  # (n_text, r_per)
        out_vis  = expert.lora_A_vis(flat_vis)    # (n_vis,  r_per)
        
        # 重建全序列张量
        a_comb = zeros(B, S, r_per)
        if has_text: a_comb[text_mask] = out_text
        if has_vis:
            if has_text:
                out_vis = cross_attention(query=out_vis, key_val=out_text)
            a_comb[vis_mask] = out_vis
        
        # 每专家独立 B 投影
        out_i = expert.lora_B(a_comb)        # (B, S, d_out)
        expert_sum += router_weights[:,:,i].unsqueeze(-1) * out_i
    
    return result + expert_sum * scaling
```

### 3.5 专家内 Cross-Attention

```python
@staticmethod
def _cross_attention(query, key_val, text_mask, vis_mask, batch_size):
    """
    query  : (n_vis,  r_per) — A_vis 输出
    key_val: (n_text, r_per) — A_text 输出（同时作 K 和 V）
    返回更新后的 visual 表示（残差连接）
    """
    for b in range(batch_size):
        q = query[vis_slice_b]   # (1, n_vis,  r_per)
        k = key_val[text_slice_b]# (1, n_text, r_per)
        score     = q @ k.T / sqrt(r_per)
        attn_probs = softmax(score, dim=-1)
        attn_out  = attn_probs @ k          # (n_vis, r_per)
        out[vis_slice_b] = query[vis_slice_b] + attn_out   # 残差
    return out
```

无额外投影矩阵 — 直接使用 A 矩阵输出作为 Q/K/V（与 MoKA 论文一致）。

---

## 4. Token Mask 基础设施

```
设置位置: ETrain/Models/LLaVA/llava_arch.py → prepare_inputs_labels_for_multimodal()
传播路径: llava_llama.py → _apply_lora_token_mask() → 每个 LoRA 模块
取值含义: 2 = 文本 token, 1 = 视觉 token, 0 = padding（始终忽略）
```

**推理时 lora_mode 支持：**

| `--lora-mode` | 效果 |
|---------------|------|
| `all`（默认）| 文本 + 视觉 token 均经过 LoRA |
| `text` | 视觉 token 不经过 LoRA（测量 text bias） |
| `vision` | 文本 token 不经过 LoRA（测量 visual 独立性） |

---

## 5. 保存与加载

### 5.1 为何绕过标准 PEFT

`CoIN/peft/utils/save_and_load.py` 的 `get_peft_model_state_dict()` 有类型白名单，不含 `MOE_MOKA_CoIN`，调用时抛 `NotImplementedError`。因此 MoE-MoKA 全程使用 `torch.save` / `model.load_state_dict(strict=False)`。

### 5.2 每任务保存的文件

```
<output_dir>/
├── adapter_model.bin          # 所有 lora_ 前缀参数（含 experts、router）
├── adapter_config.json        # peft_type=MOE_MOKA_CoIN, expert_num=8, r=32, ...
├── non_lora_trainables.bin    # mm_projector + embed_tokens
└── config.json                # LLaVA 模型结构配置
```

### 5.3 训练保存（llava_trainer.py）

```python
if getattr(training_args, 'moe_moka_enable', False):
    state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), bias)
    torch.save(state_dict, output_dir / WEIGHTS_NAME)         # adapter_model.bin
    model.peft_config['default'].save_pretrained(output_dir)  # adapter_config.json
    torch.save(non_lora_dict, output_dir / 'non_lora_trainables.bin')
```

### 5.4 任务间加载（llava_trainer.py）

```python
if moe_moka_enable:
    weights = torch.load(prev_path / WEIGHTS_NAME, map_location='cpu')
    model.load_state_dict(weights, strict=False)   # key 含 ".default."，直接匹配
```

### 5.5 评测加载（builder.py）

```python
# 通过读 adapter_config.json 判断类型（不依赖目录名）
_peft_type = json.load(open(model_path / 'adapter_config.json')).get('peft_type', '')
is_moe_moka = (_peft_type == 'MOE_MOKA_CoIN')

if is_moe_moka:
    adapter_config = MoEMoKALoraConfig.from_pretrained(model_path)
    model = get_peft_model(model, adapter_config)
    weights = torch.load(model_path / WEIGHTS_NAME, map_location='cpu')
    model.load_state_dict(weights, strict=False)
```

**注意：** MoE-MoKA 不支持权重合并（merge_and_unload），始终以 adapter 形式推理。

---

## 6. 训练配置

### 6.1 关键超参数

```bash
--moe_moka_enable True  # 启用 MoE-MoKA
--lora_r 32             # 总 rank，r_per = 32 // 8 = 4
--lora_alpha 64         # scaling = alpha / r = 2.0
--expert_num 8          # 专家数量（与 CoIN MoE-LoRA 对齐）
```

### 6.2 DeepSpeed 配置自动选择

`scripts/LLaVA/Train_MoEMoKA/coin_paths.sh` 根据可见 GPU 数量自动选择：
- **≥4 卡**：`scripts/zero3.json`（无 CPU offload，适合多卡）
- **1-3 卡**：`scripts/zero3_offload.json`（CPU offload，适合单卡调试）
- 可用 `COIN_DS_CONFIG` 环境变量强制覆盖

---

## 7. ZeRO-3 兼容性

**问题：** 条件分支调用（`if has_text: A_text(x)`）在不同 rank 上执行路径不同，导致 ZeRO-3 参数预取 trace 不一致，引发 NCCL hang。

**修复：** 两个 A 矩阵始终无条件调用，结果再按 mask 写入：

```python
out_text = expert.lora_A_text(flat_text)  # 始终执行
out_vis  = expert.lora_A_vis(flat_vis)    # 始终执行
if has_text: a_comb[text_mask] = out_text
if has_vis:  a_comb[vis_mask]  = out_vis
```

---

## 8. 评测脚本

### 8.1 脚本列表

| 脚本 | 对应 Python 模块 | 支持参数 |
|------|----------------|---------|
| `eval/1_eval_sqa.sh` | `ETrain.Eval.LLaVA.CoIN.model_vqa_science` | `--max-new-tokens` |
| `eval/2_eval_textqa.sh` | `ETrain.Eval.LLaVA.CoIN.model_text_vqa` | `--max-new-tokens` |
| `eval/3_eval_imagenet.sh` | `ETrain.Eval.LLaVA.CoIN.model_vqa` | `--max_new_tokens` |
| `eval/4_eval_gqa.sh` | `ETrain.Eval.LLaVA.CoIN.model_gqa` | `--max-new-tokens` |
| `eval/5_eval_vizwiz.sh` | `ETrain.Eval.LLaVA.CoIN.model_vizwiz` | `--max-new-tokens` |
| `eval/6_eval_grounding.sh` | `ETrain.Eval.LLaVA.CoIN.model_vqa` | `--max_new_tokens` |
| `eval/7_eval_vqav2.sh` | `ETrain.Eval.LLaVA.CoIN.model_vqa` | `--max_new_tokens` |
| `eval/8_eval_ocrvqa.sh` | `ETrain.Eval.LLaVA.CoIN.model_ocr_vqa` | `--max-new-tokens` |

所有脚本均通过 `MAX_NEW_TOKENS` 环境变量控制生成长度（默认 1024）。

### 8.2 评测运行方式

```bash
# 单任务评测（从仓库根目录）
bash scripts/LLaVA/Train_MoEMoKA/eval/1_eval_sqa.sh \
    <STAGE名> \
    <checkpoint路径> \
    <lora_mode: all|text|vision>

# 快速验证（短生成）
MAX_NEW_TOKENS=32 bash scripts/LLaVA/Train_MoEMoKA/eval/1_eval_sqa.sh \
    debug_run \
    checkpoints/LLaVA/CoIN/ScienceQA_llava_MoEMoKA_lora \
    all
```

### 8.3 结果位置

```
results/CoIN/LLaVA/MoEMoKA/<DATASET>/<STAGE>/
├── merge.jsonl          # 所有预测结果合并
├── output.jsonl         # 评测中间文件
└── output_result.jsonl  # 最终准确率（含 acc 字段）
```

---

## 9. 持续学习流程

### 9.1 Checkpoint 流转

```
T1 ScienceQA:  基础模型 → 训练 → ScienceQA_llava_MoEMoKA_lora/
T2 TextVQA:    T1 权重 → 加载 → 训练 → TextVQA_llava_MoEMoKA_lora/
T3 ImageNet:   T2 权重 → 加载 → 训练 → ImageNet_llava_MoEMoKA_lora/
T4 GQA:        T3 权重 → 加载 → 训练 → GQA_llava_MoEMoKA_lora/
T5 VizWiz:     T4 权重 → 加载 → 训练 → VizWiz_llava_MoEMoKA_lora/
T6 Grounding:  T5 权重 → 加载 → 训练 → Grounding_llava_MoEMoKA_lora/
T7 VQAv2:      T6 权重 → 加载 → 训练 → VQAv2_llava_MoEMoKA_lora/
T8 OCRVQA:     T7 权重 → 加载 → 训练 → OCRVQA_llava_MoEMoKA_lora/
```

### 9.2 运行命令

```bash
# 完整持续学习（自动检测 GPU，4/8 卡用 zero3.json）
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh

# Mini 冒烟测试（每任务3步，验证流程正确）
bash scripts/LLaVA/Train_MoEMoKA/mini_pipeline_test.sh
# 已验证：2026-04-28，T1-T8 全部 PASS，exit code 0
```

---

## 10. 与 MoKA 论文的差异

| 项目 | MoKA 论文 | 本实现 | 原因 |
|------|----------|--------|------|
| 专家数量 | 1（单任务） | 8（MoE 扩展） | MoE-MoKA 设计目标 |
| 每专家 B | 共享 B | **独立 B**（per-expert） | 与 CoIN MoE-LoRA 对齐 |
| Rank per expert | r（全量） | r//N = 4 | 控制参数规模 |
| Cross-attn 缩放 | `√N_t`（文本token数） | `√r_per` | N_t 随样本变化不稳定 |
| 持续学习 | 不涉及 | T1→T8 链式加载 | CoIN 框架需求 |

---

## 11. 常见问题

**Q: 为什么 T2-T8 的 train_loss=0.0（mini 测试）？**  
A: `model_max_length=512` 截断长序列，图像 token（576个）已超过 512，导致回答部分的标签全部被 mask 掉（-100），loss 计算分母为 0。正式训练设置 `model_max_length=2048` 不会出现此问题。

**Q: LoRA module check 显示 active_with_r>0=0 怎么办？**  
A: 已修复。`model_vqa_science.py` 的诊断检测现在先查 `lora_experts`（MoE-MoKA），再查 `lora_A/lora_B`（标准 LoRA）。

**Q: 如何在单卡上跑 eval？**  
A: `eval_common.sh` 自动检测 CUDA_VISIBLE_DEVICES，单卡时 CHUNKS=1，无需修改任何脚本。
