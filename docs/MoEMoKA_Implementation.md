# MoE-MoKA 实现说明文档

**用途：** 供人工审查的 MoE-MoKA（混合专家多模态 LoRA 适配）在 CoIN 持续学习框架中的完整实现参考。

---

## 1. MoE-MoKA 是什么

MoE-MoKA 在 MoKA（NeurIPS 2025）基础上引入了混合专家路由层。每个专家持有模态专用的 A 矩阵对 `(A_text_i, A_vis_i)`，i=1..N。软路由器（softmax）对专家输出加权融合后，再与共享 B 矩阵相乘。

**每个 LoRA 线性层的核心前向公式：**

```
ΔW·x = B · Σ_i  w_i · [ A_text_i · x_text  ;  A_vis_i · x_vis + CrossAttn(A_vis_i·x_vis, A_text_i·x_text) ]
```

其中：
- `x_text`、`x_vis` = 由 `token_mask` 切分的 token（2=文本，1=视觉，0=padding）
- `w_i` = 专家 i 的路由 softmax 权重（路由器输入为所有 token A 输出的均值）
- 交叉注意力：视觉 token（Q）在 rank-r 空间内关注文本 token（K, V），由可学习标量 `lora_attn` 缩放
- B 矩阵在所有专家间共享

---

## 2. 文件地图

| 文件 | 作用 |
|------|------|
| `CoIN/peft/tuners/mokamoelora.py` | **核心 MoE-MoKA 层** — 配置、线性模块、前向计算 |
| `CoIN/peft/mapping.py` | 注册 `MoEMoKALoraConfig` 和 `PeftModelForCausalLMLORAMOE` |
| `CoIN/peft/utils/save_and_load.py` | PEFT 存取工具 — MoE-MoKA **绕过**此处（见第5节） |
| `ETrain/Train/LLaVA/train.py` | 训练入口；`ModelArguments.moe_moka_enable` 标志 |
| `ETrain/Train/LLaVA/llava_trainer.py` | `save_trained_model` + `load_model_from_previous_task` |
| `ETrain/Models/LLaVA/builder.py` | 评测时模型加载（`load_pretrained_model`） |
| `ETrain/Models/LLaVA/llava_arch.py` | 在模型上设置 `current_lora_mask`（token_mask） |
| `ETrain/Models/LLaVA/language_model/llava_llama.py` | `_apply_lora_token_mask()` 将 mask 传播到每一层 |

---

## 3. 核心模块：`mokamoelora.py`

**路径：** `CoIN/peft/tuners/mokamoelora.py`

### 3.1 配置 — `MoEMoKALoraConfig`

```python
@dataclass
class MoEMoKALoraConfig(LoraConfig):
    peft_type: PeftType = field(default=PeftType.MOE_MOKA_CoIN)
    expert_num: int = field(default=4)
```

继承所有标准 LoRA 超参数（`r`、`lora_alpha`、`target_modules` 等）。

### 3.2 层 — `MoEMoKALoraLinear`

**关键属性：**

| 属性 | Shape | 说明 |
|------|-------|------|
| `lora_A_text[adapter]` | N 个 `nn.Linear(in, r, bias=False)` 的 ModuleList | 各专家的文本 A 矩阵 |
| `lora_A_vis[adapter]` | N 个 `nn.Linear(in, r, bias=False)` 的 ModuleList | 各专家的视觉 A 矩阵 |
| `lora_B[adapter]` | `nn.Linear(r, out, bias=False)` | 共享 B 矩阵 |
| `lora_router[adapter]` | `nn.Linear(r, N, bias=False)` | 软路由器 |
| `lora_attn[adapter]` | `nn.ParameterDict` → 标量 | 交叉注意力门控 |

**初始化：** A 矩阵使用 Kaiming uniform，B 初始化为零（确保初始 ΔW=0，与标准 LoRA 一致）。

### 3.3 前向传播

**文件：** `mokamoelora.py` → `MoEMoKALoraLinear.forward()`

逐步说明：

1. **Token mask 裁剪**（约第180-210行）：若 `token_mask.shape[1] != S`（KV-cache 生成时），将 mask 裁剪为 `token_mask[:, -S:]`；仍不匹配则回退到全文本 mask。
2. **无条件 A 矩阵调用**（约第220-250行）：每个专家的 A_text 和 A_vis 始终被调用（即使 `has_text=False` 或 `has_vis=False`），结果再按条件写入 `a_out`。这是 ZeRO-3 trace 一致性的必要条件。
3. **交叉注意力**（约第255-275行）：视觉 A 输出（Q）通过 `_cross_attention()` 关注文本 A 输出（K, V），结果由 `lora_attn` 标量缩放。
4. **路由器**（约第280-300行）：对 `a_out` 均值池化，经 `lora_router` + softmax 得权重 `w`。
5. **专家融合**（约第300-320行）：`Σ_i w_i * per_expert_a_out_i`，再经 `lora_B` 输出。
6. **残差相加**（约第320-330行）：将 LoRA 增量加到基础线性输出，缩放系数为 `lora_alpha / r`。

### 3.4 交叉注意力 — `_cross_attention()`

```python
def _cross_attention(self, q, k):
    # q: (n_vis, r), k: (n_text, r)
    scale = math.sqrt(k.shape[-1])
    scores = (q @ k.T) / scale          # (n_vis, n_text)
    weights = torch.softmax(scores, -1)
    return weights @ k                  # (n_vis, r)
```

无额外学习投影 — A 矩阵直接充当 Q/K/V 投影器（忠实于 MoKA 论文设计）。

---

## 4. Token Mask 基础设施

**设置于：** `ETrain/Models/LLaVA/llava_arch.py` → `prepare_inputs_labels_for_multimodal()`

```
token_mask 取值：  2 = 文本 token，  1 = 视觉 token，  0 = padding
```

**传播路径：** `ETrain/Models/LLaVA/language_model/llava_llama.py` → `_apply_lora_token_mask()`
每次前向传播调用一次，将 `token_mask` 属性设置到每个 LoRA 模块上。

**使用位置：** `MoEMoKALoraLinear.forward()` 中，用于将 token 路由到对应的 A 矩阵。

---

## 5. 保存与加载

### 5.1 为何绕过标准 PEFT 存取

`CoIN/peft/utils/save_and_load.py`：
- `get_peft_model_state_dict()` 有类型白名单，**不含** `MOE_MOKA_CoIN` → 抛出 `NotImplementedError`。
- `set_peft_model_state_dict()` 同样问题，且对嵌套 `lora_` key 的重写逻辑会出错。

**决策：** MoE-MoKA 完全绕过上述两个函数，改用 `torch.save` / `model.load_state_dict(strict=False)`。

### 5.2 训练保存 — `save_trained_model`

**文件：** `ETrain/Train/LLaVA/llava_trainer.py` 约第444-470行

```python
if getattr(training_args, 'moe_moka_enable', False):
    # 通过 get_peft_state_maybe_zero_3 收集 ZeRO-3 分片
    state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), training_args.lora_bias)
    if training_args.local_rank in (0, -1):
        self.model.config.save_pretrained(training_args.output_dir)
        torch.save(state_dict, os.path.join(training_args.output_dir, WEIGHTS_NAME))  # adapter_model.bin
        self.model.peft_config['default'].save_pretrained(training_args.output_dir)   # adapter_config.json
        torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
```

**每个任务保存的文件：**
```
<output_dir>/
├── adapter_model.bin          # MoE-MoKA 权重（所有 lora_ 参数）
├── adapter_config.json        # MoEMoKALoraConfig（由 from_pretrained 重建）
├── non_lora_trainables.bin    # mm_projector + embed_tokens
└── config.json                # LLaVA 模型配置
```

### 5.3 训练加载 — `load_model_from_previous_task`

**文件：** `ETrain/Train/LLaVA/llava_trainer.py` 约第295-330行

```python
if moe_moka_enable:
    # adapter_model.bin 中的 key 含 ".default."，与模型 state dict 直接匹配
    model.load_state_dict(adapters_weights, strict=False)
else:
    set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
```

在每个任务开始（第2到第8个任务）前调用，加载上一任务的 adapter 后继续训练。

### 5.4 评测加载 — `load_pretrained_model`

**文件：** `ETrain/Models/LLaVA/builder.py` 约第78-110行

```python
# 通过读取 adapter_config.json 判断类型（比依赖模型名更可靠）
_adapter_config_path = os.path.join(model_path, 'adapter_config.json')
if os.path.exists(_adapter_config_path):
    _peft_type = json.load(open(_adapter_config_path)).get('peft_type', '')
is_moe_moka = (_peft_type == 'MOE_MOKA_CoIN') or ('moka' in model_name.lower())

if is_moe_moka:
    adapter_config = MoEMoKALoraConfig.from_pretrained(model_path)
    model = get_peft_model(model, adapter_config)
    weights = torch.load(os.path.join(model_path, WEIGHTS_NAME), map_location='cpu')
    model.load_state_dict(weights, strict=False)
    # MoE-MoKA 不支持合并；始终保持未合并状态。
```

**重要：** 检测方式优先读 `adapter_config.json` 中的 `peft_type`，不再依赖目录名包含 "moka"，确保任意命名的 checkpoint 均可正确加载。

---

## 6. 训练入口

### 6.1 `ModelArguments.moe_moka_enable`

**文件：** `ETrain/Train/LLaVA/train.py` 约第64行

```python
@dataclass
class ModelArguments:
    ...
    moe_moka_enable: bool = field(default=False)
    expert_num: int = field(default=4)
```

在 `Trainer` 构建前（约第119行）传播到 `training_args.moe_moka_enable`。

### 6.2 PEFT 包装

**文件：** `ETrain/Train/LLaVA/train.py` 约第200-250行

```python
if model_args.moe_moka_enable:
    lora_config = MoEMoKALoraConfig(
        r=training_args.lora_r,
        expert_num=model_args.expert_num,
        ...
    )
    model = get_peft_model(model, lora_config)
```

---

## 7. ZeRO-3 兼容性

### 问题

DeepSpeed ZeRO-3 在启动时通过 trace 模块调用顺序来预取参数分片。若代码中出现 `if has_text: expert.lora_A_text(flat_text)` 这样的条件调用，某个 rank 上可能永远不执行某个分支，导致 trace 缓存不一致，进而引发 NCCL hang 或梯度错误。

### 修复方案

所有 A 矩阵无条件调用（若对应模态不存在则传入空张量）：

```python
out_text = expert.lora_A_text(flat_text)   # 始终调用
out_vis  = expert.lora_A_vis(flat_vis)     # 始终调用
if has_text:
    a_out[text_mask] = out_text
if has_vis:
    a_out[vis_mask]  = out_vis
```

**文件：** `CoIN/peft/tuners/mokamoelora.py` 约第220-260行

---

## 8. 持续学习序列

### 脚本结构

```
scripts/LLaVA/Train_MoEMoKA/
├── coin_paths.sh             # 路径中心化配置；自动检测 GPU 数量
├── 1_Science.sh              # 任务1训练（无前序任务）
├── 2_TextVQA.sh              # 任务2训练（加载任务1 adapter）
├── ...
├── 8_OCRVQA.sh               # 任务8训练（加载任务7 adapter）
├── run_coin_sequence.sh      # 编排全部8个任务 + 在线评测
├── mini_pipeline_test.sh     # 全8任务3步 mini 冒烟测试
└── eval/
    ├── eval_common.sh        # 共享评测环境（source coin_paths.sh）
    ├── 1_eval_sqa.sh         # ScienceQA 评测
    ├── 2_eval_textqa.sh      # TextVQA 评测
    ├── ...
    └── 8_eval_ocrvqa.sh      # OCRVQA 评测
```

### 任务间 Checkpoint 流转

```
任务1：基础模型 → [训练] → ScienceQA_llava_MoEMoKA_lora/
任务2：ScienceQA_llava_MoEMoKA_lora/ → [加载] → [训练] → TextVQA_llava_MoEMoKA_lora/
任务3：TextVQA_llava_MoEMoKA_lora/ → [加载] → [训练] → ImageNet_llava_MoEMoKA_lora/
...
任务8：VQAv2_llava_MoEMoKA_lora/ → [加载] → [训练] → OCRVQA_llava_MoEMoKA_lora/
```

### 运行方式

**完整持续学习训练（4卡或8卡）：**
```bash
# 在仓库根目录，激活 conda 环境后执行：
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh
# 或指定任务范围：
bash scripts/LLaVA/Train_MoEMoKA/run_coin_sequence.sh 1 8
```

**Mini 冒烟测试（任意 GPU 数量，每任务约3步）：**
```bash
bash scripts/LLaVA/Train_MoEMoKA/mini_pipeline_test.sh
# 跳过评测：
SKIP_EVAL=1 bash scripts/LLaVA/Train_MoEMoKA/mini_pipeline_test.sh
```

**单独评测（训练完成后）：**
```bash
# 参数：STAGE  MODEL_PATH  LORA_MODE
bash scripts/LLaVA/Train_MoEMoKA/eval/1_eval_sqa.sh \
    Task1 \
    checkpoints/LLaVA/CoIN/ScienceQA_llava_MoEMoKA_lora \
    all
```

---

## 9. 与 MoKA 论文的已知差异

| 项目 | 论文 | 实现 | 原因 |
|------|------|------|------|
| 交叉注意力缩放因子 | `sqrt(N_t)`（文本 token 数量） | `sqrt(d_k)`（rank r） | 标准注意力缩放；N_t 随样本变化导致不稳定 |
| 交叉注意力门控 | 无 | `lora_attn` 可学习标量 | 增加表达能力；初始化为小值 |
| 持续学习 | 论文未涉及 | 加载上一任务 adapter | CoIN 框架需求 |

---

## 10. 快速验证

**训练完一个任务后，验证 checkpoint：**
```bash
ls checkpoints/LLaVA/CoIN/ScienceQA_llava_MoEMoKA_lora/
# 预期输出：
# adapter_config.json   adapter_model.bin   config.json   non_lora_trainables.bin
```

**评测完成后，查看准确率：**
```bash
# ScienceQA：准确率在 output_result.jsonl 中
cat results/CoIN/LLaVA/MoEMoKA/ScienceQA/<STAGE>/output_result.jsonl \
    | python -c "import json,sys; print(json.load(sys.stdin)['acc'])"
```
