# CoIN/MoELoRA 性能优化总结（中文）

> 分支：`optimize`（worktree: `/data4/home/sqx/CoIN_optimize`）  
> 完成日期：2026-05-29  
> 测试环境：8× RTX 3090，CUDA 11.8，PyTorch 2.x，DeepSpeed 0.x

---

## 已完成的优化项

### 1. Eval 批量推理（batch_size > 1）

**文件改动：**
- `ETrain/Eval/LLaVA/model_vqa.py`
- `ETrain/Eval/LLaVA/model_vqa_science.py`
- `ETrain/Eval/LLaVA/model_vqa_grd.py`
- `ETrain/Eval/LLaVA/model_vqa_imagenet.py`
- `ETrain/Eval/LLaVA/model_vqa_imagenet_val.py`

**问题根因：**  
所有 eval 脚本原来使用 batch=1 逐样本推理（每次 `model.generate()` 只处理 1 个样本），
GPU 利用率极低，绝大多数时间花在 Python 循环、tokenize 和图像预处理上。

**改动方式：**  
为每个脚本添加 `--batch-size` 参数（默认 1，向后兼容），在 collate 阶段将多个样本的
`input_ids`、`image_tensor` 拼接为批次，统一调用一次 `model.generate()`。

**实测加速：**  
- batch=1：基准（1.0×）  
- batch=8：**2.57×**（GPU 6 单卡，200 样本测试）

**使用方式：**
```bash
bash scripts/LLaVA/bench_eval_batch.sh 200   # 自动测试 batch 1/2/4/8
# 或直接：
python ETrain/Eval/LLaVA/model_vqa_science.py --batch-size 8 ...
```

---

### 2. MoELoRA Forward 向量化（2 × einsum）

**文件改动：**
- `CoIN/peft/tuners/coinmoelora.py`（optimize 分支）
- `/data4/home/sqx/CoIN/src/etrain/CoIN/peft/tuners/lora.py`（原始仓库，补加缺失方法）

**问题根因：**  
原始 `CoINMOELoraLinear` 在 forward 中对 8 个专家做串行 Python 循环（8 次分别的 matmul），
每次产生独立 CUDA kernel 调用，完全阻止 CUDA kernel fusion，且有大量 Python dispatch overhead。

**改动方式：**  
提取纯函数 `_moe_lora_compute`，将 8 个专家的权重矩阵堆叠为 3D tensor，
用 2 次 `torch.einsum` 替代 8 次串行 matmul：

```python
def _moe_lora_compute(lora_x, A, B, router, scaling):
    # A: (N, r_per, d_in), B: (N, d_out, r_per), router: (B, T, N)
    out_A = torch.einsum('bti,nri->btnr', lora_x, A)   # 所有专家 lora_A 一次完成
    out_B = torch.einsum('btnr,nor->btno', out_A, B)   # 所有专家 lora_B 一次完成
    return (out_B * router.unsqueeze(-1)).sum(dim=2) * scaling
```

通过环境变量控制开启/关闭：
```bash
COIN_USE_VECTORIZED_LORA=1   # 默认开启（向量化 einsum）
COIN_USE_VECTORIZED_LORA=0   # 关闭（原始串行 loop，用于对比测试）
COIN_USE_COMPILED_LORA=1     # 额外开启 torch.compile（实验性）
```

**补丁说明：**  
coin conda 环境通过 editable install 将所有 `CoIN` 模块路由至原始仓库
（`/data4/home/sqx/CoIN/src/etrain/CoIN/peft/tuners/lora.py`），
因此还需在原始 `LoraLayer` 中补加 `_apply_token_mask` 和 `_get_effective_token_mask` 方法，
否则 `CoINMOELoraLinear.forward()` 调用时会 `AttributeError`。
备份文件：`lora.py.bak`（原始版本）。

**实测加速：**
- **Eval / 推理场景（无 ZeRO-3 通信）：2.57× 加速**（已验证，batch=8）  
- **训练场景（ZeRO-3 + CPU offload）：无加速（±0.3%，噪声范围内）**

训练无加速的根因：ZeRO-3 CPU offload 每步需 AllGather 全量参数（14GB），
耗时占总步时 >99%，LoRA forward（μs 量级）的加速完全被掩盖。

---

### 3. DeepSpeed CPUAdam 编译修复（GCC 9）

**问题根因：**  
系统默认 GCC 版本为 12.3，而 CUDA 11.8 的 JIT 编译限制最高支持 GCC 11。
运行 `bench_train_lora.sh` 时 CPUAdam 无法 JIT 编译，训练进程崩溃。

**修复方式：**  
在 `bench_train_lora.sh` 中添加以下环境变量，将编译器指向系统已安装的 GCC 9：

```bash
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
```

GCC 9 和 GCC 11 均已安装于 `/usr/bin/`，均与 CUDA 11.8 兼容。
第一次运行会 JIT 编译并缓存至 `~/.cache/torch_extensions/py310_cu117/cpu_adam/cpu_adam.so`，
后续运行直接加载缓存，无需重新编译（编译耗时约 32 秒）。

---

## 已排除的优化方向

### ZeRO-2 替代 ZeRO-3

**结论：不可行（已实测 OOM）**

ZeRO-2 不做参数 offload，但 DeepSpeedCPUAdam 优化器状态（FP32，28GB）仍需存在 CPU RAM 中。
实测：4 卡 × 14GB/卡 = 56GB 参数 + 28GB 优化器 ≈ 84GB，超出可用 CPU RAM（83GB）。
8 卡配置下更严重（8 × 14GB = 112GB > 83GB），进程被 OOM killer（SIGKILL -9）终止。

ZeRO-3 通过参数分片将每 GPU 的 CPU 负担降至 3.5GB/GPU，是当前唯一可行方案。

---

## Benchmark 脚本

```bash
# 训练向量化 benchmark（loop vs vectorized）
cd /data4/home/sqx/CoIN_optimize
bash scripts/LLaVA/bench_train_lora.sh 400 50

# 结果位置
cat bench_results/train_lora/bench_train_results.txt
ls logs/LLaVA/bench_train/

# Eval batch benchmark
bash scripts/LLaVA/bench_eval_batch.sh 200
cat bench_results/eval_batch/bench_eval_results.txt
```

---

## 训练 Benchmark 实测结果（2026-05-29）

| 模式 | train_runtime | steps/sec | samples/sec | 稳态 s/step |
|------|-------------|-----------|-------------|------------|
| Loop（原始） | 1054.8s | 0.047 | 0.758 | ~16.6s |
| Einsum（向量化） | 1057.9s | 0.047 | 0.756 | ~16.9s |

**加速比：0.997×（即无加速）。根因：ZeRO-3 CPU offload 主导步时。**

---

## 完整优化报告

详见：`/data4/home/sqx/CoIN/perf_analysis.html`（在浏览器中打开）
