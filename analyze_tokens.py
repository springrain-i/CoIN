"""
统计 JSONL 文件中每行 text 字段的 token 数量（使用 LLaVA/Vicuna 的 HuggingFace tokenizer）。
并生成 token 长度分布图（histogram + KDE 波峰曲线）。

用法:
    python analyze_tokens.py <path_to_jsonl> [--model /hy-tmp/Vicuna/vicuna-7b-v1.5]
"""

import json
import sys
import argparse
import statistics

try:
    from transformers import AutoTokenizer
except ImportError:
    print("请先安装 transformers：pip install transformers", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    from scipy.stats import gaussian_kde
except ImportError:
    print("请先安装依赖：pip install matplotlib numpy scipy", file=sys.stderr)
    sys.exit(1)


def plot_distribution(lengths: list[int], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    # histogram
    bins = min(80, max(10, (max(lengths) - min(lengths))))
    n, edges, patches = ax.hist(
        lengths, bins=bins, color="#4C8BF5", alpha=0.5,
        edgecolor="white", linewidth=0.4, label="Histogram"
    )

    # KDE 波峰曲线
    kde = gaussian_kde(lengths, bw_method="scott")
    x = np.linspace(min(lengths) - 1, max(lengths) + 1, 500)
    kde_y = kde(x)
    # 缩放 KDE 到和 histogram 同一 y 轴（密度 → 频次）
    bin_width = edges[1] - edges[0]
    kde_scaled = kde_y * len(lengths) * bin_width
    ax.plot(x, kde_scaled, color="#E8402A", linewidth=2.2, label="KDE")

    # 统计线
    mean_val = sum(lengths) / len(lengths)
    median_val = statistics.median(lengths)
    std_val = statistics.stdev(lengths) if len(lengths) > 1 else 0
    ax.axvline(mean_val,   color="#F5A623", linewidth=1.6, linestyle="--", label=f"Mean={mean_val:.1f}")
    ax.axvline(median_val, color="#7ED321", linewidth=1.6, linestyle=":",  label=f"Median={median_val:.1f}")
    ax.axvline(mean_val + 2 * std_val, color="#9B59B6", linewidth=1.2,
               linestyle="-.", label=f"Mean+2σ={mean_val + 2*std_val:.1f}")

    ax.set_xlabel("Token Length", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Token Length Distribution (text field)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"分布图已保存：{output_path}")


def analyze(filepath: str, model_path: str):
    print(f"加载 tokenizer：{model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception as e:
        print(f"无法加载 tokenizer：{e}", file=sys.stderr)
        sys.exit(1)

    records = []  # list of (token_length, question_id)

    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] Line {lineno} is not valid JSON: {e}", file=sys.stderr)
                continue

            text = obj.get("text", "")
            qid = obj.get("question_id", f"<line {lineno}>")
            length = len(tokenizer.encode(text, add_special_tokens=False))
            records.append((length, qid))

    if not records:
        print("No valid records found.")
        return

    lengths = [r[0] for r in records]
    min_len = min(lengths)
    max_len = max(lengths)
    mean_val = sum(lengths) / len(lengths)
    std_val = statistics.stdev(lengths) if len(lengths) > 1 else 0

    min_ids = [qid for l, qid in records if l == min_len]
    max_ids = [qid for l, qid in records if l == max_len]

    print(f"\nModel    : {model_path}")
    print(f"Records  : {len(lengths)}")
    print(f"Min      : {min_len}  (question_id: {', '.join(str(i) for i in min_ids)})")
    print(f"Max      : {max_len}  (question_id: {', '.join(str(i) for i in max_ids)})")
    print(f"Avg      : {mean_val:.2f}")
    print(f"Median   : {statistics.median(lengths):.1f}")
    print(f"Std      : {std_val:.2f}")
    print()
    print(f"建议 max_new_tokens 设置参考：")
    print(f"  保守（覆盖 max）  : {max_len}")
    print(f"  均值 + 2σ        : {int(mean_val + 2 * std_val)}")

    # 输出图与 jsonl 同目录，同名 .png
    import os
    base = os.path.splitext(os.path.abspath(filepath))[0]
    plot_distribution(lengths, base + "_token_dist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 JSONL text 字段的 LLaVA/Vicuna token 数并绘图")
    parser.add_argument("filepath", help="JSONL 文件路径")
    parser.add_argument(
        "--model",
        default="/hy-tmp/Vicuna/vicuna-7b-v1.5",
        help="HuggingFace tokenizer 路径（默认: /hy-tmp/Vicuna/vicuna-7b-v1.5）",
    )
    args = parser.parse_args()

    analyze(args.filepath, args.model)