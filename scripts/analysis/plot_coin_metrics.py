import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

TASK_NAMES = {
    1: "ScienceQA",
    2: "TextVQA",
    3: "ImageNet",
    4: "GQA",
    5: "VizWiz",
    6: "Grounding",
    7: "VQAv2",
    8: "OCRVQA",
}


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "regime": r["regime"],
                    "mode": r["mode"],
                    "train_task": int(r["train_task"]),
                    "eval_task": int(r["eval_task"]),
                    "accuracy": float(r["accuracy"]),
                }
            )
    return rows


def plot_single(rows, out_dir):
    data = defaultdict(dict)
    modes = sorted({r["mode"] for r in rows})
    tasks = sorted({r["train_task"] for r in rows})

    for r in rows:
        if r["train_task"] == r["eval_task"]:
            data[r["mode"]][r["train_task"]] = r["accuracy"]

    x = list(range(len(tasks)))
    width = 0.25
    plt.figure(figsize=(13, 5))
    for i, mode in enumerate(modes):
        ys = [data[mode].get(t, 0.0) for t in tasks]
        xs = [p + (i - 1) * width for p in x]
        plt.bar(xs, ys, width=width, label=mode)

    plt.title("CoIN 单任务独立训练: 三模式正确率对比")
    plt.xlabel("实验顺序")
    plt.ylabel("正确率 /%")
    plt.xticks(x, [TASK_NAMES[t] for t in tasks], rotation=20)
    plt.legend(title="模式")
    plt.tight_layout()
    out = os.path.join(out_dir, "single_task_mode_accuracy.png")
    plt.savefig(out, dpi=180)
    plt.close()


def plot_continual(rows, out_dir):
    rows_by_mode = defaultdict(list)
    for r in rows:
        rows_by_mode[r["mode"]].append(r)

    modes = sorted(rows_by_mode.keys())
    train_stages = sorted({r["train_task"] for r in rows})

    # Current-task accuracy curves: eval_task == train_task
    plt.figure(figsize=(12, 5))
    for mode in modes:
        curve = []
        idx_map = {(r["train_task"], r["eval_task"]): r["accuracy"] for r in rows_by_mode[mode]}
        for t in train_stages:
            curve.append(idx_map.get((t, t), 0.0))
        plt.plot(train_stages, curve, marker="o", label=mode)

    plt.title("CoIN 持续学习: 三模式正确率对比")
    plt.xlabel("实验顺序")
    plt.ylabel("正确率 /%")
    plt.xticks(train_stages, [TASK_NAMES[t] for t in train_stages], rotation=20)
    plt.legend(title="模式")
    plt.tight_layout()
    out_curve = os.path.join(out_dir, "continual_current_task_accuracy_curve.png")
    plt.savefig(out_curve, dpi=180)
    plt.close()

    # Forgetting per task at final stage
    final_stage = max(train_stages)
    width = 0.25
    x = list(range(1, final_stage))
    plt.figure(figsize=(13, 5))

    for i, mode in enumerate(modes):
        rows_mode = rows_by_mode[mode]
        acc_map = defaultdict(dict)
        for r in rows_mode:
            acc_map[r["eval_task"]][r["train_task"]] = r["accuracy"]

        forgetting_vals = []
        for task in range(1, final_stage):
            stage_accs = [acc for st, acc in sorted(acc_map[task].items()) if st >= task]
            if not stage_accs:
                forgetting_vals.append(0.0)
                continue
            peak = max(stage_accs)
            final = acc_map[task].get(final_stage, stage_accs[-1])
            forgetting_vals.append(max(0.0, peak - final))

        xs = [p + (i - 1) * width for p in x]
        plt.bar(xs, forgetting_vals, width=width, label=mode)

        if forgetting_vals:
            peak_idx = max(range(len(forgetting_vals)), key=lambda k: forgetting_vals[k])
            peak_task = x[peak_idx]
            peak_val = forgetting_vals[peak_idx]
            plt.text(
                xs[peak_idx],
                peak_val + 0.2,
                "峰值 {:.2f}".format(peak_val),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=20,
            )

    plt.title("CoIN 持续学习: 三模式遗忘率对比")
    plt.xlabel("实验顺序")
    plt.ylabel("遗忘率 /%")
    plt.xticks(x, [TASK_NAMES[t] for t in x], rotation=20)
    plt.legend(title="模式")
    plt.tight_layout()
    out_forget = os.path.join(out_dir, "continual_forgetting_bar.png")
    plt.savefig(out_forget, dpi=180)
    plt.close()


def save_summary_json(rows, out_dir):
    out = os.path.join(out_dir, "metric_summary.json")
    by_regime = defaultdict(list)
    for r in rows:
        by_regime[r["regime"]].append(r)

    summary = {}
    for regime, items in by_regime.items():
        summary[regime] = {
            "num_records": len(items),
            "modes": sorted({i["mode"] for i in items}),
            "tasks": sorted({i["train_task"] for i in items}),
        }

    import json
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="results/CoIN/LLaVA/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.csv)

    contin = [r for r in rows if r["regime"] == "continual"]
    single = [r for r in rows if r["regime"] == "single"]

    if contin:
        plot_continual(contin, args.out_dir)
    if single:
        plot_single(single, args.out_dir)
    save_summary_json(rows, args.out_dir)

    print("Saved plots and summary to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
