"""
merge_grad_csvs.py — merge per-rank grad_stats CSVs into one file per task.

Merging strategy:
  At each (step, layer), average G_text and G_vis across ranks FIRST,
  then recompute R from the averages.  This matches the DDP all-reduce
  semantics (all-reduce averages actual gradient tensors, not their norms).

Usage:
  python scripts/analysis/merge_grad_csvs.py \\
      --input_dir analysis/gradient_dominance \\
      --output_dir analysis/gradient_dominance/merged

Output per task:
  {task}_merged_grad_stats.csv   — one row per (step, layer), averaged over ranks
  {task}_merged_summary.csv      — per-layer + global R summary
"""
import argparse
import csv
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path


def _ratio(t: float, v: float) -> float:
    return t / max(v, 1e-8)


def _ratio_tok(t: float, v: float, nt: int, nv: int) -> float:
    if nt <= 0 or nv <= 0 or v < 1e-8:
        return float("inf")
    return (t / nt) / max(v / nv, 1e-12)


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def merge_task(task_name: str, rank_files: list[str], output_dir: str) -> None:
    # Accumulate G values per (step, layer) across all ranks
    # key: (step, layer) → lists of G values
    agg: dict = defaultdict(lambda: {
        "G_text_A": [], "G_vis_A": [],
        "G_text_B": [], "G_vis_B": [],
        "G_text_dW": [], "G_vis_dW": [],
        "n_text": [], "n_vis": [],
    })

    for path in rank_files:
        for row in load_csv(path):
            key = (int(row["step"]), row["layer"])
            d = agg[key]
            d["G_text_A"].append(float(row["G_text_A"]))
            d["G_vis_A"].append(float(row["G_vis_A"]))
            d["G_text_B"].append(float(row["G_text_B"]))
            d["G_vis_B"].append(float(row["G_vis_B"]))
            d["G_text_dW"].append(float(row["G_text_dW"]))
            d["G_vis_dW"].append(float(row["G_vis_dW"]))
            d["n_text"].append(int(row["n_text"]))
            d["n_vis"].append(int(row["n_vis"]))

    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, f"{task_name}_merged_grad_stats.csv")

    # Per-layer accumulators for summary
    layer_data: dict = defaultdict(lambda: {
        "G_text_A": [], "G_vis_A": [],
        "G_text_B": [], "G_vis_B": [],
        "G_text_dW": [], "G_vis_dW": [],
        "n_text": [], "n_vis": [],
    })

    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "layer",
            "G_text_A", "G_vis_A", "R_A", "R_A_tok",
            "G_text_B", "G_vis_B", "R_B", "R_B_tok",
            "G_text_dW", "G_vis_dW", "R_dW", "R_dW_tok",
            "n_text", "n_vis", "n_ranks",
        ])
        for (step, layer), d in sorted(agg.items()):
            n_ranks = len(d["G_text_A"])
            mt_A  = statistics.mean(d["G_text_A"]);  mv_A  = statistics.mean(d["G_vis_A"])
            mt_B  = statistics.mean(d["G_text_B"]);  mv_B  = statistics.mean(d["G_vis_B"])
            mt_dW = statistics.mean(d["G_text_dW"]); mv_dW = statistics.mean(d["G_vis_dW"])
            nt = int(statistics.mean(d["n_text"]));  nv = int(statistics.mean(d["n_vis"]))
            w.writerow([
                step, layer,
                f"{mt_A:.6f}",  f"{mv_A:.6f}",  f"{_ratio(mt_A, mv_A):.4f}",   f"{_ratio_tok(mt_A, mv_A, nt, nv):.4f}",
                f"{mt_B:.6f}",  f"{mv_B:.6f}",  f"{_ratio(mt_B, mv_B):.4f}",   f"{_ratio_tok(mt_B, mv_B, nt, nv):.4f}",
                f"{mt_dW:.6f}", f"{mv_dW:.6f}", f"{_ratio(mt_dW, mv_dW):.4f}", f"{_ratio_tok(mt_dW, mv_dW, nt, nv):.4f}",
                nt, nv, n_ranks,
            ])
            ld = layer_data[layer]
            ld["G_text_A"].append(mt_A);  ld["G_vis_A"].append(mv_A)
            ld["G_text_B"].append(mt_B);  ld["G_vis_B"].append(mv_B)
            ld["G_text_dW"].append(mt_dW); ld["G_vis_dW"].append(mv_dW)
            ld["n_text"].append(nt);       ld["n_vis"].append(nv)

    # Write summary
    summary_path = os.path.join(output_dir, f"{task_name}_merged_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scope", "layer",
            "mean_G_text_A", "mean_G_vis_A", "R_A", "R_A_tok",
            "mean_G_text_B", "mean_G_vis_B", "R_B", "R_B_tok",
            "mean_G_text_dW", "mean_G_vis_dW", "R_dW", "R_dW_tok",
            "n_steps",
        ])
        all_Gt_A, all_Gv_A = [], []
        all_Gt_B, all_Gv_B = [], []
        all_Gt_dW, all_Gv_dW = [], []
        all_nt, all_nv = [], []
        for layer, ld in sorted(layer_data.items()):
            n = len(ld["G_text_A"])
            mt_A  = statistics.mean(ld["G_text_A"]);  mv_A  = statistics.mean(ld["G_vis_A"])
            mt_B  = statistics.mean(ld["G_text_B"]);  mv_B  = statistics.mean(ld["G_vis_B"])
            mt_dW = statistics.mean(ld["G_text_dW"]); mv_dW = statistics.mean(ld["G_vis_dW"])
            mean_nt = statistics.mean(ld["n_text"]);   mean_nv = statistics.mean(ld["n_vis"])
            w.writerow([
                "layer", layer,
                f"{mt_A:.6f}", f"{mv_A:.6f}", f"{_ratio(mt_A,mv_A):.4f}", f"{_ratio_tok(mt_A,mv_A,int(mean_nt),int(mean_nv)):.4f}",
                f"{mt_B:.6f}", f"{mv_B:.6f}", f"{_ratio(mt_B,mv_B):.4f}", f"{_ratio_tok(mt_B,mv_B,int(mean_nt),int(mean_nv)):.4f}",
                f"{mt_dW:.6f}", f"{mv_dW:.6f}", f"{_ratio(mt_dW,mv_dW):.4f}", f"{_ratio_tok(mt_dW,mv_dW,int(mean_nt),int(mean_nv)):.4f}",
                n,
            ])
            all_Gt_A.extend(ld["G_text_A"]);  all_Gv_A.extend(ld["G_vis_A"])
            all_Gt_B.extend(ld["G_text_B"]);  all_Gv_B.extend(ld["G_vis_B"])
            all_Gt_dW.extend(ld["G_text_dW"]); all_Gv_dW.extend(ld["G_vis_dW"])
            all_nt.extend(ld["n_text"]);       all_nv.extend(ld["n_vis"])

        gmt_A  = statistics.mean(all_Gt_A);  gmv_A  = statistics.mean(all_Gv_A)
        gmt_B  = statistics.mean(all_Gt_B);  gmv_B  = statistics.mean(all_Gv_B)
        gmt_dW = statistics.mean(all_Gt_dW); gmv_dW = statistics.mean(all_Gv_dW)
        mean_nt_g = statistics.mean(all_nt);  mean_nv_g = statistics.mean(all_nv)
        w.writerow([
            "global", "ALL",
            f"{gmt_A:.6f}", f"{gmv_A:.6f}", f"{_ratio(gmt_A,gmv_A):.4f}", f"{_ratio_tok(gmt_A,gmv_A,int(mean_nt_g),int(mean_nv_g)):.4f}",
            f"{gmt_B:.6f}", f"{gmv_B:.6f}", f"{_ratio(gmt_B,gmv_B):.4f}", f"{_ratio_tok(gmt_B,gmv_B,int(mean_nt_g),int(mean_nv_g)):.4f}",
            f"{gmt_dW:.6f}", f"{gmv_dW:.6f}", f"{_ratio(gmt_dW,gmv_dW):.4f}", f"{_ratio_tok(gmt_dW,gmv_dW,int(mean_nt_g),int(mean_nv_g)):.4f}",
            sum(len(ld["G_text_A"]) for ld in layer_data.values()),
        ])

    print(f"[merge] {task_name}: {len(agg)} (step,layer) pairs from {len(rank_files)} rank(s)")
    print(f"  stats  → {stats_path}")
    print(f"  summary→ {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="analysis/gradient_dominance")
    parser.add_argument("--output_dir", default="analysis/gradient_dominance/merged")
    args = parser.parse_args()

    # Group files by task name: {task}_rank{N}_grad_stats.csv
    pattern = re.compile(r"^(.+)_rank\d+_grad_stats\.csv$")
    tasks: dict = defaultdict(list)
    for fname in sorted(os.listdir(args.input_dir)):
        m = pattern.match(fname)
        if m:
            tasks[m.group(1)].append(os.path.join(args.input_dir, fname))

    if not tasks:
        print(f"No rank CSV files found in {args.input_dir}")
        return

    for task_name, files in sorted(tasks.items()):
        print(f"\n── {task_name} ({len(files)} rank file(s)) ──")
        merge_task(task_name, files, args.output_dir)


if __name__ == "__main__":
    main()
