"""
merge_grad_csvs.py — merge per-rank grad_stats and attn_stats CSVs into one file per task.

Merging strategy (both grad and attn):
  At each (step, layer), average metric values across ranks FIRST,
  then recompute derived ratios from the averages.  Hooks fire before
  reduce-scatter so each rank holds a local shard; averaging is correct.

Usage:
  python scripts/analysis/merge_grad_csvs.py \\
      --grad_dir  analysis/gradient_dominance \\
      --attn_dir  analysis/attn_dominance \\
      --output_dir analysis/merged

  # legacy flag (grad only):
  python scripts/analysis/merge_grad_csvs.py \\
      --input_dir analysis/gradient_dominance \\
      --output_dir analysis/gradient_dominance/merged

Output per task:
  {task}_merged_grad_stats.csv   — one row per (step, layer), averaged over ranks
  {task}_merged_grad_summary.csv — per-layer + global R summary
  {task}_merged_attn_stats.csv   — one row per (step, layer), averaged over ranks
  {task}_merged_attn_summary.csv — per-layer + global R_att summary
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


def merge_task_attn(task_name: str, rank_files: list[str], output_dir: str) -> None:
    """Merge per-rank attn_stats CSVs: average values across ranks, recompute ratios.

    Input columns (attn_stats.csv):
      step, layer, A_tt, A_tv, R_att_text, U_vis, U_text, R_info, n_post_text
    Output merged_attn_stats.csv:
      step, layer, A_tt, A_tv, R_att_text, U_vis, U_text, R_info, n_post_text, n_ranks
    Output merged_attn_summary.csv:
      scope, layer, mean_A_tt, mean_A_tv, R_att_text, mean_U_vis, mean_U_text, R_info, n_steps
    """
    agg: dict = defaultdict(lambda: {
        "A_tt": [], "A_tv": [],
        "U_vis": [], "U_text": [],
        "n_post_text": [],
    })
    for path in rank_files:
        for row in load_csv(path):
            key = (int(row["step"]), row["layer"])
            d = agg[key]
            d["A_tt"].append(float(row["A_tt"]))
            d["A_tv"].append(float(row["A_tv"]))
            d["U_vis"].append(float(row["U_vis"]))
            d["U_text"].append(float(row["U_text"]))
            d["n_post_text"].append(float(row["n_post_text"]))

    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, f"{task_name}_merged_attn_stats.csv")

    layer_data: dict = defaultdict(lambda: {
        "A_tt": [], "A_tv": [],
        "U_vis": [], "U_text": [],
    })

    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "layer",
            "A_tt", "A_tv", "R_att_text",
            "U_vis", "U_text", "R_info",
            "n_post_text", "n_ranks",
        ])
        for (step, layer), d in sorted(agg.items()):
            n_ranks  = len(d["A_tt"])
            A_tt     = statistics.mean(d["A_tt"]);  A_tv  = statistics.mean(d["A_tv"])
            U_vis    = statistics.mean(d["U_vis"]); U_text = statistics.mean(d["U_text"])
            n_post   = statistics.mean(d["n_post_text"])
            R_att    = A_tt  / max(A_tt  + A_tv,   1e-8)
            R_info   = U_vis / max(U_vis + U_text,  1e-8)
            w.writerow([
                step, layer,
                f"{A_tt:.6f}",  f"{A_tv:.6f}",  f"{R_att:.4f}",
                f"{U_vis:.6f}", f"{U_text:.6f}", f"{R_info:.4f}",
                f"{n_post:.1f}", n_ranks,
            ])
            ld = layer_data[layer]
            ld["A_tt"].append(A_tt);   ld["A_tv"].append(A_tv)
            ld["U_vis"].append(U_vis); ld["U_text"].append(U_text)

    summary_path = os.path.join(output_dir, f"{task_name}_merged_attn_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scope", "layer",
            "mean_A_tt", "mean_A_tv", "R_att_text",
            "mean_U_vis", "mean_U_text", "R_info",
            "n_steps",
        ])
        all_tt, all_tv, all_uv, all_ut = [], [], [], []
        for layer, ld in sorted(layer_data.items()):
            n    = len(ld["A_tt"])
            A_tt = statistics.mean(ld["A_tt"]);  A_tv  = statistics.mean(ld["A_tv"])
            U_vis = statistics.mean(ld["U_vis"]); U_text = statistics.mean(ld["U_text"])
            w.writerow([
                "layer", layer,
                f"{A_tt:.6f}",  f"{A_tv:.6f}",  f"{A_tt/max(A_tt+A_tv,1e-8):.4f}",
                f"{U_vis:.6f}", f"{U_text:.6f}", f"{U_vis/max(U_vis+U_text,1e-8):.4f}",
                n,
            ])
            all_tt.extend(ld["A_tt"]);  all_tv.extend(ld["A_tv"])
            all_uv.extend(ld["U_vis"]); all_ut.extend(ld["U_text"])

        g_tt  = statistics.mean(all_tt); g_tv  = statistics.mean(all_tv)
        g_uv  = statistics.mean(all_uv); g_ut  = statistics.mean(all_ut)
        w.writerow([
            "global", "ALL",
            f"{g_tt:.6f}",  f"{g_tv:.6f}",  f"{g_tt/max(g_tt+g_tv,1e-8):.4f}",
            f"{g_uv:.6f}",  f"{g_ut:.6f}",  f"{g_uv/max(g_uv+g_ut,1e-8):.4f}",
            sum(len(ld["A_tt"]) for ld in layer_data.values()),
        ])

    print(f"[merge-attn] {task_name}: {len(agg)} (step,layer) pairs from {len(rank_files)} rank(s)")
    print(f"  stats  → {stats_path}")
    print(f"  summary→ {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    # New flags
    parser.add_argument("--grad_dir",   default=None,
                        help="Dir with *_rank*_grad_stats.csv files")
    parser.add_argument("--attn_dir",   default=None,
                        help="Dir with *_rank*_attn_stats.csv files")
    parser.add_argument("--output_dir", default="analysis/merged")
    # Legacy flag (grad only)
    parser.add_argument("--input_dir",  default=None,
                        help="Legacy: same as --grad_dir")
    args = parser.parse_args()

    grad_dir = args.grad_dir or args.input_dir or "analysis/gradient_dominance"
    attn_dir = args.attn_dir

    # ── merge grad CSVs ───────────────────────────────────────────────────────
    grad_pattern = re.compile(r"^(.+)_rank\d+_grad_stats\.csv$")
    grad_tasks: dict = defaultdict(list)
    if os.path.isdir(grad_dir):
        for fname in sorted(os.listdir(grad_dir)):
            m = grad_pattern.match(fname)
            if m:
                grad_tasks[m.group(1)].append(os.path.join(grad_dir, fname))

    if grad_tasks:
        for task_name, files in sorted(grad_tasks.items()):
            print(f"\n── grad: {task_name} ({len(files)} rank file(s)) ──")
            merge_task(task_name, files, args.output_dir)
    else:
        print(f"No grad rank CSV files found in {grad_dir}")

    # ── merge attn CSVs ───────────────────────────────────────────────────────
    if attn_dir and os.path.isdir(attn_dir):
        attn_pattern = re.compile(r"^(.+)_rank\d+_attn_stats\.csv$")
        attn_tasks: dict = defaultdict(list)
        for fname in sorted(os.listdir(attn_dir)):
            m = attn_pattern.match(fname)
            if m:
                attn_tasks[m.group(1)].append(os.path.join(attn_dir, fname))

        if attn_tasks:
            for task_name, files in sorted(attn_tasks.items()):
                print(f"\n── attn: {task_name} ({len(files)} rank file(s)) ──")
                merge_task_attn(task_name, files, args.output_dir)
        else:
            print(f"No attn rank CSV files found in {attn_dir}")


if __name__ == "__main__":
    main()
