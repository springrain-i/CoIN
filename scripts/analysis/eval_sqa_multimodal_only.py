"""
Recompute ScienceQA accuracy on multimodal-only samples (those with an image).

Reads the existing output.jsonl produced by eval_science_qa.py — does NOT
modify any existing files.

Usage:
    python scripts/analysis/eval_sqa_multimodal_only.py \
        --output-file results/CoIN/LLaVA/.../output.jsonl

    # Or scan an entire result tree and print a summary table:
    python scripts/analysis/eval_sqa_multimodal_only.py \
        --result-dir results/CoIN/LLaVA/ScienceQA_NoMerge_Visual
"""

import argparse
import json
import os


def compute_mm_acc(output_file: str) -> dict:
    data = json.load(open(output_file))
    mm_correct = [x for x in data["correct"] if x.get("is_multimodal")]
    mm_incorrect = [x for x in data["incorrect"] if x.get("is_multimodal")]
    total_all = len(data["correct"]) + len(data["incorrect"])
    mm_total = len(mm_correct) + len(mm_incorrect)
    if mm_total == 0:
        return None
    return {
        "all_correct": len(data["correct"]),
        "all_total": total_all,
        "all_acc": len(data["correct"]) / total_all * 100,
        "mm_correct": len(mm_correct),
        "mm_total": mm_total,
        "mm_acc": len(mm_correct) / mm_total * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to a single output.jsonl")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="Root directory; recursively finds all output.jsonl")
    args = parser.parse_args()

    if args.output_file:
        r = compute_mm_acc(args.output_file)
        if r is None:
            print("No multimodal samples found.")
            return
        print(f"All samples  : {r['all_correct']}/{r['all_total']} = {r['all_acc']:.2f}%")
        print(f"Multimodal   : {r['mm_correct']}/{r['mm_total']} = {r['mm_acc']:.2f}%")
        unimodal_total = r["all_total"] - r["mm_total"]
        print(f"Unimodal only: {r['all_total'] - r['mm_total'] - (r['all_correct'] - r['mm_correct'])}"
              f"/{unimodal_total} correct (excluded)")
        return

    if args.result_dir:
        rows = []
        for root, dirs, files in os.walk(args.result_dir):
            if "output.jsonl" in files:
                path = os.path.join(root, "output.jsonl")
                r = compute_mm_acc(path)
                if r is None:
                    continue
                label = os.path.relpath(root, args.result_dir)
                rows.append((label, r))

        if not rows:
            print("No output.jsonl files found.")
            return

        # Print table
        col = max(len(r[0]) for r in rows)
        header = f"{'Stage':<{col}}  {'All acc':>8}  {'MM acc':>8}  {'MM n':>6}"
        print(header)
        print("-" * len(header))
        for label, r in sorted(rows):
            print(f"{label:<{col}}  {r['all_acc']:>7.2f}%  {r['mm_acc']:>7.2f}%  {r['mm_total']:>6}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
