"""
Merge per-rank grad, attention, and expert diagnostic CSVs.

The loggers now own the schema. This script therefore reads CSV headers and
averages every numeric diagnostic field across ranks for the same identity key.

Identity keys:
  grad_stats:   step, microbatch, layer
  attn_stats:   step, microbatch, layer
  expert_stats: step, microbatch, layer, expert

Output:
  {task}_merged_grad_stats.csv
  {task}_merged_grad_summary.csv
  {task}_merged_attn_stats.csv
  {task}_merged_attn_summary.csv
  {task}_merged_expert_stats.csv
  {task}_merged_expert_summary.csv
"""
import argparse
import csv
import math
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


BASE_ID_FIELDS = ("step", "microbatch", "layer")
EXPERT_ID_FIELDS = ("step", "microbatch", "layer", "expert")


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _mean(values: Iterable[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("inf")
    return statistics.mean(finite_values)


def _format_float(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else str(value)


def _grad_ratio(num: float, den: float) -> float:
    return num / max(den, 1e-8)


def _grad_ratio_tok(num: float, den: float, n_num: float, n_den: float) -> float:
    if n_num <= 0 or n_den <= 0 or den < 1e-8:
        return float("inf")
    return (num / n_num) / max(den / n_den, 1e-12)


def _recompute_known_ratios(row: Dict[str, float]) -> None:
    if {"A_ans_vis", "A_ans_prompt", "A_ans_prev_answer"}.issubset(row):
        denom = row["A_ans_vis"] + row["A_ans_prompt"] + row["A_ans_prev_answer"]
        row["R_att_ans_vis"] = row["A_ans_vis"] / max(denom, 1e-8)
    if {"U_ans_vis", "U_ans_prompt", "U_ans_prev_answer"}.issubset(row):
        denom = row["U_ans_vis"] + row["U_ans_prompt"] + row["U_ans_prev_answer"]
        row["R_info_ans_vis"] = row["U_ans_vis"] / max(denom, 1e-8)
    for bucket in ("prompt", "vis", "answer"):
        mass_field = f"route_mass_{bucket}"
        count_field = f"n_{bucket}"
        mean_field = f"route_mean_{bucket}"
        if {mass_field, count_field, mean_field}.issubset(row):
            row[mean_field] = row[mass_field] / max(row[count_field], 1e-8)
    for prefix in ("A", "B", "dW"):
        prompt_field = f"G_prompt_{prefix}"
        visual_field = f"G_vis_{prefix}"
        answer_field = f"G_answer_{prefix}"
        if {prompt_field, visual_field, f"R_prompt_vis_{prefix}"}.issubset(row):
            row[f"R_prompt_vis_{prefix}"] = _grad_ratio(row[prompt_field], row[visual_field])
        if {answer_field, visual_field, f"R_answer_vis_{prefix}"}.issubset(row):
            row[f"R_answer_vis_{prefix}"] = _grad_ratio(row[answer_field], row[visual_field])
        if {prompt_field, visual_field, "n_prompt", "n_vis", f"R_prompt_vis_{prefix}_tok"}.issubset(row):
            row[f"R_prompt_vis_{prefix}_tok"] = _grad_ratio_tok(
                row[prompt_field], row[visual_field], row["n_prompt"], row["n_vis"])
        if {answer_field, visual_field, "n_answer", "n_vis", f"R_answer_vis_{prefix}_tok"}.issubset(row):
            row[f"R_answer_vis_{prefix}_tok"] = _grad_ratio_tok(
                row[answer_field], row[visual_field], row["n_answer"], row["n_vis"])


def _identity_value(field: str, value: str):
    if field in {"step", "microbatch", "expert"}:
        return int(value)
    return value


def _add_derived_fields(fields: List[str]) -> List[str]:
    result = list(fields)
    available = set(fields)
    for prefix in ("A", "B", "dW"):
        if {f"G_prompt_{prefix}", f"G_vis_{prefix}"}.issubset(available):
            result.append(f"R_prompt_vis_{prefix}")
        if {f"G_answer_{prefix}", f"G_vis_{prefix}"}.issubset(available):
            result.append(f"R_answer_vis_{prefix}")
        if {f"G_prompt_{prefix}", f"G_vis_{prefix}", "n_prompt", "n_vis"}.issubset(available):
            result.append(f"R_prompt_vis_{prefix}_tok")
        if {f"G_answer_{prefix}", f"G_vis_{prefix}", "n_answer", "n_vis"}.issubset(available):
            result.append(f"R_answer_vis_{prefix}_tok")
    deduped = []
    seen = set()
    for field in result:
        if field not in seen:
            deduped.append(field)
            seen.add(field)
    return deduped


def merge_task_generic(
    task_name: str,
    rank_files: list[str],
    output_dir: str,
    kind: str,
    id_fields: Tuple[str, ...],
) -> None:
    rows = []
    for path in rank_files:
        rows.extend(load_csv(path))
    if not rows:
        return

    header = list(rows[0].keys())
    missing_ids = [field for field in id_fields if field not in header]
    if missing_ids:
        raise ValueError(f"{rank_files[0]} missing identity fields: {missing_ids}")
    data_fields = _add_derived_fields([field for field in header if field not in id_fields])

    agg: dict = defaultdict(lambda: {field: [] for field in data_fields})
    for row in rows:
        key = tuple(_identity_value(field, row[field]) for field in id_fields)
        bucket = agg[key]
        for field in data_fields:
            bucket[field].append(_to_float(row.get(field)))

    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, f"{task_name}_merged_{kind}_stats.csv")

    merged_rows: List[Tuple[Tuple[str, ...], Dict[str, float], int]] = []
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(id_fields) + data_fields + ["n_ranks"])
        for key, values in sorted(agg.items()):
            merged = {field: _mean(values[field]) for field in data_fields}
            _recompute_known_ratios(merged)
            n_ranks = max(len(next(iter(values.values()))), 0) if values else 0
            writer.writerow(
                list(key)
                + [_format_float(merged[field]) for field in data_fields]
                + [n_ranks]
            )
            merged_rows.append((key, merged, n_ranks))

    summary_path = os.path.join(output_dir, f"{task_name}_merged_{kind}_summary.csv")
    _write_summary(summary_path, merged_rows, data_fields, id_fields)

    print(f"[merge-{kind}] {task_name}: {len(agg)} rows from {len(rank_files)} rank file(s)")
    print(f"  stats   -> {stats_path}")
    print(f"  summary -> {summary_path}")


def _write_summary(
    summary_path: str,
    merged_rows: List[Tuple[Tuple[str, ...], Dict[str, float], int]],
    data_fields: List[str],
    id_fields: Tuple[str, ...],
) -> None:
    layer_idx = id_fields.index("layer")
    expert_idx = id_fields.index("expert") if "expert" in id_fields else None
    group_data: dict = defaultdict(lambda: {field: [] for field in data_fields})

    for key, row, _ in merged_rows:
        group_key = (key[layer_idx], key[expert_idx]) if expert_idx is not None else (key[layer_idx],)
        bucket = group_data[group_key]
        for field in data_fields:
            bucket[field].append(row.get(field, 0.0))

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        if expert_idx is not None:
            writer.writerow(["scope", "layer", "expert", "n_records"] + [f"mean_{field}" for field in data_fields])
        else:
            writer.writerow(["scope", "layer", "n_records"] + [f"mean_{field}" for field in data_fields])

        all_values = {field: [] for field in data_fields}
        for group_key, values in sorted(group_data.items()):
            means = {field: _mean(values[field]) for field in data_fields}
            _recompute_known_ratios(means)
            for field in data_fields:
                all_values[field].extend(values[field])
            if expert_idx is not None:
                writer.writerow(
                    ["layer", group_key[0], group_key[1], len(next(iter(values.values())))]
                    + [_format_float(means[field]) for field in data_fields]
                )
            else:
                writer.writerow(
                    ["layer", group_key[0], len(next(iter(values.values())))]
                    + [_format_float(means[field]) for field in data_fields]
                )

        global_means = {field: _mean(all_values[field]) for field in data_fields}
        _recompute_known_ratios(global_means)
        if expert_idx is not None:
            writer.writerow(
                ["global", "ALL", "ALL", len(merged_rows)]
                + [_format_float(global_means[field]) for field in data_fields]
            )
        else:
            writer.writerow(
                ["global", "ALL", len(merged_rows)]
                + [_format_float(global_means[field]) for field in data_fields]
            )


def _collect_tasks(directory: str, pattern: re.Pattern) -> dict:
    tasks: dict = defaultdict(list)
    if not directory or not os.path.isdir(directory):
        return tasks
    for fname in sorted(os.listdir(directory)):
        match = pattern.match(fname)
        if match:
            tasks[match.group(1)].append(os.path.join(directory, fname))
    return tasks


def _merge_group(label: str, tasks: dict, output_dir: str, id_fields: Tuple[str, ...]) -> None:
    if not tasks:
        print(f"No {label} rank CSV files found")
        return
    for task_name, files in sorted(tasks.items()):
        print(f"\n-- {label}: {task_name} ({len(files)} rank file(s)) --")
        merge_task_generic(task_name, files, output_dir, label, id_fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_dir", default=None,
                        help="Dir with *_rank*_grad_stats.csv and *_rank*_expert_stats.csv")
    parser.add_argument("--attn_dir", default=None,
                        help="Dir with *_rank*_attn_stats.csv")
    parser.add_argument("--output_dir", default="analysis/merged")
    parser.add_argument("--input_dir", default=None,
                        help="Legacy: same as --grad_dir")
    args = parser.parse_args()

    grad_dir = args.grad_dir or args.input_dir or "analysis/gradient_dominance"
    attn_dir = args.attn_dir

    grad_tasks = _collect_tasks(
        grad_dir, re.compile(r"^(.+)_rank\d+_grad_stats\.csv$"))
    expert_tasks = _collect_tasks(
        grad_dir, re.compile(r"^(.+)_rank\d+_expert_stats\.csv$"))
    attn_tasks = _collect_tasks(
        attn_dir, re.compile(r"^(.+)_rank\d+_attn_stats\.csv$")) if attn_dir else {}

    _merge_group("grad", grad_tasks, args.output_dir, BASE_ID_FIELDS)
    _merge_group("expert", expert_tasks, args.output_dir, EXPERT_ID_FIELDS)
    if attn_dir:
        _merge_group("attn", attn_tasks, args.output_dir, BASE_ID_FIELDS)


if __name__ == "__main__":
    main()
