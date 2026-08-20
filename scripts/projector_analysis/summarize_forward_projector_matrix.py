#!/usr/bin/env python3
"""Combine T1-T7 projector-swap runs into a diagonal projector/eval summary."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--run-timestamp", required=True)
    parser.add_argument("--start-early-task", type=int, default=1, choices=range(1, 8))
    parser.add_argument("--end-early-task", type=int, default=7, choices=range(1, 8))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_early_task > args.end_early_task:
        raise ValueError("start early task must not exceed end early task")

    metrics_root = args.metrics_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, str]] = []
    source_files = []

    for early_id in range(args.start_early_task, args.end_early_task + 1):
        early_name = TASK_NAMES[early_id]
        arm_id = f"early_T{early_id}_{early_name}__final_T8_OCRVQA"
        metrics_path = metrics_root / arm_id / args.run_timestamp / "metrics.csv"
        source_files.append(str(metrics_path))
        if not metrics_path.is_file():
            if args.allow_incomplete:
                continue
            raise FileNotFoundError(metrics_path)
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if int(row["projector_task_id"]) != early_id:
                raise ValueError(f"Projector ID mismatch in {metrics_path}")
            if int(row["eval_task_id"]) != early_id:
                raise ValueError(
                    f"Non-diagonal pair in {metrics_path}: "
                    f"projector T{early_id} -> eval T{row['eval_task_id']}"
                )
            if row["mode"] != "all" or int(row["batch_size"]) != 4:
                raise ValueError(f"Protocol drift in {metrics_path}")
            if row["status"] != "complete":
                raise ValueError(f"Incomplete row in {metrics_path}")
            if row["expected_predictions"] != row["actual_predictions"]:
                raise ValueError(f"Prediction-count mismatch in {metrics_path}")
            combined_rows.append(row)

    values: dict[tuple[int, int], str] = {}
    for row in combined_rows:
        pair = (int(row["projector_task_id"]), int(row["eval_task_id"]))
        if pair in values:
            raise ValueError(f"Duplicate projector/eval pair: {pair}")
        values[pair] = row["accuracy"]

    expected_pairs = {
        (task_id, task_id)
        for task_id in range(args.start_early_task, args.end_early_task + 1)
    }
    missing_pairs = sorted(expected_pairs - set(values))
    if missing_pairs and not args.allow_incomplete:
        formatted = ", ".join(f"projectorT{p}/evalT{e}" for p, e in missing_pairs)
        raise RuntimeError(f"Missing diagonal entries: {formatted}")

    combined_path = output_dir / "combined_metrics.csv"
    if combined_rows:
        fieldnames = list(combined_rows[0])
        with combined_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                sorted(
                    combined_rows,
                    key=lambda row: (
                        int(row["projector_task_id"]), int(row["eval_task_id"])
                    ),
                )
            )
    else:
        combined_path.write_text("", encoding="utf-8")

    diagonal_fields = [
        "task_id",
        "task_name",
        "projector_task_id",
        "eval_task_id",
        "accuracy",
    ]
    diagonal_rows = []
    for task_id in range(args.start_early_task, args.end_early_task + 1):
        diagonal_rows.append(
            {
                "task_id": task_id,
                "task_name": TASK_NAMES[task_id],
                "projector_task_id": task_id,
                "eval_task_id": task_id,
                "accuracy": values.get((task_id, task_id), ""),
            }
        )
    diagonal_path = output_dir / "projector_eval_diagonal.csv"
    with diagonal_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagonal_fields)
        writer.writeheader()
        writer.writerows(diagonal_rows)

    summary = [
        "# Forward early-projector diagonal sweep",
        "",
        "- Final adapter/router/config: T8 OCRVQA",
        f"- Early projector arms: T{args.start_early_task}–T{args.end_early_task}",
        "- T8 final projector arm: not re-evaluated",
        "- Protocol: projector Tn is evaluated only on task Tn",
        "- Execution: one diagonal pair at a time, using all 8 GPUs (8 chunks)",
        "- Mode: `all`",
        "- Eval batch size: `4` per GPU worker",
        "",
        "| Pair | Task | Accuracy |",
        "| --- | --- | ---: |",
    ]
    for task_id in range(args.start_early_task, args.end_early_task + 1):
        accuracy = values.get((task_id, task_id))
        accuracy_text = f"{float(accuracy):.6f}" if accuracy is not None else "—"
        summary.append(
            f"| projector T{task_id} → eval T{task_id} | "
            f"{TASK_NAMES[task_id]} | {accuracy_text} |"
        )
    summary.extend(
        [
            "",
            (
                "All requested diagonal projector/eval pairs completed and passed validation."
                if not missing_pairs
                else f"Incomplete: {len(missing_pairs)} projector/eval pairs are missing."
            ),
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment": "forward_projector_swap_diagonal_sweep",
        "run_timestamp": args.run_timestamp,
        "start_early_task": args.start_early_task,
        "end_early_task": args.end_early_task,
        "final_task_id": 8,
        "final_task_name": "OCRVQA",
        "final_projector_re_evaluated": False,
        "mode": "all",
        "eval_batch_size": 4,
        "gpus_per_eval_task": 8,
        "expected_pairs": len(expected_pairs),
        "completed_pairs": len(values),
        "missing_pairs": [
            {"projector_task_id": projector_id, "eval_task_id": eval_id}
            for projector_id, eval_id in missing_pairs
        ],
        "source_metrics": source_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(diagonal_path)


if __name__ == "__main__":
    main()
