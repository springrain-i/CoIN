#!/usr/bin/env python3
"""Summarize one forward projector-swap evaluation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELD_NAMES = (
    "run_id",
    "projector_task_id",
    "projector_task_name",
    "final_task_id",
    "eval_task_id",
    "eval_task_name",
    "mode",
    "batch_size",
    "accuracy",
    "expected_predictions",
    "actual_predictions",
    "checkpoint_path",
    "result_dir",
    "log_path",
    "status",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.csv"
    manifest_path = run_dir / "run_manifest.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELD_NAMES:
            raise ValueError(f"Unexpected metrics columns: {reader.fieldnames}")
        rows = list(reader)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    rows.sort(key=lambda row: int(row["eval_task_id"]))
    seen = set()
    for row in rows:
        task_id = int(row["eval_task_id"])
        if task_id in seen:
            raise ValueError(f"Duplicate eval task in metrics: T{task_id}")
        seen.add(task_id)
        if row["mode"] != "all" or int(row["batch_size"]) != 4:
            raise ValueError(
                f"Protocol drift in T{task_id}: mode={row['mode']}, bs={row['batch_size']}"
            )
        if row["expected_predictions"] != row["actual_predictions"]:
            raise ValueError(f"Prediction count mismatch remains in T{task_id}")
        if row["status"] != "complete":
            raise ValueError(f"Non-complete metrics row for T{task_id}: {row['status']}")
        if task_id != int(manifest["early_task_id"]):
            raise ValueError(
                f"Non-diagonal pair: projector T{manifest['early_task_id']} -> eval T{task_id}"
            )

    summary_lines = [
        "# Forward standard-LoRA projector-swap evaluation",
        "",
        f"- Run: `{manifest['run_id']}`",
        (
            f"- Projector: T{manifest['early_task_id']} "
            f"{manifest['early_task_name']} (immediately after training)"
        ),
        "- Final standard-LoRA adapter/config: T8 OCRVQA",
        f"- Hybrid checkpoint: `{manifest['hybrid_checkpoint']}`",
        "- Mode: `all`",
        "- Eval batch size: `4`",
        "- SDPA patch: enabled",
        "",
        "| Eval task | Accuracy | Predictions | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        summary_lines.append(
            f"| T{row['eval_task_id']} {row['eval_task_name']} | "
            f"{float(row['accuracy']):.6f} | {row['actual_predictions']} | "
            f"`{row['result_dir']}` |"
        )
    expected = list(
        range(int(manifest["start_eval_task"]), int(manifest["end_eval_task"]) + 1)
    )
    missing = [task_id for task_id in expected if task_id not in seen]
    if missing:
        summary_lines.extend(
            ["", "Incomplete run: missing eval tasks " + ", ".join(f"T{x}" for x in missing) + "."]
        )
    else:
        summary_lines.extend(["", "The requested diagonal eval pair completed and passed validation."])

    (run_dir / "summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    result_paths = {
        f"T{row['eval_task_id']}_{row['eval_task_name']}": row["result_dir"]
        for row in rows
    }
    (run_dir / "result_paths.json").write_text(
        json.dumps(result_paths, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(run_dir / "summary.md")


if __name__ == "__main__":
    main()
