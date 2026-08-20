#!/usr/bin/env python3
"""Build an eval-only checkpoint with a selected early mm_projector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


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
FINAL_TASK_ID = 8
EXPECTED_PROJECTOR_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")
CONFIG_KEYS = (
    "hidden_size",
    "mm_hidden_size",
    "mm_projector_type",
    "mm_vision_select_layer",
    "mm_vision_tower",
    "mm_use_im_start_end",
    "mm_use_im_patch_token",
)


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"Expected a non-empty state dict: {path}")
    if not all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()):
        raise RuntimeError(f"Checkpoint is not a tensor state dict: {path}")
    return value


def projector_entries(
    state_dict: dict[str, torch.Tensor], checkpoint_path: Path
) -> dict[str, tuple[str, torch.Tensor]]:
    marker = "mm_projector."
    entries: dict[str, tuple[str, torch.Tensor]] = {}
    for full_key, tensor in state_dict.items():
        if marker not in full_key:
            continue
        suffix = full_key.split(marker, 1)[1]
        if suffix in entries:
            raise RuntimeError(
                f"Duplicate normalized projector key {suffix!r} in {checkpoint_path}"
            )
        entries[suffix] = (full_key, tensor)

    actual = set(entries)
    expected = set(EXPECTED_PROJECTOR_KEYS)
    if actual != expected:
        raise RuntimeError(
            f"Unexpected projector keys in {checkpoint_path}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return entries


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def validate_configs(early_config: Path, final_config: Path) -> dict[str, Any]:
    early = load_json(early_config)
    final = load_json(final_config)
    mismatches = {
        key: {"early": early.get(key), "final": final.get(key)}
        for key in CONFIG_KEYS
        if early.get(key) != final.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Early/final multimodal configs are incompatible: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    if final.get("mm_projector_type") != "mlp2x_gelu":
        raise RuntimeError(
            f"Expected mm_projector_type=mlp2x_gelu, got {final.get('mm_projector_type')!r}"
        )
    return {key: final.get(key) for key in CONFIG_KEYS}


def build_hybrid_state(
    early_state: dict[str, torch.Tensor],
    final_state: dict[str, torch.Tensor],
    early_path: Path,
    final_path: Path,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    early_projector = projector_entries(early_state, early_path)
    final_projector = projector_entries(final_state, final_path)
    hybrid = {key: tensor.detach().cpu().clone() for key, tensor in final_state.items()}
    tensor_manifest = []

    for suffix in EXPECTED_PROJECTOR_KEYS:
        _, early_tensor = early_projector[suffix]
        final_full_key, final_tensor = final_projector[suffix]
        if early_tensor.shape != final_tensor.shape:
            raise RuntimeError(
                f"Shape mismatch for {suffix}: early={tuple(early_tensor.shape)}, "
                f"final={tuple(final_tensor.shape)}"
            )
        if early_tensor.dtype != final_tensor.dtype:
            raise RuntimeError(
                f"Dtype mismatch for {suffix}: early={early_tensor.dtype}, final={final_tensor.dtype}"
            )
        hybrid[final_full_key] = early_tensor.detach().cpu().clone()
        tensor_manifest.append(
            {
                "normalized_key": suffix,
                "stored_key": final_full_key,
                "shape": list(early_tensor.shape),
                "dtype": str(early_tensor.dtype),
            }
        )

    return hybrid, tensor_manifest


def verify_hybrid(
    hybrid_path: Path,
    early_path: Path,
    final_path: Path,
) -> None:
    hybrid = load_state_dict(hybrid_path)
    early = load_state_dict(early_path)
    final = load_state_dict(final_path)
    hybrid_projector = projector_entries(hybrid, hybrid_path)
    early_projector = projector_entries(early, early_path)
    final_projector_keys = {
        full_key for full_key, _ in projector_entries(final, final_path).values()
    }

    if set(hybrid) != set(final):
        raise RuntimeError("Hybrid non-LoRA key set differs from final checkpoint")

    for suffix in EXPECTED_PROJECTOR_KEYS:
        if not torch.equal(hybrid_projector[suffix][1], early_projector[suffix][1]):
            raise RuntimeError(f"Hybrid projector tensor does not equal early tensor: {suffix}")

    for key, final_tensor in final.items():
        if key in final_projector_keys:
            continue
        if not torch.equal(hybrid[key], final_tensor):
            raise RuntimeError(f"Non-projector tensor changed unexpectedly: {key}")


def required_file(checkpoint: Path, filename: str) -> Path:
    path = checkpoint / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty checkpoint file: {path}")
    return path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-task-id", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--materialize-adapter",
        action="store_true",
        help="Copy the final adapter instead of creating a read-only absolute symlink.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and reuse an existing matching derived checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    checkpoint_root = (
        args.checkpoint_root
        or repo_root / "backup/legacy_llava_20260818/checkpoints/LLaVA/CoIN"
    ).resolve()
    output_root = (
        args.output_root or repo_root / "checkpoints/LLaVA/CoIN_projector_swap"
    ).resolve()

    early_task_id = args.early_task_id
    early_task_name = TASK_NAMES[early_task_id]
    final_task_name = TASK_NAMES[FINAL_TASK_ID]
    early_checkpoint = checkpoint_root / f"{early_task_name}_llava_MOE_lora"
    final_checkpoint = checkpoint_root / f"{final_task_name}_llava_MOE_lora"
    arm_id = (
        f"early_T{early_task_id}_{early_task_name}__"
        f"final_T{FINAL_TASK_ID}_{final_task_name}"
    )
    arm_dir = output_root / arm_id
    destination = arm_dir / f"{final_task_name}_llava_MOE_lora"

    early_non_lora = required_file(early_checkpoint, "non_lora_trainables.bin")
    final_non_lora = required_file(final_checkpoint, "non_lora_trainables.bin")
    final_adapter = required_file(final_checkpoint, "adapter_model.bin")
    final_adapter_config = required_file(final_checkpoint, "adapter_config.json")
    early_config = required_file(early_checkpoint, "config.json")
    final_config = required_file(final_checkpoint, "config.json")
    multimodal_config = validate_configs(early_config, final_config)

    source_hashes = {
        "early_non_lora_trainables": sha256_file(early_non_lora),
        "final_non_lora_trainables": sha256_file(final_non_lora),
        "final_adapter_model": sha256_file(final_adapter),
        "final_adapter_config": sha256_file(final_adapter_config),
        "final_config": sha256_file(final_config),
    }

    if destination.exists():
        manifest_path = destination / "swap_manifest.json"
        if not args.reuse_existing:
            raise FileExistsError(
                f"Derived checkpoint already exists: {destination}. "
                "Pass --reuse-existing to validate and reuse it."
            )
        manifest = load_json(manifest_path)
        if manifest.get("arm_id") != arm_id or manifest.get("source_sha256") != source_hashes:
            raise RuntimeError(f"Existing derived checkpoint provenance does not match: {destination}")
        required_file(destination, "adapter_model.bin")
        verify_hybrid(destination / "non_lora_trainables.bin", early_non_lora, final_non_lora)
        eprint(f"[ProjectorSwap] Validated existing checkpoint: {destination}")
        print(destination)
        return

    early_state = load_state_dict(early_non_lora)
    final_state = load_state_dict(final_non_lora)
    hybrid_state, tensor_manifest = build_hybrid_state(
        early_state, final_state, early_non_lora, final_non_lora
    )

    arm_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=arm_dir))
    try:
        hybrid_non_lora = temporary / "non_lora_trainables.bin"
        torch.save(hybrid_state, hybrid_non_lora)

        for filename in ("adapter_config.json", "config.json", "trainer_state.json", "README.md"):
            source = final_checkpoint / filename
            if source.is_file():
                shutil.copy2(source, temporary / filename)

        if args.materialize_adapter:
            shutil.copy2(final_adapter, temporary / "adapter_model.bin")
            adapter_storage = "copy"
        else:
            os.symlink(final_adapter.resolve(), temporary / "adapter_model.bin")
            adapter_storage = "absolute_symlink"

        verify_hybrid(hybrid_non_lora, early_non_lora, final_non_lora)

        manifest = {
            "schema_version": 1,
            "experiment": "forward_projector_swap",
            "arm_id": arm_id,
            "eval_only": True,
            "early_task_id": early_task_id,
            "early_task_name": early_task_name,
            "final_task_id": FINAL_TASK_ID,
            "final_task_name": final_task_name,
            "early_checkpoint": str(early_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "hybrid_checkpoint": str(destination),
            "adapter_storage": adapter_storage,
            "projector_tensors": tensor_manifest,
            "preserved_final_non_projector_keys": sorted(
                key for key in final_state if "mm_projector." not in key
            ),
            "multimodal_config": multimodal_config,
            "source_sha256": source_hashes,
            "hybrid_non_lora_sha256": sha256_file(hybrid_non_lora),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
            "git_status_short": git_value(repo_root, "status", "--short"),
        }
        with (temporary / "swap_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    eprint(
        f"[ProjectorSwap] Built {arm_id}: final adapter/config + "
        f"T{early_task_id} {early_task_name} projector"
    )
    eprint(f"[ProjectorSwap] Hybrid checkpoint: {destination}")
    print(destination)


if __name__ == "__main__":
    main()
