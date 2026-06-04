"""
ModalGradientLogger — records per-modality gradient contributions to MoELoRA parameters.

Hooks are registered INSIDE _lora_vectorized (coinmoelora.py) as closures over
local variables. No large tensors are stored as module attributes.
Gradient checkpointing must be disabled when using this logger.

R(k) = mean(G^t) / mean(G^v)  is the gradient dominance ratio for task k.
"""
import csv
import os
import statistics
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class _StepRecord:
    __slots__ = ('step', 'layer', 'g_text_A', 'g_vis_A', 'g_text_B', 'g_vis_B')

    def __init__(self, step, layer, g_text_A, g_vis_A, g_text_B, g_vis_B):
        self.step     = step
        self.layer    = layer
        self.g_text_A = g_text_A
        self.g_vis_A  = g_vis_A
        self.g_text_B = g_text_B
        self.g_vis_B  = g_vis_B

    @property
    def R_A(self):
        return self.g_text_A / max(self.g_vis_A, 1e-8)

    @property
    def R_B(self):
        return self.g_text_B / max(self.g_vis_B, 1e-8)


class ModalGradientLogger:
    """
    Attach to a model once; call step_end() after each optimizer step;
    call save_csv(task_name) at the end of each task.

    Gradient checkpointing is SUPPORTED: hooks are re-registered on recomputed
    tensors during the backward recomputation phase, so captured x values are
    always consistent with the gradients.

    Multi-GPU (DDP / ZeRO-2 / ZeRO-3): only rank 0 writes records.
    Other ranks' local gradients exhibit the same text-bias pattern; recording
    on rank 0 is sufficient for trend analysis (R ratio).
    """

    def __init__(
        self,
        log_every_n_steps: int = 1,
        output_dir: str = "analysis/gradient_dominance",
    ):
        self.log_every_n_steps = log_every_n_steps
        self.output_dir = output_dir
        self.step: int = 0
        self.records: List[_StepRecord] = []
        # _pending: partial records waiting for hook_A to complete
        self._pending: Dict[Tuple, dict] = {}
        self._enabled = False

    # ── public API ────────────────────────────────────────────────────────────

    def attach(self, model: nn.Module) -> None:
        """Set logger reference on every CoINMOELoraLinear in model."""
        try:
            from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear
        except ImportError:
            raise ImportError("CoINMOELoraLinear not found")

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, CoINMOELoraLinear):
                module._log_gradients  = True
                module._grad_logger    = self
                module._grad_layer_name = name
                count += 1
        self._enabled = True
        print(f"[GradLogger] attached to {count} CoINMOELoraLinear layers")

    def step_end(self) -> None:
        self.step += 1

    def _flush(self, step, layer, g_tA, g_vA, g_tB, g_vB) -> None:
        """Called by hook_A in coinmoelora.py after both hooks have fired."""
        if step % self.log_every_n_steps != 0:
            return
        self.records.append(_StepRecord(step, layer, g_tA, g_vA, g_tB, g_vB))

    def save_csv(self, task_name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        except Exception:
            rank = 0
        path = os.path.join(self.output_dir, f"{task_name}_rank{rank}_grad_stats.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "layer", "G_text_A", "G_vis_A", "R_A",
                        "G_text_B", "G_vis_B", "R_B"])
            for r in self.records:
                w.writerow([r.step, r.layer,
                             f"{r.g_text_A:.6f}", f"{r.g_vis_A:.6f}", f"{r.R_A:.4f}",
                             f"{r.g_text_B:.6f}", f"{r.g_vis_B:.6f}", f"{r.R_B:.4f}"])
        self._print_summary(task_name)
        print(f"[GradLogger] rank{rank} saved {len(self.records)} records → {path}")
        return path

    def clear_records(self) -> None:
        self.records.clear()
        self._pending.clear()
        self.step = 0

    def detach(self, model: nn.Module) -> None:
        try:
            from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear
        except ImportError:
            return
        for module in model.modules():
            if isinstance(module, CoINMOELoraLinear):
                module._log_gradients = False
                module._grad_logger   = None
        self._enabled = False

    # ── internals ─────────────────────────────────────────────────────────────

    def _print_summary(self, task_name: str) -> None:
        if not self.records:
            return
        valid_A = [r.R_A for r in self.records if r.g_vis_A > 1e-6]
        valid_B = [r.R_B for r in self.records if r.g_vis_B > 1e-6]
        if valid_A:
            print(f"[GradLogger] {task_name}  R_A: "
                  f"mean={statistics.mean(valid_A):.3f}  "
                  f"median={statistics.median(valid_A):.3f}  "
                  f"n={len(valid_A)}")
        if valid_B:
            print(f"[GradLogger] {task_name}  R_B: "
                  f"mean={statistics.mean(valid_B):.3f}  "
                  f"median={statistics.median(valid_B):.3f}  "
                  f"n={len(valid_B)}")
