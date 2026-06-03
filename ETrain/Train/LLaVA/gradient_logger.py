"""
ModalGradientLogger — records per-modality gradient contributions to MoELoRA parameters.

For each CoINMOELoraLinear in the model, hooks into the backward pass to compute:
  G^t_A = ||∂L/∂W_A contribution from text tokens||_F
  G^v_A = ||∂L/∂W_A contribution from visual tokens||_F
  G^t_B = ||∂L/∂W_B contribution from text tokens||_F
  G^v_B = ||∂L/∂W_B contribution from visual tokens||_F

R(k) = mean(G^t) / mean(G^v) is the gradient dominance ratio for task k.

Mathematical basis (see gradient_decomp_explainer.html for derivation):
  ∂L/∂W_A = G_out_A^T @ lora_x
           = G_out_A^T @ lora_x_text   +   G_out_A^T @ lora_x_visual
           =      G^t_A               +        G^v_A

For vectorized MoE path:
  out_A = einsum('bti,nri->btnr', lora_x, A_stack)   # (B,T,N,r_per)
  G^t_A = einsum('btnr,bti->nri', grad_out_A, lora_x_text)
  G^v_A = einsum('btnr,bti->nri', grad_out_A, lora_x_visual)
"""
import csv
import os
import statistics
from typing import List, Optional

import torch
import torch.nn as nn


class _StepRecord:
    __slots__ = ('step', 'layer', 'g_text_A', 'g_vis_A', 'g_text_B', 'g_vis_B')

    def __init__(self, step, layer, g_text_A, g_vis_A, g_text_B, g_vis_B):
        self.step = step
        self.layer = layer
        self.g_text_A = g_text_A
        self.g_vis_A = g_vis_A
        self.g_text_B = g_text_B
        self.g_vis_B = g_vis_B

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
        self._fwd_hooks = []
        self._enabled = False

    # ── public API ────────────────────────────────────────────────────────────

    def attach(self, model: nn.Module) -> None:
        """Register forward hooks on every CoINMOELoraLinear in model."""
        try:
            from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear
        except ImportError:
            raise ImportError("CoINMOELoraLinear not found — run from CoIN repo root")

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, CoINMOELoraLinear):
                h = module.register_forward_hook(self._make_fwd_hook(name))
                self._fwd_hooks.append(h)
                count += 1
        print(f"[GradLogger] attached to {count} CoINMOELoraLinear layers")
        self._enabled = True

    def step_end(self) -> None:
        self.step += 1

    def save_csv(self, task_name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{task_name}_grad_stats.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "layer", "G_text_A", "G_vis_A", "R_A",
                        "G_text_B", "G_vis_B", "R_B"])
            for r in self.records:
                w.writerow([r.step, r.layer,
                             f"{r.g_text_A:.6f}", f"{r.g_vis_A:.6f}", f"{r.R_A:.4f}",
                             f"{r.g_text_B:.6f}", f"{r.g_vis_B:.6f}", f"{r.R_B:.4f}"])

        self._print_summary(task_name)
        print(f"[GradLogger] saved {len(self.records)} records → {path}")
        return path

    def clear_records(self) -> None:
        self.records.clear()
        self.step = 0

    def remove_hooks(self) -> None:
        for h in self._fwd_hooks:
            h.remove()
        self._fwd_hooks.clear()
        self._enabled = False

    # ── internals ─────────────────────────────────────────────────────────────

    def _make_fwd_hook(self, layer_name: str):
        logger = self

        def fwd_hook(module, inputs, output):
            if not logger._enabled:
                return
            if logger.step % logger.log_every_n_steps != 0:
                return

            # These attributes are written by _lora_vectorized_logged() in coinmoelora.py
            out_A = getattr(module, "_log_out_A", None)
            out_B = getattr(module, "_log_out_B", None)
            lora_x = getattr(module, "_log_lora_x", None)
            raw_mask = getattr(module, "_log_raw_mask", None)

            if any(v is None for v in (out_A, out_B, lora_x, raw_mask)):
                return
            if not out_A.requires_grad or not out_B.requires_grad:
                return

            step = logger.step
            text_mask = (raw_mask == 2).float()   # (B, T)
            vis_mask  = (raw_mask == 1).float()   # (B, T)

            # Detach inputs used inside hooks to avoid keeping full graph alive
            lx_text  = (lora_x.detach() * text_mask.unsqueeze(-1))   # (B,T,d_in)
            lx_vis   = (lora_x.detach() * vis_mask.unsqueeze(-1))
            oA_text  = (out_A.detach() * text_mask[:, :, None, None]) # (B,T,N,r)
            oA_vis   = (out_A.detach() * vis_mask[:, :, None, None])

            tmp = {}   # shared between hook_A and hook_B

            # Backward order: hook_B fires BEFORE hook_A (out_B depends on out_A).
            # Strategy: hook_B creates the record; hook_A updates it afterwards.

            def hook_B(grad_out_B):
                # grad_out_B: (B, T, N, d_out)  — fires first during backward
                with torch.no_grad():
                    G_t = torch.einsum("btno,btnr->nor", grad_out_B, oA_text)
                    G_v = torch.einsum("btno,btnr->nor", grad_out_B, oA_vis)
                    logger.records.append(_StepRecord(
                        step=step,
                        layer=layer_name,
                        g_text_A=0.0,           # placeholder — hook_A will fill this
                        g_vis_A=0.0,
                        g_text_B=G_t.norm().item(),
                        g_vis_B=G_v.norm().item(),
                    ))

            def hook_A(grad_out_A):
                # grad_out_A: (B, T, N, r_per)  — fires second during backward
                with torch.no_grad():
                    G_t = torch.einsum("btnr,bti->nri", grad_out_A, lx_text)
                    G_v = torch.einsum("btnr,bti->nri", grad_out_A, lx_vis)
                    # Update the record just created by hook_B
                    for r in reversed(logger.records):
                        if r.step == step and r.layer == layer_name:
                            r.g_text_A = G_t.norm().item()
                            r.g_vis_A  = G_v.norm().item()
                            break

            out_A.register_hook(hook_A)
            out_B.register_hook(hook_B)

        return fwd_hook

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
