"""
ModalGradientLogger — records per-modality gradient contributions to MoELoRA parameters.

Three metrics per layer per logged step (all EXACT):
  W_A:  G^t_A[n,r,d] = Σ_{b,t∈text} g_out_A[b,t,n,r] * lora_x[b,t,d]
  W_B:  G^t_B[n,d,r] = Σ_{b,t∈text} g_out_B[b,t,n,d] * out_A[b,t,n,r]
  ΔW:   G^t_dW[o,i]  = Σ_{b,t∈text} g_lora_out[b,t,o] * lora_x[b,t,i]   shape (d_out,d_in)

R variants (text/visual ratio; >1 = text dominant):
  R_A      = ||G^t_A||_F  / ||G^v_A||_F
  R_B      = ||G^t_B||_F  / ||G^v_B||_F
  R_dW     = ||G^t_dW||_F / ||G^v_dW||_F
  R_A_tok  = (||G^t_A||_F  / n_text) / (||G^v_A||_F  / n_vis)   per-token normalised
  R_B_tok  = (||G^t_B||_F  / n_text) / (||G^v_B||_F  / n_vis)
  R_dW_tok = (||G^t_dW||_F / n_text) / (||G^v_dW||_F / n_vis)
"""
import csv
import os
import statistics
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class _StepRecord:
    __slots__ = (
        'step', 'layer',
        'g_text_A', 'g_vis_A',
        'g_text_B', 'g_vis_B',
        'g_text_dW', 'g_vis_dW',
        'n_text', 'n_vis',
    )

    def __init__(self, step, layer,
                 g_text_A, g_vis_A,
                 g_text_B, g_vis_B,
                 g_text_dW, g_vis_dW,
                 n_text, n_vis):
        self.step       = step
        self.layer      = layer
        self.g_text_A   = g_text_A
        self.g_vis_A    = g_vis_A
        self.g_text_B   = g_text_B
        self.g_vis_B    = g_vis_B
        self.g_text_dW  = g_text_dW
        self.g_vis_dW   = g_vis_dW
        self.n_text     = n_text
        self.n_vis      = n_vis

    @property
    def R_A(self):
        return self.g_text_A / max(self.g_vis_A, 1e-8)

    @property
    def R_B(self):
        return self.g_text_B / max(self.g_vis_B, 1e-8)

    @property
    def R_B_tok(self):
        if self.n_text == 0 or self.n_vis == 0:
            return float('inf')
        return (self.g_text_B / self.n_text) / max(self.g_vis_B / self.n_vis, 1e-12)

    @property
    def R_dW(self):
        return self.g_text_dW / max(self.g_vis_dW, 1e-8)

    @property
    def R_A_tok(self):
        if self.n_text == 0 or self.n_vis == 0:
            return float('inf')
        return (self.g_text_A / self.n_text) / max(self.g_vis_A / self.n_vis, 1e-12)

    @property
    def R_dW_tok(self):
        if self.n_text == 0 or self.n_vis == 0:
            return float('inf')
        return (self.g_text_dW / self.n_text) / max(self.g_vis_dW / self.n_vis, 1e-12)


class ModalGradientLogger:
    """
    Attach to a model once; call step_end() after each optimizer step;
    call save_csv(task_name) at the end of each task.

    Gradient checkpointing SUPPORTED: hooks are re-registered on recomputed
    tensors during backward recomputation, so captured x values are consistent.

    Multi-GPU: each rank writes its own {task}_rank{N}_grad_stats.csv.
    Merge files downstream for combined analysis.
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
        self._pending:    Dict[Tuple, dict] = {}   # hook_B results
        self._pending_dw: Dict[Tuple, dict] = {}   # hook_lora_out results
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
                module._log_gradients   = True
                module._grad_logger     = self
                module._grad_layer_name = name
                count += 1
        self._enabled = True
        print(f"[GradLogger] attached to {count} CoINMOELoraLinear layers")

    def step_end(self) -> None:
        self.step += 1

    def _flush(self, step, layer,
               g_tA, g_vA,
               g_tB, g_vB,
               g_tdW, g_vdW,
               n_text, n_vis) -> None:
        """Called by hook_A after all three hooks have fired."""
        if step % self.log_every_n_steps != 0:
            return
        self.records.append(_StepRecord(
            step, layer,
            g_tA, g_vA,
            g_tB, g_vB,
            g_tdW, g_vdW,
            n_text, n_vis,
        ))

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
            w.writerow([
                "step", "layer",
                "G_text_A", "G_vis_A", "R_A", "R_A_tok",
                "G_text_B", "G_vis_B", "R_B", "R_B_tok",
                "G_text_dW", "G_vis_dW", "R_dW", "R_dW_tok",
                "n_text", "n_vis",
            ])
            for r in self.records:
                w.writerow([
                    r.step, r.layer,
                    f"{r.g_text_A:.6f}",  f"{r.g_vis_A:.6f}",
                    f"{r.R_A:.4f}",       f"{r.R_A_tok:.4f}",
                    f"{r.g_text_B:.6f}",  f"{r.g_vis_B:.6f}",
                    f"{r.R_B:.4f}",       f"{r.R_B_tok:.4f}",
                    f"{r.g_text_dW:.6f}", f"{r.g_vis_dW:.6f}",
                    f"{r.R_dW:.4f}",      f"{r.R_dW_tok:.4f}",
                    r.n_text, r.n_vis,
                ])
        self._save_summary_csv(task_name, rank)
        self._print_summary(task_name)
        print(f"[GradLogger] rank{rank} saved {len(self.records)} records → {path}")
        return path

    def _save_summary_csv(self, task_name: str, rank: int) -> None:
        """Write per-layer and global summary CSV alongside the main stats file.

        Summary rows are computed from raw G values (not R), so the ratio is
        derived from mean(G_text)/mean(G_vis) — consistent with multi-rank merging.
        """
        if not self.records:
            return

        # Collect per-layer raw G values
        from collections import defaultdict
        layer_data: dict = defaultdict(lambda: {
            'G_text_A': [], 'G_vis_A': [],
            'G_text_B': [], 'G_vis_B': [],
            'G_text_dW': [], 'G_vis_dW': [],
            'n_text': [], 'n_vis': [],
        })
        for r in self.records:
            d = layer_data[r.layer]
            d['G_text_A'].append(r.g_text_A);  d['G_vis_A'].append(r.g_vis_A)
            d['G_text_B'].append(r.g_text_B);  d['G_vis_B'].append(r.g_vis_B)
            d['G_text_dW'].append(r.g_text_dW); d['G_vis_dW'].append(r.g_vis_dW)
            d['n_text'].append(r.n_text);       d['n_vis'].append(r.n_vis)

        def _ratio(t_vals, v_vals):
            mt = statistics.mean(t_vals)
            mv = statistics.mean(v_vals)
            return mt / max(mv, 1e-8)

        def _ratio_tok(t_vals, v_vals, nt_vals, nv_vals):
            pairs = [(t/max(nt,1), v/max(nv,1))
                     for t, v, nt, nv in zip(t_vals, v_vals, nt_vals, nv_vals)
                     if nt > 0 and nv > 0 and v > 1e-8]
            if not pairs:
                return float('inf')
            return statistics.mean(t/max(v,1e-12) for t,v in pairs)

        summary_path = os.path.join(
            self.output_dir, f"{task_name}_rank{rank}_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "scope", "layer",
                "mean_G_text_A", "mean_G_vis_A", "R_A", "R_A_tok",
                "mean_G_text_B", "mean_G_vis_B", "R_B", "R_B_tok",
                "mean_G_text_dW", "mean_G_vis_dW", "R_dW", "R_dW_tok",
                "n_steps",
            ])
            # Per-layer rows
            all_Gt_A, all_Gv_A = [], []
            all_Gt_B, all_Gv_B = [], []
            all_Gt_dW, all_Gv_dW = [], []
            all_nt, all_nv = [], []
            for layer, d in sorted(layer_data.items()):
                n = len(d['G_text_A'])
                mt_A  = statistics.mean(d['G_text_A']);  mv_A  = statistics.mean(d['G_vis_A'])
                mt_B  = statistics.mean(d['G_text_B']);  mv_B  = statistics.mean(d['G_vis_B'])
                mt_dW = statistics.mean(d['G_text_dW']); mv_dW = statistics.mean(d['G_vis_dW'])
                R_A    = mt_A  / max(mv_A,  1e-8)
                R_B    = mt_B  / max(mv_B,  1e-8)
                R_dW   = mt_dW / max(mv_dW, 1e-8)
                R_Atk  = _ratio_tok(d['G_text_A'],  d['G_vis_A'],  d['n_text'], d['n_vis'])
                R_Btk  = _ratio_tok(d['G_text_B'],  d['G_vis_B'],  d['n_text'], d['n_vis'])
                R_dWtk = _ratio_tok(d['G_text_dW'], d['G_vis_dW'], d['n_text'], d['n_vis'])
                w.writerow([
                    "layer", layer,
                    f"{mt_A:.6f}", f"{mv_A:.6f}", f"{R_A:.4f}", f"{R_Atk:.4f}",
                    f"{mt_B:.6f}", f"{mv_B:.6f}", f"{R_B:.4f}", f"{R_Btk:.4f}",
                    f"{mt_dW:.6f}", f"{mv_dW:.6f}", f"{R_dW:.4f}", f"{R_dWtk:.4f}",
                    n,
                ])
                all_Gt_A.extend(d['G_text_A']);  all_Gv_A.extend(d['G_vis_A'])
                all_Gt_B.extend(d['G_text_B']);  all_Gv_B.extend(d['G_vis_B'])
                all_Gt_dW.extend(d['G_text_dW']); all_Gv_dW.extend(d['G_vis_dW'])
                all_nt.extend(d['n_text']);       all_nv.extend(d['n_vis'])

            # Global summary row
            gmt_A  = statistics.mean(all_Gt_A);  gmv_A  = statistics.mean(all_Gv_A)
            gmt_B  = statistics.mean(all_Gt_B);  gmv_B  = statistics.mean(all_Gv_B)
            gmt_dW = statistics.mean(all_Gt_dW); gmv_dW = statistics.mean(all_Gv_dW)
            gR_A    = gmt_A  / max(gmv_A,  1e-8)
            gR_B    = gmt_B  / max(gmv_B,  1e-8)
            gR_dW   = gmt_dW / max(gmv_dW, 1e-8)
            gR_Atk  = _ratio_tok(all_Gt_A,  all_Gv_A,  all_nt, all_nv)
            gR_Btk  = _ratio_tok(all_Gt_B,  all_Gv_B,  all_nt, all_nv)
            gR_dWtk = _ratio_tok(all_Gt_dW, all_Gv_dW, all_nt, all_nv)
            w.writerow([
                "global", "ALL",
                f"{gmt_A:.6f}", f"{gmv_A:.6f}", f"{gR_A:.4f}", f"{gR_Atk:.4f}",
                f"{gmt_B:.6f}", f"{gmv_B:.6f}", f"{gR_B:.4f}", f"{gR_Btk:.4f}",
                f"{gmt_dW:.6f}", f"{gmv_dW:.6f}", f"{gR_dW:.4f}", f"{gR_dWtk:.4f}",
                len(self.records),
            ])
        print(f"[GradLogger] rank{rank} summary → {summary_path}")

    def clear_records(self) -> None:
        self.records.clear()
        self._pending.clear()
        self._pending_dw.clear()
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
        valid_A    = [r.R_A     for r in self.records if r.g_vis_A  > 1e-6]
        valid_B    = [r.R_B     for r in self.records if r.g_vis_B  > 1e-6]
        valid_dW   = [r.R_dW    for r in self.records if r.g_vis_dW > 1e-6]
        valid_Atk  = [r.R_A_tok   for r in self.records
                      if r.n_vis > 0 and r.g_vis_A  > 1e-6 and r.R_A_tok  < 1e6]
        valid_Btk  = [r.R_B_tok   for r in self.records
                      if r.n_vis > 0 and r.g_vis_B  > 1e-6 and r.R_B_tok  < 1e6]
        valid_dWtk = [r.R_dW_tok  for r in self.records
                      if r.n_vis > 0 and r.g_vis_dW > 1e-6 and r.R_dW_tok < 1e6]

        def _fmt(vals, label):
            if not vals:
                return
            print(f"[GradLogger] {task_name}  {label:<12}"
                  f"mean={statistics.mean(vals):.3f}  "
                  f"median={statistics.median(vals):.3f}  "
                  f"n={len(vals)}")

        _fmt(valid_A,    "R_A:")
        _fmt(valid_B,    "R_B:")
        _fmt(valid_dW,   "R_dW:")
        _fmt(valid_Atk,  "R_A/tok:")
        _fmt(valid_Btk,  "R_B/tok:")
        _fmt(valid_dWtk, "R_dW/tok:")
