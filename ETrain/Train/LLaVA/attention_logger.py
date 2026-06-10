"""
AttentionLogger — records per-modality cross-attention statistics for LlamaAttention layers.

Two attention-budget fractions per (step, layer).  Query positions are restricted to
post-image text tokens (B-1 fix): text tokens that appear AFTER the last visual token,
because only these can causally attend to visual keys.  Pre-image text tokens have
A_tv = 0 by causal constraint and are excluded to avoid polluting the signal.

  A_tt  mean fraction of post-image text query attention on text  keys
  A_tv  mean fraction of post-image text query attention on vis   keys  (key metric)

  A_tt + A_tv ≈ 1  (softmax normalisation; padding keys masked to -inf)

Aggregation across batch elements uses token-count-weighted average (B-2 fix):
each post-image text query token contributes equally, regardless of how many
such tokens a given batch element has.

Information-flow metrics (Wu et al., CoLM 2025, arXiv 2510.02608):

  u_vis^(b,h,q)  = W_O_h @ Σ_{j∈vis}  w[b,h,q,j] · v[b,h,j,:]   [hidden_dim]
  u_text^(b,h,q) = W_O_h @ Σ_{j∈text} w[b,h,q,j] · v[b,h,j,:]   [hidden_dim]
  U_vis  = E_{b,h,q∈ans-tok}[ ‖u_vis^(b,h,q)‖₂ ]
  U_text = E_{b,h,q∈ans-tok}[ ‖u_text^(b,h,q)‖₂ ]
  R_info = U_vis / (U_vis + U_text)

W_O is split per head: W_O_h = o_proj.weight[:, h*d:(h+1)*d], shape [hidden_dim, head_dim].
U values use the same token-count-weighted average as A_tt / A_tv.

Derived ratios:
  R_att_text = A_tt / (A_tt + A_tv)   ≈ A_tt
  R_info     = U_vis / (U_vis + U_text)

Multi-GPU: each rank writes its own CSV; merge with merge_grad_csvs.py.
"""
import csv
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class _AttnStepRecord:
    __slots__ = ("step", "layer", "A_tt", "A_tv", "U_vis", "U_text", "n_post_text")

    def __init__(self, step, layer, A_tt, A_tv, U_vis, U_text, n_post_text):
        self.step        = step
        self.layer       = layer
        self.A_tt        = A_tt
        self.A_tv        = A_tv
        self.U_vis       = U_vis
        self.U_text      = U_text
        self.n_post_text = n_post_text

    @property
    def R_att_text(self):
        return self.A_tt / max(self.A_tt + self.A_tv, 1e-8)

    @property
    def R_info(self):
        return self.U_vis / max(self.U_vis + self.U_text, 1e-8)


class AttentionLogger:
    """
    Attach to model once; receives _record() calls from the SDPA patch on each
    prefill forward; call step_end() after each optimizer step; save_csv() at task end.

    Token mask propagation: monkey-patches _apply_lora_token_mask so that
    LlamaAttention modules also receive _attn_token_mask on each forward — no
    changes to llava_llama.py needed.
    """

    def __init__(
        self,
        log_every_n_steps: int = 1,
        output_dir: str = "analysis/attn_dominance",
    ):
        self.log_every_n_steps = log_every_n_steps
        self.output_dir = output_dir
        self.step: int = 0
        self.records: List[_AttnStepRecord] = []
        self._enabled = False

    # ── public API ────────────────────────────────────────────────────────────

    def attach(self, model: nn.Module) -> None:
        """
        1. Set _log_attn / _attn_logger / _attn_layer_name on each LlamaAttention.
        2. Monkey-patch _apply_lora_token_mask to also push token_mask to attn layers.
        """
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
        except ImportError:
            raise ImportError("LlamaAttention not found in transformers")

        attn_count = 0
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                module._log_attn         = True
                module._attn_logger      = self
                module._attn_layer_name  = name
                module._attn_token_mask  = None
                attn_count += 1

        # Monkey-patch _apply_lora_token_mask to also propagate to attn modules
        orig_apply = model._apply_lora_token_mask.__func__   # unbound method

        logger_ref = self

        def _patched_apply(self_model):
            orig_apply(self_model)
            token_mask = getattr(self_model, "current_lora_mask", None)
            for module in self_model.model.modules():
                if isinstance(module, LlamaAttention):
                    module._attn_token_mask = token_mask

        import types
        model._apply_lora_token_mask = types.MethodType(_patched_apply, model)

        self._enabled = True
        print(f"[AttnLogger] attached to {attn_count} LlamaAttention layers")

    def step_end(self) -> None:
        self.step += 1

    def _record(
        self,
        step: int,
        layer_name: str,
        attn_weights: torch.Tensor,   # [B, H, T_q, T_k]  float32
        token_mask: torch.Tensor,     # [B, T]  int (2=text,1=vis,0=pad)
        value_states: torch.Tensor,   # [B, H, T_k, head_dim]
        o_proj_weight: torch.Tensor,  # [hidden_dim, H*head_dim]
    ) -> None:
        if step % self.log_every_n_steps != 0:
            return

        bsz = attn_weights.shape[0]
        H        = attn_weights.shape[1]
        head_d   = value_states.shape[-1]
        hidden_d = o_proj_weight.shape[0]

        # B-2: weighted sums (weight = n_post_text per element)
        A_tt_wsum   = A_tv_wsum   = 0.0
        U_vis_wsum  = U_text_wsum = 0.0
        n_post_text_total = 0   # total post-image text query tokens across batch
        valid_b = 0

        with torch.no_grad():
            # W_O split into per-head blocks: [H, head_dim, hidden_dim]
            # Non-contiguous view; matmul handles this without an extra copy.
            W_O_T = (o_proj_weight
                     .float()
                     .view(hidden_d, H, head_d)
                     .permute(1, 2, 0))  # [H, head_d, hidden_d]

            for b in range(bsz):
                t_mask = (token_mask[b] == 2)   # [T]  all text positions
                v_mask = (token_mask[b] == 1)   # [T]  all visual positions (key side)
                nv = int(v_mask.sum())

                if nv == 0:
                    continue   # skip pure-text elements (no visual keys to measure)

                # B-1: restrict text queries to positions AFTER the last visual token.
                # Pre-image text tokens have A_tv = 0 by causal masking (they cannot
                # attend to future visual tokens), which would bias the mean toward 0
                # regardless of model behaviour.
                vis_positions = v_mask.nonzero(as_tuple=True)[0]
                last_vis_pos  = int(vis_positions[-1])
                post_img_t_mask = t_mask.clone()
                post_img_t_mask[:last_vis_pos + 1] = False   # exclude pre-image text
                n_post = int(post_img_t_mask.sum())

                if n_post == 0:
                    continue   # image is last token; no answer tokens to measure

                w = attn_weights[b]  # [H, T_q, T_k]

                # Answer token queries only (B-1 fix).
                # sum(-1) over key dimension; mean over (H, T_post) positions.
                # A_tt + A_tv ≈ 1: pad/future keys masked to -inf.
                w_tq = w[:, post_img_t_mask, :]          # [H, T_post, T_k]
                A_tt = w_tq[:, :, t_mask].sum(-1).mean().item()
                A_tv = w_tq[:, :, v_mask].sum(-1).mean().item()

                # --- U_vis / U_text: information-flow metrics ---
                # Per head h, per answer token q:
                #   vis_sum[h,q] = Σ_{j∈vis}  w[h,q,j] · v[h,j,:]  → [head_d]
                #   u_vis[h,q]   = W_O_h @ vis_sum[h,q]              → [hidden_d]
                # U_vis_b = mean over (H, T_post) of ‖u_vis[h,q]‖₂
                v_b = value_states[b].float()  # [H, T_k, head_d]

                vis_sum  = torch.matmul(
                    w_tq[:, :, v_mask].float(),  # [H, T_post, n_vis]
                    v_b[:, v_mask, :],            # [H, n_vis,  head_d]
                )                                 # [H, T_post, head_d]
                text_sum = torch.matmul(
                    w_tq[:, :, t_mask].float(),  # [H, T_post, n_text]
                    v_b[:, t_mask, :],            # [H, n_text, head_d]
                )                                 # [H, T_post, head_d]

                # Apply per-head W_O: [H,T_post,head_d] @ [H,head_d,hidden_d]
                u_vis  = torch.matmul(vis_sum,  W_O_T)  # [H, T_post, hidden_d]
                u_text = torch.matmul(text_sum, W_O_T)  # [H, T_post, hidden_d]

                U_vis_b  = u_vis.norm(dim=-1).mean().item()
                U_text_b = u_text.norm(dim=-1).mean().item()

                del v_b, vis_sum, text_sum, u_vis, u_text

                # B-2: accumulate weighted by token count so each token contributes equally.
                A_tt_wsum  += A_tt   * n_post
                A_tv_wsum  += A_tv   * n_post
                U_vis_wsum  += U_vis_b  * n_post
                U_text_wsum += U_text_b * n_post
                n_post_text_total += n_post
                valid_b += 1

        if valid_b == 0 or n_post_text_total == 0:
            return

        self.records.append(_AttnStepRecord(
            step, layer_name,
            A_tt_wsum  / n_post_text_total,
            A_tv_wsum  / n_post_text_total,
            U_vis_wsum  / n_post_text_total,
            U_text_wsum / n_post_text_total,
            n_post_text_total / valid_b,
        ))

    def save_csv(self, task_name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        except Exception:
            rank = 0

        path = os.path.join(self.output_dir, f"{task_name}_rank{rank}_attn_stats.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "layer",
                "A_tt", "A_tv", "R_att_text",
                "U_vis", "U_text", "R_info",
                "n_post_text",
            ])
            for r in self.records:
                w.writerow([
                    r.step, r.layer,
                    f"{r.A_tt:.6f}", f"{r.A_tv:.6f}", f"{r.R_att_text:.4f}",
                    f"{r.U_vis:.6f}", f"{r.U_text:.6f}", f"{r.R_info:.4f}",
                    r.n_post_text,
                ])

        self._save_summary_csv(task_name, rank)
        print(f"[AttnLogger] rank{rank} saved {len(self.records)} records → {path}")
        return path

    def _save_summary_csv(self, task_name: str, rank: int) -> None:
        if not self.records:
            return
        layer_data: dict = defaultdict(
            lambda: {"A_tt": [], "A_tv": [], "U_vis": [], "U_text": []})
        for r in self.records:
            d = layer_data[r.layer]
            d["A_tt"].append(r.A_tt);   d["A_tv"].append(r.A_tv)
            d["U_vis"].append(r.U_vis); d["U_text"].append(r.U_text)

        summary_path = os.path.join(
            self.output_dir, f"{task_name}_rank{rank}_attn_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "scope", "layer",
                "mean_A_tt", "mean_A_tv", "R_att_text",
                "mean_U_vis", "mean_U_text", "R_info",
                "n_steps",
            ])
            all_tt, all_tv, all_uv, all_ut = [], [], [], []
            for layer, d in sorted(layer_data.items()):
                mt  = statistics.mean(d["A_tt"]);  mv  = statistics.mean(d["A_tv"])
                muv = statistics.mean(d["U_vis"]); mut = statistics.mean(d["U_text"])
                R_text = mt  / max(mt  + mv,  1e-8)
                R_info = muv / max(muv + mut, 1e-8)
                w.writerow([
                    "layer", layer,
                    f"{mt:.6f}",  f"{mv:.6f}",  f"{R_text:.4f}",
                    f"{muv:.6f}", f"{mut:.6f}",  f"{R_info:.4f}",
                    len(d["A_tt"]),
                ])
                all_tt.extend(d["A_tt"]);  all_tv.extend(d["A_tv"])
                all_uv.extend(d["U_vis"]); all_ut.extend(d["U_text"])

            gtt  = statistics.mean(all_tt); gtv  = statistics.mean(all_tv)
            guv  = statistics.mean(all_uv); gut  = statistics.mean(all_ut)
            w.writerow([
                "global", "ALL",
                f"{gtt:.6f}",  f"{gtv:.6f}",  f"{gtt/max(gtt+gtv,1e-8):.4f}",
                f"{guv:.6f}",  f"{gut:.6f}",  f"{guv/max(guv+gut,1e-8):.4f}",
                len(self.records),
            ])
        print(f"[AttnLogger] rank{rank} summary → {summary_path}")

    def clear_records(self) -> None:
        self.records.clear()
        self.step = 0

    def detach(self, model: nn.Module) -> None:
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
        except ImportError:
            return
        for module in model.modules():
            if isinstance(module, LlamaAttention):
                module._log_attn    = False
                module._attn_logger = None
        self._enabled = False
