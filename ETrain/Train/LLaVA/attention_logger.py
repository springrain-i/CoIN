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
from typing import Dict, List, Optional, Set, TextIO, Tuple

import torch
import torch.nn as nn


class _AttnStepRecord:
    __slots__ = ("step", "microbatch", "layer", "A_tt", "A_tv", "U_vis", "U_text", "n_post_text")

    def __init__(self, step, microbatch, layer, A_tt, A_tv, U_vis, U_text, n_post_text):
        self.step        = step
        self.microbatch  = microbatch
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
        task_name: str = "task_unknown",
        total_optimizer_steps: Optional[int] = None,
        step_schedule: str = "interval",
        layer_blocks: Optional[str] = None,
        grad_accum_steps: Optional[int] = None,
        microbatch_sample: Optional[str] = "0.25",
    ):
        self.log_every_n_steps = max(int(log_every_n_steps), 1)
        self.output_dir = output_dir
        self.task_name = task_name
        self.step: int = 0
        self.total_optimizer_steps = total_optimizer_steps
        self.step_schedule = step_schedule
        self.selected_steps = self._build_step_schedule(total_optimizer_steps, step_schedule)
        self.layer_blocks = self._parse_layer_blocks(layer_blocks)
        self.grad_accum_steps = int(grad_accum_steps) if grad_accum_steps not in (None, "") else None
        self.microbatch_sample = microbatch_sample
        self.selected_microbatches = self._build_microbatch_schedule(
            self.grad_accum_steps, microbatch_sample)
        self.records: List[_AttnStepRecord] = []
        self._enabled = False
        self._stream_file: Optional[TextIO] = None
        self._stream_writer = None
        self._stream_path: Optional[str] = None
        self._microbatch_anchor_layer: Optional[str] = None
        self._active_step: Optional[int] = None
        self._microbatch_index = 0
        self._current_microbatch_selected = True
        self._record_calls = 0
        self._record_written = 0
        self._skip_no_visual = 0
        self._skip_no_post_text = 0

    @property
    def current_step(self) -> int:
        return self.step + 1

    @staticmethod
    def _rank() -> int:
        try:
            import torch.distributed as dist
            return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        except Exception:
            return 0

    def _stats_path(self) -> str:
        rank = self._rank()
        return os.path.join(self.output_dir, f"{self.task_name}_rank{rank}_attn_stats.csv")

    @staticmethod
    def _parse_layer_blocks(layer_blocks: Optional[str]) -> Optional[Set[int]]:
        if layer_blocks is None or str(layer_blocks).strip() == "":
            return None
        blocks: Set[int] = set()
        for item in str(layer_blocks).split(','):
            item = item.strip()
            if item:
                blocks.add(int(item))
        return blocks

    @staticmethod
    def _block_id(layer_name: str) -> Optional[int]:
        import re
        match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', layer_name)
        return int(match.group(1)) if match else None

    def should_log_layer(self, layer_name: str) -> bool:
        if self.layer_blocks is None:
            return True
        block = self._block_id(layer_name)
        return block in self.layer_blocks

    @staticmethod
    def _spread_points(start: int, end: int, count: int) -> List[int]:
        if count <= 0 or end < start:
            return []
        if count == 1:
            return [start]
        return [round(start + i * (end - start) / (count - 1)) for i in range(count)]

    @classmethod
    def _build_step_schedule(cls, total_steps: Optional[int], schedule: str) -> Optional[Set[int]]:
        if schedule != "staged" or total_steps is None or total_steps <= 0:
            return None
        total_steps = int(total_steps)
        if total_steps <= 50:
            return set(range(1, total_steps + 1))
        if total_steps <= 100:
            ratio = 0.30
        elif total_steps <= 200:
            ratio = 0.20
        elif total_steps <= 600:
            ratio = 0.10
        else:
            ratio = 0.08
        target = max(1, round(total_steps * ratio))
        front_end = max(1, math.ceil(total_steps * 0.20))
        mid_end = max(front_end, math.ceil(total_steps * 0.60))
        front_n = math.ceil(target * 0.50)
        mid_n = math.ceil(target * 0.30)
        late_n = max(0, target - front_n - mid_n)
        points = set(cls._spread_points(1, front_end, front_n))
        points.update(cls._spread_points(front_end + 1, mid_end, mid_n))
        points.update(cls._spread_points(mid_end + 1, total_steps, late_n))
        for step in range(1, total_steps + 1):
            if len(points) >= target:
                break
            points.add(step)
        return {s for s in points if 1 <= s <= total_steps}

    @staticmethod
    def _resolve_microbatch_count(
        grad_accum_steps: Optional[int], microbatch_sample: Optional[str]
    ) -> Optional[int]:
        if grad_accum_steps is None or grad_accum_steps <= 0:
            return None
        if microbatch_sample is None or str(microbatch_sample).strip() == "":
            return grad_accum_steps
        value = str(microbatch_sample).strip().lower()
        if value in {"all", "full", "none", "0"}:
            return grad_accum_steps
        if value.endswith("%"):
            ratio = float(value[:-1]) / 100.0
        else:
            ratio = float(value)
        if 0 < ratio <= 1:
            return max(1, min(grad_accum_steps, math.ceil(grad_accum_steps * ratio)))
        return max(1, min(grad_accum_steps, int(ratio)))

    @classmethod
    def _build_microbatch_schedule(
        cls, grad_accum_steps: Optional[int], microbatch_sample: Optional[str]
    ) -> Optional[Set[int]]:
        if grad_accum_steps is None or grad_accum_steps <= 0:
            return None
        count = cls._resolve_microbatch_count(grad_accum_steps, microbatch_sample)
        if count is None or count >= grad_accum_steps:
            return set(range(1, grad_accum_steps + 1))
        return set(cls._spread_points(1, grad_accum_steps, count))

    def should_log_step(self, layer_name: Optional[str] = None) -> bool:
        if layer_name is not None and not self.should_log_layer(layer_name):
            return False
        if self.selected_steps is not None:
            return self.current_step in self.selected_steps
        return self.current_step % self.log_every_n_steps == 0

    def _observe_microbatch(self, layer_name: str) -> bool:
        step = self.current_step
        if self._active_step != step:
            self._active_step = step
            self._microbatch_index = 0
            self._current_microbatch_selected = True
        if self._microbatch_anchor_layer is None:
            self._microbatch_anchor_layer = layer_name
        if layer_name == self._microbatch_anchor_layer:
            self._microbatch_index += 1
            if self.selected_microbatches is None:
                self._current_microbatch_selected = True
            else:
                self._current_microbatch_selected = self._microbatch_index in self.selected_microbatches
        return self._current_microbatch_selected

    def should_log_record(self, layer_name: Optional[str] = None) -> bool:
        if layer_name is None:
            return self.should_log_step(layer_name)
        if not self.should_log_step(layer_name):
            return False
        return self._observe_microbatch(layer_name)

    def _ensure_stream(self):
        if self._stream_writer is not None:
            return self._stream_writer
        os.makedirs(self.output_dir, exist_ok=True)
        self._stream_path = self._stats_path()
        self._stream_file = open(self._stream_path, "w", newline="")
        self._stream_writer = csv.writer(self._stream_file)
        self._stream_writer.writerow([
            "step", "microbatch", "layer",
            "A_tt", "A_tv", "R_att_text",
            "U_vis", "U_text", "R_info",
            "n_post_text",
        ])
        self._stream_file.flush()
        return self._stream_writer

    def _write_record(self, record: _AttnStepRecord) -> None:
        writer = self._ensure_stream()
        writer.writerow([
            record.step, record.microbatch, record.layer,
            f"{record.A_tt:.6f}", f"{record.A_tv:.6f}", f"{record.R_att_text:.4f}",
            f"{record.U_vis:.6f}", f"{record.U_text:.6f}", f"{record.R_info:.4f}",
            record.n_post_text,
        ])
        self._record_written += 1
        if self._stream_file is not None:
            self._stream_file.flush()

    def _close_stream(self) -> None:
        if self._stream_file is not None:
            self._stream_file.flush()
            self._stream_file.close()
        self._stream_file = None
        self._stream_writer = None

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
        skipped = 0
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                if self.should_log_layer(name):
                    module._log_attn         = True
                    module._attn_logger      = self
                    module._attn_layer_name  = name
                    module._attn_token_mask  = None
                    attn_count += 1
                else:
                    module._log_attn = False
                    skipped += 1

        import types
        patch_targets = []
        seen = set()
        roots = [model]
        if hasattr(model, "modules"):
            roots.extend(list(model.modules()))
        for target in roots:
            if id(target) in seen:
                continue
            seen.add(id(target))
            if callable(getattr(target, "_apply_lora_token_mask", None)):
                patch_targets.append(target)

        for target in patch_targets:
            if hasattr(target, "_attn_orig_apply_lora_token_mask"):
                continue
            bound_apply = target._apply_lora_token_mask
            orig_func = getattr(bound_apply, "__func__", None)
            target._attn_orig_apply_lora_token_mask = bound_apply

            def _patched_apply(self_model, _orig_func=orig_func, _orig_bound=bound_apply):
                if _orig_func is not None:
                    _orig_func(self_model)
                else:
                    _orig_bound()
                token_mask = getattr(self_model, "current_lora_mask", None)
                module_root = getattr(self_model, "model", self_model)
                for module in module_root.modules():
                    if isinstance(module, LlamaAttention):
                        module._attn_token_mask = token_mask

            target._apply_lora_token_mask = types.MethodType(_patched_apply, target)

        self._enabled = True
        schedule_desc = (f"staged {len(self.selected_steps)} steps"
                         if self.selected_steps is not None else
                         f"every {self.log_every_n_steps} optimizer steps")
        layer_desc = ("all blocks" if self.layer_blocks is None else
                      f"blocks {sorted(self.layer_blocks)}")
        mb_desc = ("unknown micro-batches" if self.selected_microbatches is None else
                   f"{len(self.selected_microbatches)}/{self.grad_accum_steps} micro-batches {sorted(self.selected_microbatches)}")
        print(f"[AttnLogger] attached to {attn_count} LlamaAttention layers "
              f"({skipped} skipped; {layer_desc}; {schedule_desc}; {mb_desc}); "
              f"patched {len(patch_targets)} token-mask owner(s)")

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
        self._record_calls += 1

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

        record = _AttnStepRecord(
            step, self._microbatch_index, layer_name,
            A_tt_wsum  / n_post_text_total,
            A_tv_wsum  / n_post_text_total,
            U_vis_wsum  / n_post_text_total,
            U_text_wsum / n_post_text_total,
            n_post_text_total / valid_b,
        )
        self.records.append(record)
        self._write_record(record)

    def save_csv(self, task_name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        rank = self._rank()
        path = os.path.join(self.output_dir, f"{task_name}_rank{rank}_attn_stats.csv")
        if self._stream_writer is None and not os.path.exists(path):
            self._ensure_stream()
        self._close_stream()
        self._save_summary_csv(task_name, rank)
        print(f"[AttnLogger] rank{rank} streamed {len(self.records)} records -> {path}")
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
        self._close_stream()
        self.records.clear()
        self.step = 0
        self._active_step = None
        self._microbatch_index = 0
        self._current_microbatch_selected = True
        self._record_calls = 0
        self._record_written = 0
        self._skip_no_visual = 0
        self._skip_no_post_text = 0

    def detach(self, model: nn.Module) -> None:
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
        except ImportError:
            return
        self._close_stream()
        for module in model.modules():
            if isinstance(module, LlamaAttention):
                module._log_attn    = False
                module._attn_logger = None
        self._enabled = False
