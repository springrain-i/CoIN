"""
AttentionLogger records answer-query attention usage for LlamaAttention layers.

Query buckets:
  ans            all positions that predict answer tokens
  assistant      prompt/assistant position that predicts the first answer token
  answer_prefix  answer-token positions that predict later answer tokens

Source buckets:
  vis            visual tokens
  prompt         prompt text tokens, including the assistant prefix token
  prev_answer    answer tokens available as causal history

For each query/source pair, the logger writes:
  A_{query}_{source}: mean attention mass assigned to that source bucket
  U_{query}_{source}: mean output-space norm of attention-weighted values

The A metrics follow the generated-token attention analysis style used by
"What's in the Image? A Deep-Dive into the Vision of Vision Language Models".
The U metrics keep the value-contribution view used in CoLM 2025-style
attention information-flow diagnostics: attention_weight * value, projected
through the per-head output projection block.

Multi-GPU: each rank writes its own CSV; merge downstream for combined analysis.
"""
import csv
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Set, TextIO

import torch
import torch.nn as nn


QUERY_GROUPS = ("ans", "assistant", "answer_prefix")
SOURCE_GROUPS = ("vis", "prompt", "prev_answer")
ATTN_METRIC_FIELDS = (
    [f"A_{query}_{source}" for query in QUERY_GROUPS for source in SOURCE_GROUPS]
    + [f"U_{query}_{source}" for query in QUERY_GROUPS for source in SOURCE_GROUPS]
    + ["R_att_ans_vis", "R_info_ans_vis"]
)
ATTN_COUNT_FIELDS = ["n_ans_query", "n_assistant_query", "n_answer_prefix_query"]


class _AttnStepRecord:
    __slots__ = ("step", "microbatch", "layer", "metrics", "counts")

    def __init__(self, step: int, microbatch: int, layer: str, metrics: Dict[str, float], counts: Dict[str, int]):
        self.step = step
        self.microbatch = microbatch
        self.layer = layer
        self.metrics = metrics
        self.counts = counts

    def metric(self, name: str) -> float:
        return float(self.metrics.get(name, 0.0))

    def count(self, name: str) -> int:
        return int(self.counts.get(name, 0))


class AttentionLogger:
    """
    Attach to model once; receives _record() calls from the SDPA patch on each
    prefill forward; call step_end() after each optimizer step; save_csv() at
    task end.

    Token mask propagation: monkey-patches _apply_lora_token_mask so that
    LlamaAttention modules receive the same diagnostic masks used by the grad
    logger.
    """

    QUERY_GROUPS = QUERY_GROUPS
    SOURCE_GROUPS = SOURCE_GROUPS
    METRIC_FIELDS = ATTN_METRIC_FIELDS
    COUNT_FIELDS = ATTN_COUNT_FIELDS

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
        self._skip_no_answer_query = 0

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

    def _stats_path(self, task_name: Optional[str] = None) -> str:
        rank = self._rank()
        return os.path.join(self.output_dir, f"{task_name or self.task_name}_rank{rank}_attn_stats.csv")

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
        self._stream_writer.writerow(
            ["step", "microbatch", "layer"] + self.METRIC_FIELDS + self.COUNT_FIELDS
        )
        self._stream_file.flush()
        return self._stream_writer

    def _write_record(self, record: _AttnStepRecord) -> None:
        writer = self._ensure_stream()
        metric_values = []
        for field in self.METRIC_FIELDS:
            value = record.metric(field)
            metric_values.append(f"{value:.6f}" if math.isfinite(value) else str(value))
        writer.writerow(
            [record.step, record.microbatch, record.layer]
            + metric_values
            + [record.count(field) for field in self.COUNT_FIELDS]
        )
        self._record_written += 1
        if self._stream_file is not None:
            self._stream_file.flush()

    def _close_stream(self) -> None:
        if self._stream_file is not None:
            self._stream_file.flush()
            self._stream_file.close()
        self._stream_file = None
        self._stream_writer = None

    def attach(self, model: nn.Module) -> None:
        """
        1. Set _log_attn / _attn_logger / _attn_layer_name on LlamaAttention.
        2. Patch _apply_lora_token_mask to also push diagnostic masks to attention.
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
                    module._log_attn = True
                    module._attn_logger = self
                    module._attn_layer_name = name
                    module._attn_token_mask = None
                    module._attn_grad_token_mask = None
                    module._attn_answer_query_mask = None
                    module._attn_assistant_query_mask = None
                    module._attn_answer_prefix_query_mask = None
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
                grad_token_mask = getattr(self_model, "current_grad_token_mask", token_mask)
                answer_query_mask = getattr(self_model, "current_answer_query_mask", None)
                assistant_query_mask = getattr(self_model, "current_assistant_query_mask", None)
                answer_prefix_query_mask = getattr(self_model, "current_answer_prefix_query_mask", None)
                module_root = getattr(self_model, "model", self_model)
                for module in module_root.modules():
                    if isinstance(module, LlamaAttention):
                        module._attn_token_mask = token_mask
                        module._attn_grad_token_mask = grad_token_mask
                        module._attn_answer_query_mask = answer_query_mask
                        module._attn_assistant_query_mask = assistant_query_mask
                        module._attn_answer_prefix_query_mask = answer_prefix_query_mask

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

    @staticmethod
    def _align_mask(
        mask: Optional[torch.Tensor],
        bsz: int,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        if mask is None or mask.dim() != 2 or mask.shape[0] != bsz:
            return None
        if mask.shape[1] == seq_len:
            aligned = mask
        elif mask.shape[1] > seq_len:
            aligned = mask[:, -seq_len:]
        else:
            return None
        if dtype is not None:
            aligned = aligned.to(dtype=dtype)
        return aligned.to(device=device)

    def _record(
        self,
        step: int,
        layer_name: str,
        attn_weights: torch.Tensor,          # [B, H, T_q, T_k]
        token_mask: torch.Tensor,            # [B, T_k], 1=visual, 2=text, 0=pad
        grad_token_mask: Optional[torch.Tensor],       # [B, T_k], 1=visual, 2=prompt, 3=answer
        answer_query_mask: Optional[torch.Tensor],     # [B, T_q]
        assistant_query_mask: Optional[torch.Tensor],  # [B, T_q]
        answer_prefix_query_mask: Optional[torch.Tensor],  # [B, T_q]
        value_states: torch.Tensor,          # [B, H, T_k, head_dim]
        o_proj_weight: torch.Tensor,         # [hidden_dim, H*head_dim]
    ) -> None:
        self._record_calls += 1

        bsz, num_heads, q_len, kv_len = attn_weights.shape
        head_d = value_states.shape[-1]
        hidden_d = o_proj_weight.shape[0]
        device = attn_weights.device

        token_mask = self._align_mask(token_mask, bsz, kv_len, device, torch.long)
        grad_token_mask = self._align_mask(grad_token_mask, bsz, kv_len, device, torch.long)
        if grad_token_mask is None:
            grad_token_mask = token_mask
        answer_query_mask = self._align_mask(answer_query_mask, bsz, q_len, device, torch.bool)
        assistant_query_mask = self._align_mask(assistant_query_mask, bsz, q_len, device, torch.bool)
        answer_prefix_query_mask = self._align_mask(answer_prefix_query_mask, bsz, q_len, device, torch.bool)
        if token_mask is None or grad_token_mask is None or answer_query_mask is None:
            return

        query_masks = {
            "ans": answer_query_mask,
            "assistant": assistant_query_mask if assistant_query_mask is not None else torch.zeros_like(answer_query_mask),
            "answer_prefix": (
                answer_prefix_query_mask if answer_prefix_query_mask is not None
                else torch.zeros_like(answer_query_mask)
            ),
        }
        if not query_masks["ans"].any():
            self._skip_no_answer_query += 1
            return

        metrics_sum = {field: 0.0 for field in self.METRIC_FIELDS}
        count_sum = {field: 0 for field in self.COUNT_FIELDS}

        with torch.no_grad():
            W_O_T = (
                o_proj_weight.float()
                .view(hidden_d, num_heads, head_d)
                .permute(1, 2, 0)
            )  # [H, head_d, hidden_d]

            for b in range(bsz):
                visual_mask = grad_token_mask[b] == 1
                if not visual_mask.any():
                    self._skip_no_visual += 1
                    continue
                source_masks = {
                    "vis": visual_mask,
                    "prompt": grad_token_mask[b] == 2,
                    "prev_answer": grad_token_mask[b] == 3,
                }
                w_b = attn_weights[b]       # [H, T_q, T_k]
                v_b = value_states[b].float()  # [H, T_k, head_d]

                for query_name, q_mask_all in query_masks.items():
                    q_mask = q_mask_all[b].bool()
                    n_query = int(q_mask.sum().item())
                    if n_query == 0:
                        continue
                    w_q = w_b[:, q_mask, :]  # [H, T_query, T_k]
                    count_sum[f"n_{query_name}_query"] += n_query

                    for source_name, source_mask in source_masks.items():
                        metric_a = f"A_{query_name}_{source_name}"
                        metric_u = f"U_{query_name}_{source_name}"
                        if not source_mask.any():
                            continue
                        source_weight = w_q[:, :, source_mask].float()
                        attn_mass = source_weight.sum(dim=-1).mean().item()
                        source_sum = torch.matmul(source_weight, v_b[:, source_mask, :])
                        u_source = torch.matmul(source_sum, W_O_T)
                        value_norm = u_source.norm(dim=-1).mean().item()
                        metrics_sum[metric_a] += attn_mass * n_query
                        metrics_sum[metric_u] += value_norm * n_query

        if count_sum["n_ans_query"] == 0:
            self._skip_no_answer_query += 1
            return

        metrics: Dict[str, float] = {}
        for query_name in self.QUERY_GROUPS:
            denom = max(count_sum[f"n_{query_name}_query"], 1)
            for source_name in self.SOURCE_GROUPS:
                metrics[f"A_{query_name}_{source_name}"] = metrics_sum[f"A_{query_name}_{source_name}"] / denom
                metrics[f"U_{query_name}_{source_name}"] = metrics_sum[f"U_{query_name}_{source_name}"] / denom

        ans_att_denom = sum(metrics[f"A_ans_{source}"] for source in self.SOURCE_GROUPS)
        ans_info_denom = sum(metrics[f"U_ans_{source}"] for source in self.SOURCE_GROUPS)
        metrics["R_att_ans_vis"] = metrics["A_ans_vis"] / max(ans_att_denom, 1e-8)
        metrics["R_info_ans_vis"] = metrics["U_ans_vis"] / max(ans_info_denom, 1e-8)

        record = _AttnStepRecord(step, self._microbatch_index, layer_name, metrics, count_sum)
        self.records.append(record)
        self._write_record(record)

    def save_csv(self, task_name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        rank = self._rank()
        path = self._stats_path(task_name)
        if self._stream_writer is None and not os.path.exists(path):
            self._ensure_stream()
        self._close_stream()
        self._save_summary_csv(task_name, rank)
        print(f"[AttnLogger] rank{rank} streamed {len(self.records)} records -> {path}")
        return path

    def _save_summary_csv(self, task_name: str, rank: int) -> None:
        if not self.records:
            return
        layer_data: dict = defaultdict(list)
        for record in self.records:
            layer_data[record.layer].append(record)

        def _mean_metric(records: List[_AttnStepRecord], field: str) -> float:
            values = [record.metric(field) for record in records]
            finite_values = [value for value in values if math.isfinite(value)]
            if not finite_values:
                return float("inf")
            return statistics.mean(finite_values)

        def _mean_count(records: List[_AttnStepRecord], field: str) -> float:
            return statistics.mean([record.count(field) for record in records])

        summary_path = os.path.join(
            self.output_dir, f"{task_name}_rank{rank}_attn_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["scope", "layer", "n_records"]
                + [f"mean_{field}" for field in self.METRIC_FIELDS]
                + [f"mean_{field}" for field in self.COUNT_FIELDS]
            )
            for layer, records in sorted(layer_data.items()):
                writer.writerow(
                    ["layer", layer, len(records)]
                    + [f"{_mean_metric(records, field):.6f}" for field in self.METRIC_FIELDS]
                    + [f"{_mean_count(records, field):.2f}" for field in self.COUNT_FIELDS]
                )
            writer.writerow(
                ["global", "ALL", len(self.records)]
                + [f"{_mean_metric(self.records, field):.6f}" for field in self.METRIC_FIELDS]
                + [f"{_mean_count(self.records, field):.2f}" for field in self.COUNT_FIELDS]
            )
        print(f"[AttnLogger] rank{rank} summary -> {summary_path}")

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
        self._skip_no_answer_query = 0

    def detach(self, model: nn.Module) -> None:
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
        except ImportError:
            return
        self._close_stream()
        for module in model.modules():
            if isinstance(module, LlamaAttention):
                module._log_attn = False
                module._attn_logger = None
                module._attn_token_mask = None
                module._attn_grad_token_mask = None
                module._attn_answer_query_mask = None
                module._attn_assistant_query_mask = None
                module._attn_answer_prefix_query_mask = None
        self._enabled = False
