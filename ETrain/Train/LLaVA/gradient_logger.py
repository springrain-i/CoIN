"""
ModalGradientLogger — records token-bucket gradient contributions to MoELoRA parameters.

Three metrics per layer per logged step (all EXACT):
  W_A:  G^bucket_A[n,r,d] = Σ_{b,t∈bucket} g_out_A[b,t,n,r] * lora_x[b,t,d]
  W_B:  G^bucket_B[n,d,r] = Σ_{b,t∈bucket} g_out_B[b,t,n,d] * out_A[b,t,n,r]
  ΔW:   G^bucket_dW[o,i]  = Σ_{b,t∈bucket} g_lora_out[b,t,o] * lora_x[b,t,i]

Buckets:
  visual: token mask == 1
  prompt: text token with label == IGNORE_INDEX
  answer: text token with label != IGNORE_INDEX
  assistant_query / answer_prefix_query: q_proj-only query-side buckets for
  positions that predict answer tokens.

Important interpretation:
  G_* values are "aggregate update contribution" metrics: contributions from
  all tokens in a bucket are summed into one parameter-shaped gradient matrix,
  then its Frobenius norm is reported. They are not mean per-token gradient
  activity metrics, because per-token norms are not computed before summing.

Direction metrics:
  cos_* fields flatten the bucket matrices above and compute cosine similarity,
  following the gradient-direction diagnostic idea used in MMPareto/BalGrad.

Expert-wise diagnostics are streamed separately as bucket x expert stats.
They keep A/B gradients and route usage only. Expert-wise effective ΔW is
intentionally omitted because it requires per-expert [d_out, d_in] matrices and
is much more expensive than the A/B slices.
"""
import csv
import math
import os
import re
import statistics
from typing import Dict, List, Optional, Set, TextIO, Tuple

import torch
import torch.nn as nn


EXPERT_BUCKETS = ("prompt", "vis", "answer")
EXPERT_METRIC_PREFIXES = ("A", "B")
EXPERT_FIELDS = (
    ["route_mean_prompt", "route_mean_vis", "route_mean_answer",
     "route_mass_prompt", "route_mass_vis", "route_mass_answer",
     "n_prompt", "n_vis", "n_answer"]
    + [f"expert_grad_{prefix}_{bucket}"
       for prefix in EXPERT_METRIC_PREFIXES
       for bucket in EXPERT_BUCKETS]
    + [f"cos_{prefix}_prompt_vis" for prefix in EXPERT_METRIC_PREFIXES]
    + [f"cos_{prefix}_answer_vis" for prefix in EXPERT_METRIC_PREFIXES]
    + [f"cos_{prefix}_prompt_answer" for prefix in EXPERT_METRIC_PREFIXES]
)


class _StepRecord:
    __slots__ = ('step', 'microbatch', 'layer', 'metrics', 'counts')

    def __init__(self, step, microbatch, layer, metrics, counts):
        self.step = step
        self.microbatch = microbatch
        self.layer = layer
        self.metrics = metrics
        self.counts = counts

    def metric(self, name: str) -> float:
        return float(self.metrics.get(name, 0.0))

    def count(self, name: str) -> int:
        return int(self.counts.get(name, 0))

    @property
    def g_prompt_A(self):
        return self.metric("G_prompt_A")

    @property
    def g_vis_A(self):
        return self.metric("G_vis_A")

    @property
    def g_answer_A(self):
        return self.metric("G_answer_A")

    @property
    def g_prompt_B(self):
        return self.metric("G_prompt_B")

    @property
    def g_vis_B(self):
        return self.metric("G_vis_B")

    @property
    def g_answer_B(self):
        return self.metric("G_answer_B")

    @property
    def g_prompt_dW(self):
        return self.metric("G_prompt_dW")

    @property
    def g_vis_dW(self):
        return self.metric("G_vis_dW")

    @property
    def g_answer_dW(self):
        return self.metric("G_answer_dW")

    @property
    def n_prompt(self):
        return self.count("n_prompt")

    @property
    def n_vis(self):
        return self.count("n_vis")

    @property
    def n_answer(self):
        return self.count("n_answer")


class ModalGradientLogger:
    """
    Attach to a model once; call step_end() after each optimizer step;
    call save_csv(task_name) at the end of each task.

    Gradient checkpointing SUPPORTED: hooks are re-registered on recomputed
    tensors during backward recomputation, so captured x values are consistent.

    Multi-GPU: each rank writes its own {task}_rank{N}_grad_stats.csv.
    Merge files downstream for combined analysis.
    """

    METRIC_FIELDS = [
        "G_prompt_A", "G_vis_A", "G_answer_A",
        "G_assistant_query_A", "G_answer_prefix_query_A",
        "G_prompt_B", "G_vis_B", "G_answer_B",
        "G_assistant_query_B", "G_answer_prefix_query_B",
        "G_prompt_dW", "G_vis_dW", "G_answer_dW",
        "G_assistant_query_dW", "G_answer_prefix_query_dW",
        "cos_prompt_vis_A", "cos_answer_vis_A", "cos_prompt_answer_A",
        "cos_prompt_vis_B", "cos_answer_vis_B", "cos_prompt_answer_B",
        "cos_prompt_vis_dW", "cos_answer_vis_dW", "cos_prompt_answer_dW",
        "cos_assistant_answer_query_A", "cos_assistant_answer_query_B", "cos_assistant_answer_query_dW",
    ]

    COUNT_FIELDS = [
        "n_prompt", "n_vis", "n_answer",
        "n_assistant_query", "n_answer_prefix_query",
    ]

    EXPERT_BUCKETS = EXPERT_BUCKETS
    EXPERT_METRIC_PREFIXES = EXPERT_METRIC_PREFIXES
    EXPERT_FIELDS = EXPERT_FIELDS

    def __init__(
        self,
        log_every_n_steps: int = 1,
        output_dir: str = "analysis/gradient_dominance",
        total_optimizer_steps: Optional[int] = None,
        step_schedule: str = "interval",
        layer_blocks: Optional[str] = None,
        grad_accum_steps: Optional[int] = None,
        microbatch_sample: Optional[str] = "0.25",
        task_name: str = "task_unknown",
    ):
        self.log_every_n_steps = max(int(log_every_n_steps), 1)
        self.output_dir = output_dir
        self.task_name = task_name
        self.step: int = 0  # completed optimizer steps; hooks log current step = step + 1
        self.total_optimizer_steps = total_optimizer_steps
        self.step_schedule = step_schedule
        self.selected_steps = self._build_step_schedule(total_optimizer_steps, step_schedule)
        self.layer_blocks = self._parse_layer_blocks(layer_blocks)
        self.grad_accum_steps = int(grad_accum_steps) if grad_accum_steps not in (None, "") else None
        self.microbatch_sample = microbatch_sample
        self.selected_microbatches = self._build_microbatch_schedule(
            self.grad_accum_steps, microbatch_sample)
        self.records: List[_StepRecord] = []
        self._pending:    Dict[Tuple, dict] = {}   # hook_B results
        self._pending_dw: Dict[Tuple, dict] = {}   # hook_lora_out results
        self._stream_file: Optional[TextIO] = None
        self._stream_writer = None
        self._stream_path: Optional[str] = None
        self._expert_stream_file: Optional[TextIO] = None
        self._expert_stream_writer = None
        self._expert_stream_path: Optional[str] = None
        self._microbatch_anchor_layer: Optional[str] = None
        self._active_step: Optional[int] = None
        self._microbatch_index = 0
        self._current_microbatch_selected = True
        self._enabled = False

    @property
    def current_step(self) -> int:
        return self.step + 1

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

    @staticmethod
    def _rank() -> int:
        try:
            import torch.distributed as dist
            return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        except Exception:
            return 0

    def _stats_path(self, task_name: Optional[str] = None) -> str:
        rank = self._rank()
        return os.path.join(self.output_dir, f"{task_name or self.task_name}_rank{rank}_grad_stats.csv")

    def _expert_stats_path(self, task_name: Optional[str] = None) -> str:
        rank = self._rank()
        return os.path.join(self.output_dir, f"{task_name or self.task_name}_rank{rank}_expert_stats.csv")

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

    def _write_record(self, record: _StepRecord) -> None:
        writer = self._ensure_stream()
        metric_values = []
        for name in self.METRIC_FIELDS:
            value = record.metric(name)
            metric_values.append(f"{value:.6f}" if math.isfinite(value) else str(value))
        writer.writerow(
            [record.step, record.microbatch, record.layer]
            + metric_values
            + [record.count(name) for name in self.COUNT_FIELDS]
        )
        if self._stream_file is not None:
            self._stream_file.flush()

    def _close_stream(self) -> None:
        if self._stream_file is not None:
            self._stream_file.flush()
            self._stream_file.close()
        self._stream_file = None
        self._stream_writer = None

    def _ensure_expert_stream(self):
        if self._expert_stream_writer is not None:
            return self._expert_stream_writer
        os.makedirs(self.output_dir, exist_ok=True)
        self._expert_stream_path = self._expert_stats_path()
        self._expert_stream_file = open(self._expert_stream_path, "w", newline="")
        self._expert_stream_writer = csv.writer(self._expert_stream_file)
        self._expert_stream_writer.writerow(
            ["step", "microbatch", "layer", "expert"] + self.EXPERT_FIELDS
        )
        self._expert_stream_file.flush()
        return self._expert_stream_writer

    def _write_expert_records(self, step: int, layer: str, expert_rows: List[dict]) -> None:
        if not expert_rows:
            return
        writer = self._ensure_expert_stream()
        for row in expert_rows:
            values = []
            for field in self.EXPERT_FIELDS:
                value = row.get(field, 0.0)
                if isinstance(value, int):
                    values.append(value)
                else:
                    values.append(f"{float(value):.6f}" if math.isfinite(float(value)) else str(value))
            writer.writerow([step, self._microbatch_index, layer, row.get("expert", -1)] + values)
        if self._expert_stream_file is not None:
            self._expert_stream_file.flush()

    def _close_expert_stream(self) -> None:
        if self._expert_stream_file is not None:
            self._expert_stream_file.flush()
            self._expert_stream_file.close()
        self._expert_stream_file = None
        self._expert_stream_writer = None

    # ── public API ────────────────────────────────────────────────────────────

    def attach(self, model: nn.Module) -> None:
        """Set logger reference on every CoINMOELoraLinear in model."""
        try:
            from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear
        except ImportError:
            raise ImportError("CoINMOELoraLinear not found")

        count = 0
        skipped = 0
        for name, module in model.named_modules():
            if isinstance(module, CoINMOELoraLinear):
                if self.should_log_layer(name):
                    module._log_gradients   = True
                    module._grad_logger     = self
                    module._grad_layer_name = name
                    count += 1
                else:
                    module._log_gradients = False
                    skipped += 1
        self._enabled = True
        schedule_desc = (f"staged {len(self.selected_steps)} steps"
                         if self.selected_steps is not None else
                         f"every {self.log_every_n_steps} optimizer steps")
        layer_desc = ("all blocks" if self.layer_blocks is None else
                      f"blocks {sorted(self.layer_blocks)}")
        mb_desc = ("unknown micro-batches" if self.selected_microbatches is None else
                   f"{len(self.selected_microbatches)}/{self.grad_accum_steps} micro-batches {sorted(self.selected_microbatches)}")
        print(f"[GradLogger] attached to {count} CoINMOELoraLinear layers "
              f"({skipped} skipped; {layer_desc}; {schedule_desc}; {mb_desc})")

    def step_end(self) -> None:
        self.step += 1

    def _flush(self, step, layer, metrics, counts) -> None:
        """Called by hook_A after all three hooks have fired."""
        if self.selected_steps is not None and step not in self.selected_steps:
            return
        if self.selected_steps is None and step % self.log_every_n_steps != 0:
            return
        record = _StepRecord(
            step, self._microbatch_index, layer,
            metrics, counts,
        )
        self.records.append(record)
        self._write_record(record)

    def _flush_expert(self, step, layer, expert_rows) -> None:
        if self.selected_steps is not None and step not in self.selected_steps:
            return
        if self.selected_steps is None and step % self.log_every_n_steps != 0:
            return
        self._write_expert_records(step, layer, expert_rows)

    def save_csv(self, task_name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        rank = self._rank()
        path = self._stats_path(task_name)
        if self._stream_writer is None and not os.path.exists(path):
            self._ensure_stream()
        self._close_stream()
        self._close_expert_stream()
        self._save_summary_csv(task_name, rank)
        self._print_summary(task_name)
        print(f"[GradLogger] rank{rank} streamed {len(self.records)} records -> {path}")
        if self._expert_stream_path is not None:
            print(f"[GradLogger] rank{rank} streamed expert records -> {self._expert_stream_path}")
        return path

    def _save_summary_csv(self, task_name: str, rank: int) -> None:
        """Write per-layer and global mean summaries for every streamed field."""
        if not self.records:
            return

        from collections import defaultdict
        layer_data: dict = defaultdict(list)
        for record in self.records:
            layer_data[record.layer].append(record)

        def _mean_metric(records: List[_StepRecord], field: str) -> float:
            values = [record.metric(field) for record in records]
            finite_values = [value for value in values if math.isfinite(value)]
            if not finite_values:
                return float("inf")
            return statistics.mean(finite_values)

        def _mean_count(records: List[_StepRecord], field: str) -> float:
            return statistics.mean([record.count(field) for record in records])

        summary_path = os.path.join(
            self.output_dir, f"{task_name}_rank{rank}_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["scope", "layer", "n_records"]
                + [f"mean_{field}" for field in self.METRIC_FIELDS]
                + [f"mean_{field}" for field in self.COUNT_FIELDS]
            )
            for layer, records in sorted(layer_data.items()):
                w.writerow(
                    ["layer", layer, len(records)]
                    + [f"{_mean_metric(records, field):.6f}" for field in self.METRIC_FIELDS]
                    + [f"{_mean_count(records, field):.2f}" for field in self.COUNT_FIELDS]
                )
            w.writerow(
                ["global", "ALL", len(self.records)]
                + [f"{_mean_metric(self.records, field):.6f}" for field in self.METRIC_FIELDS]
                + [f"{_mean_count(self.records, field):.2f}" for field in self.COUNT_FIELDS]
            )
        print(f"[GradLogger] rank{rank} summary -> {summary_path}")

    def clear_records(self) -> None:
        self._close_stream()
        self._close_expert_stream()
        self.records.clear()
        self._pending.clear()
        self._pending_dw.clear()
        self.step = 0
        self._active_step = None
        self._microbatch_index = 0
        self._current_microbatch_selected = True

    def detach(self, model: nn.Module) -> None:
        try:
            from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear
        except ImportError:
            return
        self._close_stream()
        self._close_expert_stream()
        for module in model.modules():
            if isinstance(module, CoINMOELoraLinear):
                module._log_gradients = False
                module._grad_logger   = None
        self._enabled = False

    # ── internals ─────────────────────────────────────────────────────────────

    def _print_summary(self, task_name: str) -> None:
        if not self.records:
            return
        fields = [
            ("G_prompt_A", "prompt_A:"),
            ("G_vis_A", "visual_A:"),
            ("G_answer_A", "answer_A:"),
            ("G_prompt_B", "prompt_B:"),
            ("G_vis_B", "visual_B:"),
            ("G_answer_B", "answer_B:"),
            ("G_prompt_dW", "prompt_dW:"),
            ("G_vis_dW", "visual_dW:"),
            ("G_answer_dW", "answer_dW:"),
        ]

        def _fmt(vals, label):
            if not vals:
                return
            print(f"[GradLogger] {task_name}  {label:<12}"
                  f"mean={statistics.mean(vals):.3f}  "
                  f"median={statistics.median(vals):.3f}  "
                  f"n={len(vals)}")

        for field, label in fields:
            _fmt([r.metric(field) for r in self.records], label)
