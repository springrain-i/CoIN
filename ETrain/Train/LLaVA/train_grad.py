"""
train_grad.py — training entry with ModalGradientLogger + AttentionLogger injected.

Usage:
  python ETrain/Train/LLaVA/train_grad.py \\
      ... (all normal train.py args) ... \\
      --log_gradient_stats True \\
      --log_attn_stats      True \\
      --grad_task_name      ScienceQA \\
      --grad_output_dir     analysis/gradient_dominance \\
      --attn_output_dir     analysis/attn_dominance \\
      --grad_log_interval   1

Both loggers are independent — either or both can be enabled.
COIN_USE_SDPA_PATCH=1 must be set when --log_attn_stats True (attn weights
require the SDPA patch's manual recompute path).
"""

import sys
import os

# Make step/loss logs visible through tee while long tasks are still running.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(line_buffering=True, write_through=True)

# ── sys.path: repo-local ETrain takes precedence over editable install ────────
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ── parse & strip our custom args before train.py sees them ──────────────────
def _pop_arg(argv, flag, default=None):
    try:
        idx = argv.index(flag)
        val = argv[idx + 1]
        del argv[idx:idx + 2]
        return val
    except (ValueError, IndexError):
        return default


_argv = sys.argv[1:]
_log_grad      = _pop_arg(_argv, "--log_gradient_stats", "False").lower() == "true"
_log_attn      = _pop_arg(_argv, "--log_attn_stats",     "False").lower() == "true"
_task_name     = _pop_arg(_argv, "--grad_task_name",     "task_unknown")
_grad_out_dir  = _pop_arg(_argv, "--grad_output_dir",    "analysis/gradient_dominance")
_attn_out_dir  = _pop_arg(_argv, "--attn_output_dir",    "analysis/attn_dominance")
_log_every     = int(_pop_arg(_argv, "--grad_log_interval", "1"))
_grad_total_steps = _pop_arg(_argv, "--grad_total_steps", None)
_grad_total_steps = int(_grad_total_steps) if _grad_total_steps not in (None, "") else None
_grad_step_schedule = _pop_arg(_argv, "--grad_step_schedule", "interval")
_grad_layer_blocks = _pop_arg(_argv, "--grad_layer_blocks", None)
_grad_accum_steps = _pop_arg(_argv, "--grad_accum_steps", None)
_grad_accum_steps = int(_grad_accum_steps) if _grad_accum_steps not in (None, "") else None
_grad_microbatch_sample = _pop_arg(_argv, "--grad_microbatch_sample", "0.25")
sys.argv[1:]   = _argv


# ── SDPA monkey patch ────────────────────────────────────────────────────────
_use_sdpa = os.environ.get("COIN_USE_SDPA_PATCH", "0") == "1"
if _use_sdpa:
    from ETrain.Train.LLaVA.llama_sdpa_monkey_patch import replace_llama_attn_with_sdpa
    replace_llama_attn_with_sdpa()
    if _log_attn:
        print("[train_grad] SDPA patch active — attention logging enabled")
else:
    from ETrain.Train.LLaVA.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    if _log_attn:
        print("[train_grad] WARNING: COIN_USE_SDPA_PATCH not set — "
              "attention logging requires SDPA patch; attn stats will be empty")


# ── build loggers (only if enabled) ──────────────────────────────────────────
_grad_logger = None
_attn_logger = None

if _log_grad:
    from ETrain.Train.LLaVA.gradient_logger import ModalGradientLogger
    _grad_logger = ModalGradientLogger(
        log_every_n_steps=_log_every,
        output_dir=_grad_out_dir,
        total_optimizer_steps=_grad_total_steps,
        step_schedule=_grad_step_schedule,
        layer_blocks=_grad_layer_blocks,
        grad_accum_steps=_grad_accum_steps,
        microbatch_sample=_grad_microbatch_sample,
        task_name=_task_name,
    )

if _log_attn:
    from ETrain.Train.LLaVA.attention_logger import AttentionLogger
    _attn_logger = AttentionLogger(
        log_every_n_steps=_log_every,
        output_dir=_attn_out_dir,
        task_name=_task_name,
        total_optimizer_steps=_grad_total_steps,
        step_schedule=_grad_step_schedule,
        layer_blocks=_grad_layer_blocks,
        grad_accum_steps=_grad_accum_steps,
        microbatch_sample=_grad_microbatch_sample,
    )


# ── inject single callback that handles both loggers ─────────────────────────
if _log_grad or _log_attn:
    from transformers import TrainerCallback, TrainerState, TrainerControl
    from ETrain.Train.LLaVA import llava_trainer as _llava_trainer_mod

    _orig_init = _llava_trainer_mod.LLaVATrainer.__init__

    class _StatsCallback(TrainerCallback):
        def on_train_begin(self, args, state: TrainerState, control: TrainerControl,
                           model=None, **kw):
            if model is None:
                return
            import torch
            torch.cuda.empty_cache()
            if _grad_logger is not None:
                _grad_logger.attach(model)
                print(f"[GradLogger] enabled for task '{_task_name}'")
            if _attn_logger is not None:
                _attn_logger.attach(model)
                print(f"[AttnLogger] enabled for task '{_task_name}'")

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kw):
            if _grad_logger is not None:
                _grad_logger.step_end()
            if _attn_logger is not None:
                _attn_logger.step_end()

        def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kw):
            if _grad_logger is not None:
                path = _grad_logger.save_csv(_task_name)
                print(f"[GradLogger] saved → {path}")
            if _attn_logger is not None:
                path = _attn_logger.save_csv(_task_name)
                print(f"[AttnLogger] saved → {path}")

    def _patched_init(self, *args, **kwargs):
        existing = list(kwargs.get("callbacks") or [])
        existing.append(_StatsCallback())
        kwargs["callbacks"] = existing
        _orig_init(self, *args, **kwargs)

    _llava_trainer_mod.LLaVATrainer.__init__ = _patched_init
    print(f"[train_grad] patched LLaVATrainer — task={_task_name}, "
          f"log_every={_log_every}, grad={_log_grad}, attn={_log_attn}, "
          f"schedule={_grad_step_schedule}, blocks={_grad_layer_blocks}, "
          f"accum={_grad_accum_steps}, microbatch_sample={_grad_microbatch_sample}")


# ── run training ──────────────────────────────────────────────────────────────
from ETrain.Train.LLaVA.train import train

if __name__ == "__main__":
    train()
