"""
train_grad.py — training entry point with ModalGradientLogger injected.

Usage (same as train_mem.py, plus two extra args):
  python -m ETrain.Train.LLaVA.train_grad \
      ... (all normal train.py args) ... \
      --log_gradient_stats True \
      --grad_task_name ScienceQA \
      --grad_output_dir analysis/gradient_dominance \
      --grad_log_every 1

How it works:
  1. Parses --log_gradient_stats / --grad_task_name / --grad_output_dir / --grad_log_every
     from sys.argv and removes them so train.py never sees them.
  2. Monkey-patches LLaVATrainer to inject a TrainerCallback that:
       - on_train_begin: attaches logger to model, enables _log_gradients on each LoRA layer
       - on_step_end:    calls logger.step_end()
       - on_train_end:   saves CSV
  3. Activates the SDPA patch (same as train_mem.py).
  4. Calls train() normally.
"""

import sys
import os

# ── sys.path: repo-local ETrain takes precedence over editable install ────────
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ── parse & strip gradient-logger specific args before train.py sees them ─────
def _pop_arg(argv, flag, default=None):
    """Remove --flag value from argv list and return value."""
    try:
        idx = argv.index(flag)
        val = argv[idx + 1]
        del argv[idx:idx + 2]
        return val
    except (ValueError, IndexError):
        return default

_argv = sys.argv[1:]
_log_stats     = _pop_arg(_argv, "--log_gradient_stats", "False").lower() == "true"
_task_name     = _pop_arg(_argv, "--grad_task_name", "task_unknown")
_grad_out_dir  = _pop_arg(_argv, "--grad_output_dir", "analysis/gradient_dominance")
_log_every     = int(_pop_arg(_argv, "--grad_log_every", "1"))
sys.argv[1:]   = _argv   # train.py will see the cleaned argv

# ── SDPA monkey patch (same as train_mem.py) ──────────────────────────────────
_use_sdpa = os.environ.get("COIN_USE_SDPA_PATCH", "0") == "1"
if _use_sdpa:
    from ETrain.Train.LLaVA.llama_sdpa_monkey_patch import replace_llama_attn_with_sdpa
    replace_llama_attn_with_sdpa()
else:
    from ETrain.Train.LLaVA.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()

# ── inject callback via monkey-patch ─────────────────────────────────────────
if _log_stats:
    from transformers import TrainerCallback, TrainerState, TrainerControl
    from ETrain.Train.LLaVA.gradient_logger import ModalGradientLogger
    from ETrain.Train.LLaVA import llava_trainer as _llava_trainer_mod

    _orig_LLaVATrainer_init = _llava_trainer_mod.LLaVATrainer.__init__

    _grad_logger = ModalGradientLogger(
        log_every_n_steps=_log_every,
        output_dir=_grad_out_dir,
    )

    class _GradLogCallback(TrainerCallback):
        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kw):
            if model is None:
                return
            _grad_logger.attach(model)
            # Enable logging path in every CoINMOELoraLinear
            try:
                from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear
                for m in model.modules():
                    if isinstance(m, CoINMOELoraLinear):
                        m._log_gradients = True
            except ImportError:
                pass
            print(f"[GradLogger] gradient logging enabled for task '{_task_name}'")

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kw):
            _grad_logger.step_end()

        def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kw):
            path = _grad_logger.save_csv(_task_name)
            print(f"[GradLogger] training finished — results at {path}")

    def _patched_init(self, *args, **kwargs):
        # Inject callback into kwargs before calling original __init__
        existing = list(kwargs.get("callbacks") or [])
        existing.append(_GradLogCallback())
        kwargs["callbacks"] = existing
        _orig_LLaVATrainer_init(self, *args, **kwargs)

    _llava_trainer_mod.LLaVATrainer.__init__ = _patched_init
    print(f"[GradLogger] monkey-patched LLaVATrainer — task={_task_name}, "
          f"log_every={_log_every}, out={_grad_out_dir}")

# ── run training ──────────────────────────────────────────────────────────────
from ETrain.Train.LLaVA.train import train

if __name__ == "__main__":
    train()
