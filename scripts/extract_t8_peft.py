#!/usr/bin/env python3
"""Extract PEFT adapter files from T8 full pytorch_model.bin checkpoint.

The T8 OCRVQA training completed successfully but save_trained_model() took
the wrong branch (safe_save_model_for_hf_trainer → pytorch_model.bin instead
of PEFT adapter format). This script re-extracts the adapter weights.
"""
import os
import json
import shutil
import torch

CKPT_DIR = "/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN_MoEMoKA/OCRVQA_llava_MoEMoKA_lora"
T7_DIR   = "/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN_MoEMoKA/VQAv2_llava_MoEMoKA_lora"
SRC_BIN  = os.path.join(CKPT_DIR, "pytorch_model.bin")

print(f"Loading {SRC_BIN} ...")
sd = torch.load(SRC_BIN, map_location="cpu", weights_only=False)
print(f"  Total keys: {len(sd)}")

# --- 1. Adapter weights (all lora_ keys) ---------------------------------
lora_sd = {k: v for k, v in sd.items() if "lora_" in k}
print(f"  LoRA adapter keys: {len(lora_sd)}")
adapter_out = os.path.join(CKPT_DIR, "adapter_model.bin")
torch.save(lora_sd, adapter_out)
print(f"  Saved → {adapter_out}  ({os.path.getsize(adapter_out)/1e6:.1f} MB)")

# --- 2. Non-LoRA trainables (mm_projector + any other trainable non-lora) --
# During training: mm_projector_lr is set so mm_projector is trainable.
# The embed_tokens / lm_head are NOT trained (frozen base model except lora).
non_lora_keys = [
    k for k in sd.keys()
    if "lora_" not in k and (
        "mm_projector" in k
        # extend here if other non-lora params were trainable
    )
]
non_lora_sd = {k: v for k, v in sd.items() if k in non_lora_keys}
print(f"  Non-LoRA trainable keys: {len(non_lora_sd)}")
non_lora_out = os.path.join(CKPT_DIR, "non_lora_trainables.bin")
torch.save(non_lora_sd, non_lora_out)
print(f"  Saved → {non_lora_out}  ({os.path.getsize(non_lora_out)/1e6:.1f} MB)")

# --- 3. adapter_config.json (same LoRA hyper-params as T7) ---------------
t7_adapter_cfg = os.path.join(T7_DIR, "adapter_config.json")
out_adapter_cfg = os.path.join(CKPT_DIR, "adapter_config.json")
shutil.copy2(t7_adapter_cfg, out_adapter_cfg)
print(f"  Copied adapter_config.json from T7 → {out_adapter_cfg}")

# --- 4. config.json (LLaVA model architecture config) --------------------
t7_cfg = os.path.join(T7_DIR, "config.json")
out_cfg = os.path.join(CKPT_DIR, "config.json")
shutil.copy2(t7_cfg, out_cfg)
print(f"  Copied config.json from T7 → {out_cfg}")

# --- 5. Verify -------------------------------------------------------
print("\nVerification — final checkpoint contents:")
for fn in sorted(os.listdir(CKPT_DIR)):
    fp = os.path.join(CKPT_DIR, fn)
    if os.path.isfile(fp):
        size = os.path.getsize(fp)
        print(f"  {fn:40s}  {size/1e6:8.1f} MB")

print("\nDone. T8 checkpoint is now in PEFT format.")
