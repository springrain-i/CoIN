#!/usr/bin/env python3
"""Diagnostic: compare T8 SciQA accuracy with extracted adapter vs full pytorch_model.bin.

Run on GPU 0:  CUDA_VISIBLE_DEVICES=0 python scripts/test_t8_fullmodel.py
"""
import os, sys, json, torch, random
sys.path.insert(0, "/data4/home/sqx/CoIN")
os.environ.setdefault("COIN_USE_SDPA_PATCH", "1")

CKPT     = "/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN_MoEMoKA/OCRVQA_llava_MoEMoKA_lora"
BASE     = "/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5"
FULL_BIN = os.path.join(CKPT, "pytorch_model.bin")
SQA_FILE = "/data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/ScienceQA/test.json"
IMG_ROOT = "/data4/wxl/MoBLoRA-backup/CoIN/cl_dataset"
N_SAMPLES = 100
SEED = 42

random.seed(SEED)

# ── imports ─────────────────────────────────────────────────────────────────
try:
    from ETrain.Train.LLaVA.attn_sdpa_eval import replace_llama_attn_with_sdpa
    replace_llama_attn_with_sdpa()
    print("[info] SDPA patch applied")
except Exception:
    pass

from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ETrain.utils.LLaVA.conversation import conv_templates, SeparatorStyle
from ETrain.Models.LLaVA.builder import load_pretrained_model
from ETrain.utils.LLaVA.utils import disable_torch_init
from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image

# ── load questions ──────────────────────────────────────────────────────────
with open(SQA_FILE) as f:
    all_questions = json.load(f)

random.shuffle(all_questions)
questions = all_questions[:N_SAMPLES]
print(f"Loaded {N_SAMPLES} SciQA samples ({sum(1 for q in questions if 'image' in q)} with image)")

# ── load model (extracted adapter_model.bin) ────────────────────────────────
disable_torch_init()
print("\n[1] Loading with extracted adapter_model.bin ...")
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=CKPT, model_base=BASE,
    model_name=get_model_name_from_path(CKPT),
    load_8bit=False, load_4bit=False, device_map="cuda:0"
)
# Set lora_mode (same as eval_model in model_vqa_science.py)
lora_mode = "all"
model.lora_mode = lora_mode
if hasattr(model, "base_model"):
    model.base_model.lora_mode = lora_mode
    if hasattr(model.base_model, "model"):
        model.base_model.model.lora_mode = lora_mode
model.eval()


def run_sqa(model, tokenizer, image_processor, questions):
    correct = 0
    garbage = 0
    total = 0
    sample_outputs = []

    for line in questions:
        idx = line["question_id"]
        qs = line["text"].replace("<image>", "").strip()
        expected = line["answer"].strip().upper()

        if "image" in line:
            img_path = os.path.join(IMG_ROOT, line["image"])
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                image_tensor = image_tensor.half().cuda()
                if getattr(model.config, "mm_use_im_start_end", False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            else:
                image_tensor = None
        else:
            image_tensor = None

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        images = image_tensor.unsqueeze(0) if image_tensor is not None else None

        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=input_ids, images=images,
                do_sample=False, temperature=0.0,
                max_new_tokens=10, use_cache=True,
            )

        output = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred = output[0].upper() if output else ""

        is_abc = pred in "ABCDE"
        if not is_abc:
            garbage += 1
        if pred == expected:
            correct += 1
        total += 1

        if len(sample_outputs) < 5:
            sample_outputs.append(f"  Q{idx}: expected={expected} pred={pred!r} raw={output[:20]!r}")

    return correct, garbage, total, sample_outputs


print("Running SciQA subset (extracted adapter, mode=all) ...")
c1, g1, t1, samples1 = run_sqa(model, tokenizer, image_processor, questions)
print(f"  Accuracy: {c1}/{t1} = {100*c1/t1:.1f}%  garbage={g1}/{t1} ({100*g1/t1:.1f}%)")
print("  Samples:")
for s in samples1: print(s)

# ── now override ALL weights with pytorch_model.bin ──────────────────────────
print(f"\n[2] Loading full pytorch_model.bin ({os.path.getsize(FULL_BIN)/1e9:.1f} GB) ...")
full_sd = torch.load(FULL_BIN, map_location="cpu", weights_only=False)
result = model.load_state_dict(full_sd, strict=False)
print(f"  Matched/loaded: {len(full_sd) - len(result.unexpected_keys)} keys")
print(f"  Missing from pytorch_model.bin: {len(result.missing_keys)}")
print(f"  Unexpected (extra) keys: {len(result.unexpected_keys)}")
if result.missing_keys:
    print("  First 5 missing:", result.missing_keys[:5])
del full_sd
torch.cuda.empty_cache()

model.lora_mode = lora_mode
if hasattr(model, "base_model"):
    model.base_model.lora_mode = lora_mode
    if hasattr(model.base_model, "model"):
        model.base_model.model.lora_mode = lora_mode
model.eval()

print("Running SciQA subset (full pytorch_model.bin, mode=all) ...")
c2, g2, t2, samples2 = run_sqa(model, tokenizer, image_processor, questions)
print(f"  Accuracy: {c2}/{t2} = {100*c2/t2:.1f}%  garbage={g2}/{t2} ({100*g2/t2:.1f}%)")
print("  Samples:")
for s in samples2: print(s)

print(f"\n{'='*50}")
print(f"  Extracted adapter: {100*c1/t1:.1f}%  garbage {100*g1/t1:.1f}%")
print(f"  Full pytorch_model: {100*c2/t2:.1f}%  garbage {100*g2/t2:.1f}%")
diff = abs(c1 - c2)
if diff <= 3:
    print("  -> Essentially identical. Extraction is correct; issue is NOT in weights.")
else:
    print(f"  -> Differ by {diff} samples. Investigate extraction.")
