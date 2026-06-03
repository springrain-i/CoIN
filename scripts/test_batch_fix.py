#!/usr/bin/env python3
"""Quick test: verify batch_size=4 now gives same accuracy as batch_size=1 after the fix.

Run: CUDA_VISIBLE_DEVICES=1 python scripts/test_batch_fix.py
"""
import os, sys, json, torch, subprocess
sys.path.insert(0, "/data4/home/sqx/CoIN")
os.environ.setdefault("COIN_USE_SDPA_PATCH", "1")

CKPT    = "/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN_MoEMoKA/OCRVQA_llava_MoEMoKA_lora"
BASE    = "/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5"
SQA_Q   = "/data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/ScienceQA/test.json"
IMG_DIR = "/data4/wxl/MoBLoRA-backup/CoIN/cl_dataset"
CHUNKS  = 1
CHUNK   = 0
OUT_B1  = "/tmp/sqa_test_b1.jsonl"
OUT_B4  = "/tmp/sqa_test_b4.jsonl"
EVAL_PY = "ETrain.Eval.LLaVA.CoIN.eval_science_qa"
VQA_PY  = "ETrain.Eval.LLaVA.CoIN.model_vqa_science"
PYTHON  = "/data4/home/sqx/.conda/envs/coin/bin/python"

import math
# Load and take first 40 questions
with open(SQA_Q) as f:
    qs = json.load(f)
mini = qs[:40]
mini_f = "/tmp/sqa_mini40.json"
with open(mini_f, "w") as f:
    json.dump(mini, f)

print("Testing 40 SciQA questions: batch_size=1 vs batch_size=4")

for bs, out in [(1, OUT_B1), (4, OUT_B4)]:
    cmd = [
        PYTHON, "-m", VQA_PY,
        "--model-path", CKPT,
        "--model-base", BASE,
        "--question-file", mini_f,
        "--image-folder", IMG_DIR,
        "--answers-file", out,
        "--num-chunks", str(CHUNKS),
        "--chunk-idx", str(CHUNK),
        "--temperature", "0",
        "--max-new-tokens", "10",
        "--merge-lora", "False",
        "--lora-mode", "all",
        "--conv-mode", "vicuna_v1",
        "--batch-size", str(bs),
    ]
    print(f"\nRunning batch_size={bs} ...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
    else:
        print("OK")

# Compare
def score(path):
    with open(path) as f:
        ans = [json.loads(l) for l in f]
    with open(SQA_Q) as f:
        qs = json.load(f)
    qmap = {q["question_id"]: q for q in qs}
    correct = garbage = total = 0
    for a in ans:
        q = qmap.get(a["question_id"], {})
        expected = q.get("answer", "?")
        text = a["text"].strip()
        pred = text[0].upper() if text else ""
        if pred not in "ABCDE":
            garbage += 1
        if pred == expected:
            correct += 1
        total += 1
    return correct, garbage, total

c1, g1, t1 = score(OUT_B1)
c4, g4, t4 = score(OUT_B4)
print(f"\n=== Results ===")
print(f"batch_size=1: {c1}/{t1} = {100*c1/t1:.1f}%  garbage={g1}")
print(f"batch_size=4: {c4}/{t4} = {100*c4/t4:.1f}%  garbage={g4}")
if g4 == 0 and abs(c1-c4) <= 2:
    print("PASS: batch_size=4 fixed, results match batch_size=1")
else:
    print("FAIL: still issues with batch_size=4")
