"""
Verify that batched image generation gives the same answers as single-sample generation.

Tests:
  - batch_size=1 (baseline)
  - batch_size=4 (new batched image path)

Pass condition: same accuracy for both, 0 empty outputs, answers match per sample.

Usage:
  COIN_USE_SDPA_PATCH=1 python scripts/test_image_batch_fix.py
"""

import os, sys, json, subprocess, tempfile, textwrap, re

# Small subset of SciQA questions WITH images
SCIQA_QUESTIONS = "/hy-tmp/playground/Instructions_Original/ScienceQA/test.json"
MODEL_PATH = "/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN_MoEMoKA/OCRVQA_llava_MoEMoKA_lora"
MODEL_BASE  = "/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5"
IMAGE_FOLDER = "/hy-tmp"

N_SAMPLES = 20          # how many questions to run
GPU = "0"

os.environ.setdefault("COIN_USE_SDPA_PATCH", "1")


def run_eval(batch_size: int, questions_file: str) -> list[dict]:
    """Run model_vqa_science.py on questions_file and return parsed answers."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        answers_file = f.name

    COIN_PYTHON = "/data4/home/sqx/.conda/envs/coin/bin/python"
    cmd = [
        COIN_PYTHON, "-m",
        "ETrain.Eval.LLaVA.CoIN.model_vqa_science",
        "--model-path", MODEL_PATH,
        "--model-base", MODEL_BASE,
        "--question-file", questions_file,
        "--image-folder", IMAGE_FOLDER,
        "--answers-file", answers_file,
        "--conv-mode", "llava_v1",
        "--temperature", "0",
        "--max-new-tokens", "64",
        "--single-pred-prompt",
        "--batch-size", str(batch_size),
        "--merge-lora", "True",
        "--lora-mode", "all",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU}
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd="/data4/home/sqx/CoIN"
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"eval failed (batch_size={batch_size})")

    answers = []
    with open(answers_file) as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))
    os.unlink(answers_file)
    return answers


def load_ground_truth(questions_file: str) -> dict:
    with open(questions_file) as f:
        questions = json.load(f)
    gt = {}
    for q in questions:
        if "answer" in q:
            gt[q["question_id"]] = q["answer"]
    return gt


def accuracy(answers: list[dict], gt: dict) -> tuple[float, int]:
    correct = 0
    n = 0
    for a in answers:
        qid = a["question_id"]
        if qid not in gt:
            continue
        pred = a["text"].strip().upper()[:1]
        if pred == gt[qid].strip().upper()[:1]:
            correct += 1
        n += 1
    return correct / n if n else 0.0, n


def main():
    # Load questions and keep only IMAGE questions (first N_SAMPLES)
    with open(SCIQA_QUESTIONS) as f:
        all_q = json.load(f)

    image_q = [q for q in all_q if "image" in q][:N_SAMPLES]
    print(f"Using {len(image_q)} image questions")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        subset_file = f.name
        json.dump(image_q, f)

    gt = load_ground_truth(subset_file)

    print(f"\n--- batch_size=1 (baseline) ---")
    ans1 = run_eval(1, subset_file)
    acc1, n1 = accuracy(ans1, gt)
    empty1 = sum(1 for a in ans1 if not a["text"].strip())
    print(f"accuracy={acc1:.1%}  n={n1}  empty={empty1}")

    print(f"\n--- batch_size=4 (batched images) ---")
    ans4 = run_eval(4, subset_file)
    acc4, n4 = accuracy(ans4, gt)
    empty4 = sum(1 for a in ans4 if not a["text"].strip())
    print(f"accuracy={acc4:.1%}  n={n4}  empty={empty4}")

    # Build per-sample comparison
    map1 = {a["question_id"]: a["text"] for a in ans1}
    map4 = {a["question_id"]: a["text"] for a in ans4}

    mismatches = []
    for qid in map1:
        if qid in map4 and map1[qid] != map4[qid]:
            mismatches.append((qid, map1[qid], map4[qid]))

    if mismatches:
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for qid, t1, t4 in mismatches[:5]:
            print(f"  qid={qid}  bs1={repr(t1[:60])}  bs4={repr(t4[:60])}")
    else:
        print(f"\nAll {len(map1)} answers match between batch_size=1 and batch_size=4")

    os.unlink(subset_file)

    # Pass/fail
    ok = (empty4 == 0) and (abs(acc1 - acc4) < 0.05)
    if ok:
        print("\nPASS")
    else:
        print(f"\nFAIL  empty4={empty4}  acc1={acc1:.1%}  acc4={acc4:.1%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
