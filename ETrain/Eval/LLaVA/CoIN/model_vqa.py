import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

# SDPA monkey patch must be applied before any transformers model is loaded.
if os.environ.get("COIN_USE_SDPA_PATCH", "0") == "1":
    from ETrain.Train.LLaVA.attn_sdpa_eval import replace_llama_attn_with_sdpa
    replace_llama_attn_with_sdpa()

from ETrain.utils.LLaVA.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ETrain.utils.LLaVA.conversation import conv_templates, SeparatorStyle
from ETrain.Models.LLaVA.builder import load_pretrained_model
from ETrain.utils.LLaVA.utils import disable_torch_init
from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from ETrain.Models.LLaVA import *
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_checkpoint_cache(answers_dir):
    """
    从 answers_dir/checkpoints.jsonl 中加载已推理的结果。
    返回 dict: {question_id -> record_dict}，若文件不存在则返回空 dict。
    """
    checkpoint_path = os.path.join(answers_dir, "checkpoints.jsonl")
    cache = {}
    if not os.path.exists(checkpoint_path):
        return cache, checkpoint_path

    with open(checkpoint_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                qid = record["question_id"]
                cache[qid] = record
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"[Resume] Loaded {len(cache)} cached records from {checkpoint_path}")
    return cache, checkpoint_path


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        merge_lora=args.merge_lora
    )
    model.lora_mode = args.lora_mode
    if hasattr(model, "base_model"):
        model.base_model.lora_mode = args.lora_mode
        if hasattr(model.base_model, "model"):
            model.base_model.model.lora_mode = args.lora_mode
    lora_total = 0
    lora_active = 0
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_total += 1
            active = getattr(module, "active_adapter", None)
            if active is not None and active in getattr(module, "lora_A", {}):
                if getattr(module, "r", {}).get(active, 0) > 0:
                    lora_active += 1
    print(f"LoRA module check: total={lora_total}, active_with_r>0={lora_active}")

    # Set tokenizer to left-pad for batched generation
    orig_padding_side = tokenizer.padding_side
    if args.batch_size > 1:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.tokenizer_padding_side = "left"

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    os.makedirs(answers_dir, exist_ok=True)

    # ── 断点续推：加载 checkpoints.jsonl ──────────────────────────────────
    checkpoint_cache, checkpoint_path = load_checkpoint_cache(answers_dir)
    # ─────────────────────────────────────────────────────────────────────

    ans_file = open(answers_file, "w")
    count = 0
    hit = 0
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(questions), batch_size)):
        batch_lines = questions[batch_start: batch_start + batch_size]

        # Filter cached samples out
        to_infer = []
        for line in batch_lines:
            count += 1
            idx = line["question_id"]
            if idx in checkpoint_cache:
                ans_file.write(json.dumps(checkpoint_cache[idx]) + "\n")
                hit += 1
            else:
                to_infer.append(line)
        if not to_infer:
            ans_file.flush()
            continue

        # Build per-sample prompts and image tensors
        samples = []
        for line in to_infer:
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            cur_prompt = qs

            if model.config.mm_use_im_start_end:
                qs_img = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs_img = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_img)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].half().cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            samples.append({
                "idx": idx,
                "cur_prompt": cur_prompt,
                "input_ids": input_ids,
                "image_tensor": image_tensor,
                "stop_str": stop_str,
            })

        if len(samples) == 1:
            # Single-sample path (avoids padding overhead)
            s = samples[0]
            input_ids = s["input_ids"].unsqueeze(0).cuda()
            images = s["image_tensor"].unsqueeze(0)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
            input_token_len = input_ids.shape[1]
            n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff > 0:
                print(f'[Warning] {n_diff} output_ids are not the same as the input_ids')
            text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
            if text.endswith(s["stop_str"]):
                text = text[:-len(s["stop_str"])]
            ans_file.write(json.dumps({"question_id": s["idx"], "prompt": s["cur_prompt"],
                                       "text": text.strip(), "answer_id": shortuuid.uuid(),
                                       "model_id": model_name, "metadata": {}}) + "\n")
        else:
            # Batched path: left-pad all input_ids
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            max_len = max(s["input_ids"].shape[0] for s in samples)
            padded_ids = []
            attention_masks = []
            for s in samples:
                ids = s["input_ids"]
                pad_len = max_len - ids.shape[0]
                if pad_len > 0:
                    pad = torch.full((pad_len,), pad_id, dtype=ids.dtype)
                    ids = torch.cat([pad, ids], dim=0)
                    mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                      torch.ones(max_len - pad_len, dtype=torch.long)])
                else:
                    mask = torch.ones(max_len, dtype=torch.long)
                padded_ids.append(ids)
                attention_masks.append(mask)
            input_ids = torch.stack(padded_ids, dim=0).cuda()
            attention_mask = torch.stack(attention_masks, dim=0).cuda()
            images = torch.stack([s["image_tensor"] for s in samples], dim=0)
            stop_str = samples[0]["stop_str"]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            for i, s in enumerate(samples):
                text = tokenizer.decode(output_ids[i, input_token_len:], skip_special_tokens=True).strip()
                if text.endswith(s["stop_str"]):
                    text = text[:-len(s["stop_str"])]
                ans_file.write(json.dumps({"question_id": s["idx"], "prompt": s["cur_prompt"],
                                           "text": text.strip(), "answer_id": shortuuid.uuid(),
                                           "model_id": model_name, "metadata": {}}) + "\n")

        ans_file.flush()

    ans_file.close()
    tokenizer.padding_side = orig_padding_side

    if checkpoint_cache:
        print(f"[Resume] {hit}/{count} samples loaded from cache, {count - hit} newly inferred.")

    lora_stats = None
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(model, "model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "_lora_token_stats"):
            lora_stats = getattr(candidate, "_lora_token_stats")
            break

    if lora_stats is not None:
        nonpad = max(1, int(lora_stats.get("nonpad", 0)))
        vision = int(lora_stats.get("vision", 0))
        text = int(lora_stats.get("text", 0))
        vision_pct = vision * 100.0 / nonpad
        text_pct = text * 100.0 / nonpad
        print(
            f"LoRA token stats (dataset): text={text} ({text_pct:.2f}%), "
            f"vision={vision} ({vision_pct:.2f}%), nonpad={nonpad}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--merge-lora", type=str2bool, default=True)
    parser.add_argument(
        "--lora-mode",
        type=str,
        default="all",
        choices=["all", "text", "vision"],
    )
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of samples per forward pass. >1 enables batched inference.")
    args = parser.parse_args()

    eval_model(args)
