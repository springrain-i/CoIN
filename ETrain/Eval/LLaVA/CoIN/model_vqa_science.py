import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ETrain.utils.LLaVA.conversation import conv_templates, SeparatorStyle
from ETrain.Models.LLaVA.builder import load_pretrained_model
from ETrain.utils.LLaVA.utils import disable_torch_init
from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def _prepare_batch(lines, args, tokenizer, image_processor, model):
    """Prepare a batch of samples. Returns per-sample dicts with padded input_ids."""
    samples = []
    for line in lines:
        idx = line["question_id"]
        question = line['text']
        qs = question.replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor = image_tensor.half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            image_tensor = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        samples.append({
            "idx": idx,
            "cur_prompt": cur_prompt,
            "input_ids": input_ids,
            "image_tensor": image_tensor,
            "stop_str": stop_str,
            "conv_version": conv.version,
            "line": line,
            "prompt": prompt,
        })
    return samples


def _left_pad_batch(samples, tokenizer):
    """Left-pad input_ids to the same length for batched generation."""
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
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(ids.shape[0] - pad_len, dtype=torch.long)])
        else:
            mask = torch.ones(ids.shape[0], dtype=torch.long)
        padded_ids.append(ids)
        attention_masks.append(mask)
    input_ids = torch.stack(padded_ids, dim=0).cuda()
    attention_mask = torch.stack(attention_masks, dim=0).cuda()
    return input_ids, attention_mask


def eval_model(args):
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

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    batch_size = args.batch_size

    # Pre-sort: image samples first, then text-only. Avoids mixed batches so
    # the batched path is taken consistently throughout.
    if batch_size > 1:
        questions = sorted(questions, key=lambda x: 0 if 'image' in x else 1)

    for batch_start in tqdm(range(0, len(questions), batch_size)):
        batch_lines = questions[batch_start: batch_start + batch_size]

        # answer_prompter with batch > 1: fall back to single-sample for simplicity
        if args.answer_prompter and batch_size > 1:
            batch_size_eff = 1
        else:
            batch_size_eff = len(batch_lines)

        if batch_size_eff == 1:
            # Single-sample path (also used for answer_prompter)
            for line in batch_lines:
                samples = _prepare_batch([line], args, tokenizer, image_processor, model)
                s = samples[0]
                input_ids = s["input_ids"].unsqueeze(0).cuda()
                images = s["image_tensor"].unsqueeze(0) if s["image_tensor"] is not None else None
                stop_str = s["stop_str"]
                keywords = [stop_str]
                stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if s["conv_version"] == "v0" else None

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        images=images,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=stopping_criteria,
                    )

                input_token_len = input_ids.shape[1]
                n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff > 0:
                    print(f'[Warning] {n_diff} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                if args.answer_prompter:
                    outputs_reasoning = outputs
                    input_ids2 = tokenizer_image_token(
                        s["prompt"] + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                    ).unsqueeze(0).cuda()
                    with torch.inference_mode():
                        output_ids2 = model.generate(
                            input_ids=input_ids2,
                            images=images,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            max_new_tokens=64,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria])
                    input_token_len2 = input_ids2.shape[1]
                    n_diff2 = (input_ids2 != output_ids2[:, :input_token_len2]).sum().item()
                    if n_diff2 > 0:
                        print(f'[Warning] {n_diff2} output_ids are not the same as the input_ids')
                    outputs2 = tokenizer.batch_decode(output_ids2[:, input_token_len2:], skip_special_tokens=True)[0]
                    outputs2 = outputs2.strip()
                    if outputs2.endswith(stop_str):
                        outputs2 = outputs2[:-len(stop_str)]
                    outputs2 = outputs2.strip()
                    outputs = outputs_reasoning + '\n The answer is ' + outputs2

                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({
                    "question_id": s["idx"],
                    "prompt": s["cur_prompt"],
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }) + "\n")
                ans_file.flush()
        else:
            # Batched path
            samples = _prepare_batch(batch_lines, args, tokenizer, image_processor, model)

            # Stack images: all-image or all-no-image batch; mixed falls back gracefully
            has_image = [s["image_tensor"] is not None for s in samples]
            if all(has_image):
                images = torch.stack([s["image_tensor"] for s in samples], dim=0)
            elif not any(has_image):
                images = None
            else:
                # Mixed batch: process individually
                for s in samples:
                    single_ids = s["input_ids"].unsqueeze(0).cuda()
                    single_img = s["image_tensor"].unsqueeze(0) if s["image_tensor"] is not None else None
                    stop_str = s["stop_str"]
                    keywords = [stop_str]
                    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, single_ids)] if s["conv_version"] == "v0" else None
                    with torch.inference_mode():
                        out = model.generate(
                            input_ids=single_ids,
                            images=single_img,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            stopping_criteria=stopping_criteria,
                        )
                    text = tokenizer.batch_decode(out[:, single_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                    if text.endswith(stop_str):
                        text = text[:-len(stop_str)]
                    ans_file.write(json.dumps({
                        "question_id": s["idx"],
                        "prompt": s["cur_prompt"],
                        "text": text.strip(),
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {},
                    }) + "\n")
                ans_file.flush()
                continue

            input_ids, attention_mask = _left_pad_batch(samples, tokenizer)
            stop_str = samples[0]["stop_str"]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            input_token_len = input_ids.shape[1]
            for i, s in enumerate(samples):
                out_tokens = output_ids[i, input_token_len:]
                text = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
                if text.endswith(stop_str):
                    text = text[:-len(stop_str)]
                text = text.strip()
                ans_file.write(json.dumps({
                    "question_id": s["idx"],
                    "prompt": s["cur_prompt"],
                    "text": text,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_name,
                    "metadata": {},
                }) + "\n")
            ans_file.flush()

    ans_file.close()
    tokenizer.padding_side = orig_padding_side

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
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
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
