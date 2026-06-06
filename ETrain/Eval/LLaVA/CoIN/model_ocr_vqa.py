import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# SDPA monkey patch must be applied before any transformers model is loaded.
if os.environ.get("COIN_USE_SDPA_PATCH", "0") == "1":
    from ETrain.Train.LLaVA.attn_sdpa_eval import replace_llama_attn_with_sdpa
    replace_llama_attn_with_sdpa()

from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ETrain.utils.LLaVA.conversation import conv_templates, SeparatorStyle
from ETrain.Models.LLaVA.builder import load_pretrained_model
from ETrain.utils.LLaVA.utils import disable_torch_init
from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

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


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def collate_fn_left_pad(batch):
    input_ids_list, image_tensors = zip(*batch)
    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded, masks = [], []
    for ids in input_ids_list:
        pl = max_len - ids.shape[0]
        if pl > 0:
            ids = torch.cat([torch.full((pl,), 0, dtype=ids.dtype), ids])
            mask = torch.cat([torch.zeros(pl, dtype=torch.long), torch.ones(max_len - pl, dtype=torch.long)])
        else:
            mask = torch.ones(max_len, dtype=torch.long)
        padded.append(ids)
        masks.append(mask)
    return torch.stack(padded), torch.stack(image_tensors), torch.stack(masks)

def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config,
                       batch_size=1, num_workers=4):
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    if batch_size == 1:
        return DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, collate_fn=collate_fn_left_pad)


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

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    batch_size = args.batch_size
    if batch_size > 1:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.tokenizer_padding_side = "left"

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor,
                                     model.config, batch_size=batch_size)

    q_iter = iter(questions)
    for batch in tqdm(data_loader, total=math.ceil(len(questions) / batch_size)):
        if batch_size == 1:
            input_ids, image_tensor = batch
            attention_mask = None
            batch_lines = [next(q_iter)]
        else:
            input_ids, image_tensor, attention_mask = batch
            batch_lines = [next(q_iter) for _ in range(input_ids.shape[0])]

        if attention_mask is not None and tokenizer.pad_token_id is not None and tokenizer.pad_token_id != 0:
            input_ids[attention_mask == 0] = tokenizer.pad_token_id

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        images = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device='cuda', non_blocking=True)

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
        for i, line in enumerate(batch_lines):
            idx = line["question_id"]
            cur_prompt = line["text"]
            n_diff = (input_ids[i] != output_ids[i, :input_token_len]).sum().item()
            if n_diff > 0:
                print(f'[Warning] {n_diff} output_ids differ for question {idx}')
            text = tokenizer.decode(output_ids[i, input_token_len:], skip_special_tokens=True).strip()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "text": text,
                                       "answer_id": shortuuid.uuid(),
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
    ans_file.close()

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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--merge-lora", type=str2bool, default=True)
    parser.add_argument(
        "--lora-mode",
        type=str,
        default="all",
        choices=["all", "text", "vision"],
    )
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Samples per forward pass. >1 enables batched inference.")
    args = parser.parse_args()

    eval_model(args)
