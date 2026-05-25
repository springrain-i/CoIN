#!/bin/bash


if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

MODELPATH='/data4/home/sqx/CoIN/checkpoints/LLaVA/CoIN/GQA_llava_MOE_lora'

LORA_MODE='all'
if [ -n "$2" ] ;then
    if [ "$2" = "all" ] || [ "$2" = "text" ] || [ "$2" = "vision" ] || [ "$2" = "visual" ]; then
        LORA_MODE=$2
    else
        MODELPATH=$2
    fi
fi

if [ -n "$3" ] ;then
    LORA_MODE=$3
fi

if [ "$LORA_MODE" = "visual" ]; then
    LORA_MODE='vision'
fi

export CUDA_VISIBLE_DEVICES=0,1,2,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

BASE_MODEL_PATH='/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5'
VISION_TOWER_PATH="/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/clip-vit-large-patch14-336"

RESULT_DIR="./results/CoIN/LLaVA/Final_MOE_only_vision"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_text_vqa \
        --model-path $MODELPATH \
        --model-base $BASE_MODEL_PATH \
        --question-file /data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/TextVQA/val.json \
        --image-folder /data4/wxl/MoBLoRA-backup/CoIN/cl_dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --merge-lora False \
        --lora-mode $LORA_MODE \
        --max_new_tokens 40 \
        --conv-mode vicuna_v1 &
done
# 三个选项: all, text, vision
wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.eval_textvqa \
    --annotation-file /data4/wxl/MoBLoRA-backup/CoIN/cl_dataset/TextVQA/TextVQA_0.5.1_val.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

