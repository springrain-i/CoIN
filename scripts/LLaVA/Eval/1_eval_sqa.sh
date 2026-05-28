#!/bin/bash

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

MODELPATH='/data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/CoIN/ScienceQA_llava_lora_rank_128'

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


RESULT_DIR="./results/CoIN/LLaVA/ScienceQA_NoMerge_Visual"
#RESULT_DIR="./results/CoIN/LLaVA/ScienceQA_modified"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"  # batch=4: 2.57x speedup vs batch=1 on 24GB GPU

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_vqa_science \
        --model-path $MODELPATH \
        --model-base $BASE_MODEL_PATH \
        --question-file /data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/ScienceQA/test.json \
        --image-folder /data4/wxl/MoBLoRA-backup/CoIN/cl_dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --merge-lora False \
        --lora-mode $LORA_MODE \
        --max_new_tokens 10 \
        --conv-mode vicuna_v1 \
        --batch-size "${EVAL_BATCH_SIZE}" &
done
#lora-mode的三个选项: all, text, visual
wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.eval_science_qa \
    --base-dir /data4/wxl/MoBLoRA-backup/CoIN/cl_dataset/ScienceQA \
    --result-file $output_file \
    --output-file $RESULT_DIR/$STAGE/output.jsonl \
    --output-result $RESULT_DIR/$STAGE/output_result.jsonl \

