#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/Instruction/Only_Pretrain_1.5/OCRVQA/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi

if [ ! -n "$3" ] ;then
    LORA_MODE='all'
else
    LORA_MODE=$3
fi

if [ "$LORA_MODE" = "visual" ]; then
    LORA_MODE='vision'
fi

BATCH_SIZE="${COIN_EVAL_BATCH_SIZE:-1}"

RESULT_DIR="./results/CoIN/LLaVA/OCRVQA"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_ocr_vqa \
        --model-path $MODELPATH \
        --model-base /data4/wxl/MoBLoRA-backup/CoIN/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
        --question-file /data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/OCRVQA/test.json \
        --image-folder /data4/wxl/MoBLoRA-backup/CoIN/cl_dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --merge-lora False \
        --lora-mode $LORA_MODE \
        --batch-size $BATCH_SIZE \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.eval_ocrvqa \
    --annotation-file /data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/OCRVQA/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
    --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
    --questions /data4/wxl/MoBLoRA-backup/CoIN/playground/Instructions_Original/OCRVQA/test.json \
    --results $output_file \
    --rule_temp CoIN