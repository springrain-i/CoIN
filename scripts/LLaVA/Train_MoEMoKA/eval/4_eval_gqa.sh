#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/eval_common.sh"

RESULT_DIR="${RESULT_ROOT}/GQA"
mkdir -p "${RESULT_DIR}/${STAGE}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_gqa \
        --model-path "$MODELPATH" \
        --model-base "$BASE_MODEL_PATH" \
        --question-file "${INSTR_ROOT}/GQA/test.json" \
        --image-folder "$IMAGE_ROOT" \
        --answers-file "${RESULT_DIR}/${STAGE}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
        --merge-lora False \
        --lora-mode "$LORA_MODE" \
        --conv-mode vicuna_v1 &
done
wait

output_file="${RESULT_DIR}/${STAGE}/merge.jsonl"
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${RESULT_DIR}/${STAGE}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.convert_gqa_for_eval \
    --src "$output_file" \
    --dst "${RESULT_DIR}/${STAGE}/testdev_balanced_predictions.json"

python -m ETrain.Eval.LLaVA.CoIN.eval_gqa \
    --tier testdev_balanced \
    --path "${RESULT_DIR}/${STAGE}" \
    --questions-dir "${IMAGE_ROOT}/GQA" \
    --output-dir "${RESULT_DIR}/${STAGE}"

echo "[eval][GQA] mode=${LORA_MODE} stage=${STAGE} -> ${RESULT_DIR}/${STAGE}"
