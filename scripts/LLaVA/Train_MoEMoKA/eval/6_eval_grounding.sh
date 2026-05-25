#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/eval_common.sh"

RESULT_DIR="${RESULT_ROOT}/Grounding"
mkdir -p "${RESULT_DIR}/${STAGE}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} "${PYTHON}" -m ETrain.Eval.LLaVA.CoIN.model_vqa \
        --model-path "$MODELPATH" \
        --model-base "$BASE_MODEL_PATH" \
        --question-file "${INSTR_ROOT}/Grounding/test.json" \
        --image-folder "$IMAGE_ROOT" \
        --answers-file "${RESULT_DIR}/${STAGE}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --max_new_tokens "${MAX_NEW_TOKENS:-50}" \
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

"${PYTHON}" -m ETrain.Eval.LLaVA.CoIN.eval_grounding \
    --test-file "${INSTR_ROOT}/Grounding/test.json" \
    --result-file "$output_file" \
    --output-dir "${RESULT_DIR}/${STAGE}"

echo "[eval][Grounding] mode=${LORA_MODE} stage=${STAGE} -> ${RESULT_DIR}/${STAGE}"
