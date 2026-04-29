#!/bin/bash
# Full T1-T8 continual learning pipeline test with online eval + final eval matrix.
#
# Flow:
#   For i in 1..8:
#     train task i (MAX_STEPS steps)
#     eval task i with current checkpoint   → "online" result
#   After all 8 tasks:
#     eval all 8 tasks with T8 checkpoint  → "final" eval matrix
#
# Mini mode by default (fast, for verification):
#   - MAX_STEPS=3      training steps per task
#   - EVAL_QUESTIONS=50  eval questions per task
#   - MAX_NEW_TOKENS=32  tokens per generated answer
#
# Usage (from repo root):
#   bash scripts/LLaVA/Train_MoEMoKA/full_pipeline_test.sh
#
# Overrides:
#   MAX_STEPS=5 EVAL_QUESTIONS=100 bash scripts/LLaVA/Train_MoEMoKA/full_pipeline_test.sh
#   SKIP_ONLINE_EVAL=1   bash ...    # skip per-task eval, only do final matrix
#   SKIP_FINAL_EVAL=1    bash ...    # skip final matrix
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

# ── Tunable knobs ─────────────────────────────────────────────────────────────
MAX_STEPS="${MAX_STEPS:-3}"
EVAL_QUESTIONS="${EVAL_QUESTIONS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SKIP_ONLINE_EVAL="${SKIP_ONLINE_EVAL:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
PROMPT_VERSION=v1

FULL_OUTPUT_ROOT="${COIN_OUTPUT_ROOT}/full_pipeline_test"
RESULT_ROOT_BASE="${COIN_REPO_ROOT}/results/CoIN/LLaVA/MoEMoKA/full_pipeline_test"
MINI_INSTR_ROOT="${FULL_OUTPUT_ROOT}/mini_instr"
LOG_FILE="${FULL_OUTPUT_ROOT}/pipeline.log"

TASK_NAMES=(ScienceQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCRVQA)

# Question files used for inference (relative to INSTR_ROOT)
# Note: VQAv2 uses val.json, the rest use test.json; VizWiz/TextVQA use val.json
TASK_QFILES=(
  "ScienceQA/test.json"
  "TextVQA/val.json"
  "ImageNet/test.json"
  "GQA/test.json"
  "VizWiz/val.json"
  "Grounding/test.json"
  "VQAv2/val.json"
  "OCRVQA/test.json"
)

TASK_TRAIN_DATA=(
  "${COIN_INSTR_ROOT}/ScienceQA/train.json"
  "${COIN_INSTR_ROOT}/TextVQA/train.json"
  "${COIN_INSTR_ROOT}/ImageNet/train.json"
  "${COIN_INSTR_ROOT}/GQA/train.json"
  "${COIN_INSTR_ROOT}/VizWiz/train.json"
  "${COIN_INSTR_ROOT}/Grounding/train.json"
  "${COIN_INSTR_ROOT}/VQAv2/train.json"
  "${COIN_INSTR_ROOT}/OCRVQA/train.json"
)

EVAL_SCRIPTS=(
  "${SCRIPT_DIR}/eval/1_eval_sqa.sh"
  "${SCRIPT_DIR}/eval/2_eval_textqa.sh"
  "${SCRIPT_DIR}/eval/3_eval_imagenet.sh"
  "${SCRIPT_DIR}/eval/4_eval_gqa.sh"
  "${SCRIPT_DIR}/eval/5_eval_vizwiz.sh"
  "${SCRIPT_DIR}/eval/6_eval_grounding.sh"
  "${SCRIPT_DIR}/eval/7_eval_vqav2.sh"
  "${SCRIPT_DIR}/eval/8_eval_ocrvqa.sh"
)

mkdir -p "${FULL_OUTPUT_ROOT}" "${RESULT_ROOT_BASE}"

# Tee all output to log
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "[pipeline] CoIN MoE-MoKA full pipeline test"
echo "[pipeline] $(date)"
echo "[pipeline] MAX_STEPS=${MAX_STEPS}  EVAL_QUESTIONS=${EVAL_QUESTIONS}  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[pipeline] Output root : ${FULL_OUTPUT_ROOT}"
echo "[pipeline] Result root : ${RESULT_ROOT_BASE}"
echo "[pipeline] Log         : ${LOG_FILE}"
echo "============================================================"

# ── Pre-flight checks ────────────────────────────────────────────────────────
require_path "${COIN_BASE_MODEL}"         "base model"
require_path "${COIN_PRETRAIN_PROJECTOR}" "mm projector"
require_path "${COIN_VISION_TOWER}"       "vision tower"
for i in "${!TASK_TRAIN_DATA[@]}"; do
    require_path "${TASK_TRAIN_DATA[$i]}" "${TASK_NAMES[$i]} train data"
done
for i in "${!TASK_QFILES[@]}"; do
    require_path "${COIN_INSTR_ROOT}/${TASK_QFILES[$i]}" "${TASK_NAMES[$i]} eval data"
done

# ── Create mini eval question files (EVAL_QUESTIONS each) ───────────────────
echo ""
echo "[pipeline] Creating mini eval files (${EVAL_QUESTIONS} questions each) ..."
for i in "${!TASK_NAMES[@]}"; do
    TASK="${TASK_NAMES[$i]}"
    QFILE="${TASK_QFILES[$i]}"
    SRC="${COIN_INSTR_ROOT}/${QFILE}"
    DST_DIR="${MINI_INSTR_ROOT}/$(dirname "${QFILE}")"
    DST="${MINI_INSTR_ROOT}/${QFILE}"
    mkdir -p "${DST_DIR}"
    ${COIN_PYTHON} -c "
import json, sys
data = json.load(open('${SRC}'))
n = min(${EVAL_QUESTIONS}, len(data))
json.dump(data[:n], open('${DST}', 'w'))
print(f'  ${TASK}: {n} questions -> ${DST}')
"
done
echo "[pipeline] Mini eval files ready."

# Export so eval scripts pick it up via coin_paths.sh
export COIN_INSTR_ROOT="${MINI_INSTR_ROOT}"
export MAX_NEW_TOKENS

# ── Helpers ──────────────────────────────────────────────────────────────────
run_eval() {
    local task_idx="$1"   # 0-based
    local stage="$2"      # e.g. "after_task_1"
    local modelpath="$3"
    local script="${EVAL_SCRIPTS[$task_idx]}"
    local task="${TASK_NAMES[$task_idx]}"
    echo ""
    echo "  ── eval ${task} [${stage}] ──"
    if bash "${script}" "${stage}" "${modelpath}" "all"; then
        echo "  [PASS] eval ${task} ${stage}"
    else
        echo "  [WARN] eval ${task} ${stage} exited with non-zero (continuing)" >&2
    fi
}

# Summary table accumulator
declare -a ONLINE_RESULTS=()
declare -a FINAL_RESULTS=()

# ── Training + online eval loop ───────────────────────────────────────────────
PREV_CKPT=""
for i in "${!TASK_NAMES[@]}"; do
    TASK="${TASK_NAMES[$i]}"
    DATA="${TASK_TRAIN_DATA[$i]}"
    OUT_DIR="${FULL_OUTPUT_ROOT}/${TASK}_llava_MoEMoKA_lora"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[pipeline] TRAIN  T$((i+1))/8 — ${TASK}  (${MAX_STEPS} steps)"
    [[ -n "$PREV_CKPT" ]] && echo "[pipeline]   prev adapter: ${PREV_CKPT}"
    echo "[pipeline] $(date)"
    echo "════════════════════════════════════════════════════════════"

    PREV_CKPT_ARGS=()
    [[ -n "$PREV_CKPT" ]] && PREV_CKPT_ARGS=(--previous_task_model_path "$PREV_CKPT")

    ${COIN_DEEPSPEED} --include "${COIN_DS_INCLUDE}" --master_port 29605 \
        ETrain/Train/LLaVA/train_mem.py \
        --deepspeed "${COIN_DS_CONFIG}" \
        --moe_moka_enable True \
        --lora_r 32 \
        --lora_alpha 64 \
        --mm_projector_lr 2e-5 \
        --expert_num 8 \
        --model_name_or_path "${COIN_BASE_MODEL}" \
        --pretrain_mm_mlp_adapter "${COIN_PRETRAIN_PROJECTOR}" \
        --version "${PROMPT_VERSION}" \
        --data_path "${DATA}" \
        --image_folder "${COIN_IMAGE_ROOT}" \
        --vision_tower "${COIN_VISION_TOWER}" \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir "${OUT_DIR}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 512 \
        --gradient_checkpointing True \
        --dataloader_num_workers 0 \
        --lazy_preprocess True \
        --max_steps "${MAX_STEPS}" \
        --report_to none \
        "${PREV_CKPT_ARGS[@]}"

    # Verify checkpoint
    for f in adapter_model.bin adapter_config.json non_lora_trainables.bin; do
        if [[ ! -f "${OUT_DIR}/${f}" ]]; then
            echo "[FAIL] Missing ${f} in ${OUT_DIR}" >&2; exit 1
        fi
    done
    echo "[pipeline] T$((i+1)) checkpoint: PASS"

    PREV_CKPT="${OUT_DIR}"

    # Online eval: evaluate current task only
    if [[ "$SKIP_ONLINE_EVAL" != "1" ]]; then
        echo ""
        echo "[pipeline] ONLINE EVAL after T$((i+1)) — ${TASK}"
        STAGE="after_task_$((i+1))"
        run_eval "$i" "${STAGE}" "${OUT_DIR}"
        ONLINE_RESULTS+=("T$((i+1))_${TASK}:${STAGE}")
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[pipeline] All 8 training tasks complete. $(date)"
echo "════════════════════════════════════════════════════════════"

# ── Final eval matrix: eval all 8 tasks with T8 checkpoint ───────────────────
if [[ "$SKIP_FINAL_EVAL" != "1" ]]; then
    FINAL_CKPT="${FULL_OUTPUT_ROOT}/OCRVQA_llava_MoEMoKA_lora"
    FINAL_STAGE="final_eval"

    echo ""
    echo "[pipeline] FINAL EVAL MATRIX (T8 checkpoint → all 8 tasks)"
    echo "[pipeline] Checkpoint: ${FINAL_CKPT}"

    for i in "${!TASK_NAMES[@]}"; do
        run_eval "$i" "${FINAL_STAGE}" "${FINAL_CKPT}"
        FINAL_RESULTS+=("T$((i+1))_${TASK_NAMES[$i]}")
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "[pipeline] DONE — $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Online eval results:"
for entry in "${ONLINE_RESULTS[@]:-}"; do echo "  ${entry}"; done

echo ""
echo "Final eval results:"
for entry in "${FINAL_RESULTS[@]:-}"; do
    TASK="${entry#*_}"
    echo "  ${entry} -> ${RESULT_ROOT_BASE}/${TASK}/final_eval/"
done

echo ""
echo "[pipeline] Full log: ${LOG_FILE}"
echo "[pipeline] Checkpoints: ${FULL_OUTPUT_ROOT}/"
echo "[pipeline] Results: ${RESULT_ROOT_BASE}/"
echo ""
echo "Accuracy summary (grep from result files):"
find "${RESULT_ROOT_BASE}" -name "*.jsonl" -o -name "accuracy.txt" 2>/dev/null \
    | xargs grep -l "accuracy\|Accuracy\|acc" 2>/dev/null \
    | while read f; do
        echo "  $f:"
        grep -i "accuracy\|acc" "$f" | tail -3
    done || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "[pipeline] PASS"
echo "════════════════════════════════════════════════════════════"
