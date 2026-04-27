#!/bin/bash
# Mini end-to-end pipeline test: train all 8 CoIN tasks (3 steps each) + eval task 1.
# Purpose: verify that the full continual learning pipeline (train → save → load → train → …)
# works on the actual data files with real GPU(s) before running the full experiment.
#
# Usage (from repo root):
#   bash scripts/LLaVA/Train_MoEMoKA/mini_pipeline_test.sh
#
# Override number of steps per task:
#   MAX_STEPS=5 bash scripts/LLaVA/Train_MoEMoKA/mini_pipeline_test.sh
#
# Skip eval (train only):
#   SKIP_EVAL=1 bash scripts/LLaVA/Train_MoEMoKA/mini_pipeline_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/coin_paths.sh"

PROMPT_VERSION=v1
MAX_STEPS="${MAX_STEPS:-3}"
SKIP_EVAL="${SKIP_EVAL:-0}"
MINI_OUTPUT_ROOT="${COIN_OUTPUT_ROOT}/mini_pipeline_test"

TASK_NAMES=(ScienceQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCRVQA)
TASK_DATA=(
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

echo "============================================================"
echo "[mini_pipeline] CoIN MoE-MoKA mini pipeline test"
echo "[mini_pipeline] MAX_STEPS=${MAX_STEPS}  SKIP_EVAL=${SKIP_EVAL}"
echo "[mini_pipeline] Output root: ${MINI_OUTPUT_ROOT}"
echo "============================================================"

# Pre-flight: verify required paths exist
require_path "${COIN_BASE_MODEL}"          "base model"
require_path "${COIN_PRETRAIN_PROJECTOR}"  "mm projector"
require_path "${COIN_VISION_TOWER}"        "vision tower"

for i in "${!TASK_DATA[@]}"; do
    require_path "${TASK_DATA[$i]}" "${TASK_NAMES[$i]} train data"
done

mkdir -p "${MINI_OUTPUT_ROOT}"

# ── Training loop ─────────────────────────────────────────────
PREV_CKPT=""
for i in "${!TASK_NAMES[@]}"; do
    TASK="${TASK_NAMES[$i]}"
    DATA="${TASK_DATA[$i]}"
    OUT_DIR="${MINI_OUTPUT_ROOT}/${TASK}_lora"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "[mini_pipeline] Training T$((i+1)) / 8 — ${TASK}  (${MAX_STEPS} steps)"
    if [[ -n "$PREV_CKPT" ]]; then
        echo "[mini_pipeline]   loading previous adapter from: ${PREV_CKPT}"
    fi
    echo "────────────────────────────────────────────────────────────"

    PREV_CKPT_ARGS=()
    if [[ -n "$PREV_CKPT" ]]; then
        PREV_CKPT_ARGS=(--previous_task_model_path "$PREV_CKPT")
    fi

    ${COIN_DEEPSPEED} --include "${COIN_DS_INCLUDE}" --master_port 29603 \
        ETrain/Train/LLaVA/train_mem.py \
        --deepspeed ./scripts/zero3_offload.json \
        --moe_moka_enable True \
        --lora_r 32 \
        --lora_alpha 64 \
        --mm_projector_lr 2e-5 \
        --expert_num 4 \
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

    echo "[mini_pipeline] T$((i+1)) done. Files saved to: ${OUT_DIR}"
    ls -lh "${OUT_DIR}/"

    # Verify mandatory checkpoint files
    if [[ ! -f "${OUT_DIR}/adapter_model.bin" ]]; then
        echo "[FAIL] adapter_model.bin missing in ${OUT_DIR}" >&2
        exit 1
    fi
    if [[ ! -f "${OUT_DIR}/adapter_config.json" ]]; then
        echo "[FAIL] adapter_config.json missing in ${OUT_DIR}" >&2
        exit 1
    fi
    if [[ ! -f "${OUT_DIR}/non_lora_trainables.bin" ]]; then
        echo "[FAIL] non_lora_trainables.bin missing in ${OUT_DIR}" >&2
        exit 1
    fi
    echo "[mini_pipeline] T$((i+1)) checkpoint verification: PASS"

    PREV_CKPT="${OUT_DIR}"
done

echo ""
echo "============================================================"
echo "[mini_pipeline] All 8 training tasks complete."
echo "============================================================"

# ── Eval loop (task 1 only by default) ────────────────────────
if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "[mini_pipeline] SKIP_EVAL=1 — skipping eval."
    echo "[mini_pipeline] PASS"
    exit 0
fi

echo ""
echo "[mini_pipeline] Running eval for task 1 (ScienceQA) using T8 checkpoint..."
EVAL_MODEL="${MINI_OUTPUT_ROOT}/OCRVQA_lora"
EVAL_STAGE="mini_pipeline_test"

bash "${EVAL_SCRIPTS[0]}" "${EVAL_STAGE}" "${EVAL_MODEL}" "all"

echo ""
echo "============================================================"
echo "[mini_pipeline] PASS — full 8-task continual train + eval verified."
echo "============================================================"
