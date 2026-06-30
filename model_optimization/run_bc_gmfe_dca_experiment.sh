#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAIN_IMAGES="${TRAIN_IMAGES:-/root/autodl-tmp/datasets/coco/train2017}"
VAL_IMAGES="${VAL_IMAGES:-/root/autodl-tmp/datasets/coco/val2017}"
VAL_ANN="${VAL_ANN:-/root/autodl-tmp/datasets/coco/annotations/instances_val2017.json}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
OUT_DIR="${OUT_DIR:-artifacts/bc_gmfe_dca}"
EVAL_DIR="${EVAL_DIR:-artifacts/bc_gmfe_dca_truth_eval}"

IMG_SIZE="${IMG_SIZE:-384}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-12}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-1}"
MAX_IMAGES="${MAX_IMAGES:-256}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-200}"
ACCUMULATE="${ACCUMULATE:-1}"
HIDDEN="${HIDDEN:-16}"
ALPHA="${ALPHA:-0.1}"
LR="${LR:-0.0005}"
EXTRA_TRAIN_ARGS=()
if [[ "${STRONG_MONOSIM:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--strong-monosim)
fi

python scripts/train_bc_gmfe_dca.py \
  --source "$TRAIN_IMAGES" \
  --weights "$WEIGHTS" \
  --out-dir "$OUT_DIR" \
  --imgsz "$IMG_SIZE" \
  --device "$DEVICE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --epochs "$EPOCHS" \
  --max-images "$MAX_IMAGES" \
  --accumulate "$ACCUMULATE" \
  --hidden "$HIDDEN" \
  --alpha "$ALPHA" \
  --lr "$LR" \
  "${EXTRA_TRAIN_ARGS[@]}"

python scripts/evaluate_gray_adapters_coco.py \
  --images "$VAL_IMAGES" \
  --annotations "$VAL_ANN" \
  --weights "$WEIGHTS" \
  --bc-gmfe-dca-adapter "${OUT_DIR}/bc_gmfe_dca_best.pt" \
  --out-dir "$EVAL_DIR" \
  --imgsz "$IMG_SIZE" \
  --device "$DEVICE" \
  --batch "$BATCH" \
  --max-images "$EVAL_MAX_IMAGES" \
  --variants baseline,bc_gmfe_dca \
  --save-visuals "${SAVE_VISUALS:-24}" \
  --save-input-samples "${SAVE_INPUT_SAMPLES:-12}"

python scripts/summarize_adapter_truth_eval.py \
  --eval-dir "$EVAL_DIR" \
  --out "$EVAL_DIR/report.md"

if [[ "${EXPORT_ONNX:-0}" == "1" ]]; then
  python scripts/export_bc_gmfe_dca_yolov8.py \
    --adapter "${OUT_DIR}/bc_gmfe_dca_best.pt" \
    --weights "$WEIGHTS" \
    --out-dir "$OUT_DIR" \
    --imgsz "$IMG_SIZE" \
    --device "$DEVICE"
fi

echo "BC-GMFE-DCA experiment done."
echo "Adapter: ${OUT_DIR}/bc_gmfe_dca_best.pt"
echo "Eval summary: ${EVAL_DIR}/truth_eval_summary.json"
echo "Report: ${EVAL_DIR}/report.md"
