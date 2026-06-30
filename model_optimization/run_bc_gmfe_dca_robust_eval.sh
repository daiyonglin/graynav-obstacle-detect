#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VAL_IMAGES="${VAL_IMAGES:-/root/autodl-tmp/datasets/coco/val2017}"
VAL_ANN="${VAL_ANN:-/root/autodl-tmp/datasets/coco/annotations/instances_val2017.json}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
ADAPTER="${ADAPTER:-artifacts/bc_gmfe_dca/bc_gmfe_dca_best.pt}"
OUT_ROOT="${OUT_ROOT:-artifacts/bc_gmfe_dca_robust_eval}"

IMG_SIZE="${IMG_SIZE:-384}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-12}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-500}"
SAVE_VISUALS="${SAVE_VISUALS:-12}"
SAVE_INPUT_SAMPLES="${SAVE_INPUT_SAMPLES:-6}"

for CORRUPTION in normal low_light high_exposure low_contrast motion_blur noise; do
  EVAL_DIR="${OUT_ROOT}/${CORRUPTION}"
  python scripts/evaluate_gray_adapters_coco.py \
    --images "$VAL_IMAGES" \
    --annotations "$VAL_ANN" \
    --weights "$WEIGHTS" \
    --bc-gmfe-dca-adapter "$ADAPTER" \
    --out-dir "$EVAL_DIR" \
    --imgsz "$IMG_SIZE" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --max-images "$EVAL_MAX_IMAGES" \
    --variants baseline,bc_gmfe_dca \
    --gray-corruption "$CORRUPTION" \
    --save-visuals "$SAVE_VISUALS" \
    --save-input-samples "$SAVE_INPUT_SAMPLES"

  python scripts/summarize_adapter_truth_eval.py \
    --eval-dir "$EVAL_DIR" \
    --out "$EVAL_DIR/report.md"
done

echo "BC-GMFE-DCA robust eval done: ${OUT_ROOT}"
