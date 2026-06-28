#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SOURCE="${SOURCE:-/data/coco/train2017}"
GOLDEN_SOURCE="${GOLDEN_SOURCE:-$SOURCE}"
WEIGHTS="${WEIGHTS:-yolov8n.pt}"
ADAPTER="${ADAPTER:-lut}"
IMG_SIZE="${IMG_SIZE:-384}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS:-8}"
OUT_DIR="${OUT_DIR:-artifacts/gray_adapter_${ADAPTER}}"
GOLDEN_DIR="${GOLDEN_DIR:-artifacts/golden_yolov8n_gray}"

python scripts/generate_yolov8_golden_cases.py \
  --source "$GOLDEN_SOURCE" \
  --weights "$WEIGHTS" \
  --out-dir "$GOLDEN_DIR" \
  --imgsz "$IMG_SIZE" \
  --max-images "${GOLDEN_MAX_IMAGES:-100}" \
  --device "${GOLDEN_DEVICE:-cpu}"

python scripts/train_gray_adapter_distill.py \
  --source "$SOURCE" \
  --weights "$WEIGHTS" \
  --out-dir "$OUT_DIR" \
  --adapter "$ADAPTER" \
  --imgsz "$IMG_SIZE" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --accumulate "${ACCUMULATE:-1}" \
  --max-images "${MAX_IMAGES:-0}"

python scripts/export_gray_adapter_yolov8.py \
  --adapter "${OUT_DIR}/gray_adapter.pt" \
  --weights "$WEIGHTS" \
  --out-dir "$OUT_DIR" \
  --imgsz "$IMG_SIZE" \
  --device "$DEVICE"

echo "Gray adapter pipeline done."
echo "Adapter: ${OUT_DIR}/gray_adapter.pt"
echo "Adapter metadata: ${OUT_DIR}/gray_adapter.json"
echo "Full adapter ONNX: ${OUT_DIR}/gray_adapter_yolov8_full.onnx"
echo "Golden cases: ${GOLDEN_DIR}/golden_predictions.json"
