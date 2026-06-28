#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

WEIGHTS="${1:-runs/detect/obstacle8_gray_yolov8n_stabilize/weights/best.pt}"
OUT_DIR="${2:-artifacts/obstacle8_yolov8n}"

python scripts/export_yolov8_head6.py \
  --weights "$WEIGHTS" \
  --out-dir "$OUT_DIR" \
  --num-classes 8

python scripts/fold_yolov8_first_conv_to_gray_onnx.py \
  --input "${OUT_DIR}/obstacle8_yolov8n.onnx" \
  --output "${OUT_DIR}/obstacle8_yolov8n_1ch_replicate_exact.onnx" \
  --mode replicate_exact

