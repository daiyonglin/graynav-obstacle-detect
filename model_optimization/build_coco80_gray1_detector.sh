#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${GRAYNAV_ROOT:-${SCRIPT_DIR}}"
OUT="${GRAYNAV_DETECTOR_OUT:-/root/autodl-tmp/graynav_coco80_gray1}"
WEIGHTS="${YOLOV8N_WEIGHTS:-${ROOT}/yolov8n.pt}"

mkdir -p "${OUT}"
python "${ROOT}/scripts/export_yolov8_head6.py" \
  --weights "${WEIGHTS}" --out-dir "${OUT}" --imgsz 384 --opset 12 \
  --num-classes 80 --prefix graynav_yolov8n80_rgb
python "${ROOT}/scripts/fold_yolov8_first_conv_to_gray_onnx.py" \
  --input "${OUT}/graynav_yolov8n80_rgb.onnx" \
  --output "${OUT}/graynav_yolov8n80_gray1.onnx" --mode replicate_exact
python "${ROOT}/scripts/fold_yolov8_first_conv_to_gray_onnx.py" \
  --input "${OUT}/graynav_yolov8n80_rgb_head6.onnx" \
  --output "${OUT}/graynav_yolov8n80_gray1_head6.onnx" --mode replicate_exact
python "${ROOT}/scripts/audit_onnx_a1_ops.py" \
  --onnx "${OUT}/graynav_yolov8n80_gray1_head6.onnx" \
  --out "${OUT}/a1_ops.json" --expect-input-channels 1 --expect-num-classes 80
sha256sum "${OUT}/graynav_yolov8n80_gray1.onnx" \
  "${OUT}/graynav_yolov8n80_gray1_head6.onnx" > "${OUT}/SHA256SUMS.txt"

PREPARED="${GRAYNAV_PREPARED:-/root/autodl-tmp/graynav_surface_depth_prepared}"
if [[ -f "${PREPARED}/manifest_val.jsonl" ]]; then
  python "${ROOT}/scripts/build_a1_surface_depth_datasets.py" \
    --data "${PREPARED}" --output "${OUT}/deploy" \
    --size 384 --calibrate 160 --evaluate 40
  sha256sum "${OUT}/deploy/datasets.zip" >> "${OUT}/SHA256SUMS.txt"
else
  echo "Prepared public validation data not found; re-run after dataset preparation to create 384x384 datasets.zip."
fi
