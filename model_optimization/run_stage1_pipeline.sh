#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MANIFEST="${MANIFEST:-configs/dataset_manifest.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/data/obstacle8_gray}"
DATA_YAML="${DATA_YAML:-${DATASET_ROOT}/obstacle8-gray.yaml}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-12}"
PROJECT="${PROJECT:-runs/detect}"
NAME_PREFIX="${NAME_PREFIX:-obstacle8_gray_yolov8n}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/obstacle8_yolov8n}"
A1_CALIB_ROOT="${A1_CALIB_ROOT:-artifacts/a1_obstacle8_calib}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  echo "Copy configs/dataset_manifest.example.yaml to configs/dataset_manifest.yaml and edit dataset paths." >&2
  exit 1
fi

python scripts/prepare_dataset_from_manifest.py --manifest "$MANIFEST"

python scripts/augment_gray_yolo_dataset.py \
  --dataset "$DATASET_ROOT" \
  --split train \
  --copies "${AUG_COPIES:-2}"

python scripts/train_yolov8n_gray_obstacle8.py \
  --data "$DATA_YAML" \
  --config configs/training_stages.yaml \
  --device "$DEVICE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --project "$PROJECT" \
  --name-prefix "$NAME_PREFIX" \
  ${CACHE_IMAGES:+--cache}

FINAL_WEIGHTS="${PROJECT}/${NAME_PREFIX}_stabilize/weights/best.pt"

python scripts/evaluate_obstacle8.py \
  --weights "$FINAL_WEIGHTS" \
  --data "$DATA_YAML" \
  --device "$DEVICE" \
  --predict-source "${DATASET_ROOT}/images/val"

python scripts/export_yolov8_head6.py \
  --weights "$FINAL_WEIGHTS" \
  --out-dir "$ARTIFACT_ROOT" \
  --num-classes 8

python scripts/make_a1_calibration_dataset.py \
  --input-dir "${DATASET_ROOT}/images/train" \
  --output-root "$A1_CALIB_ROOT" \
  --calib-num "${A1_CALIB_NUM:-80}" \
  --eval-num "${A1_EVAL_NUM:-20}"

echo "Stage 1 done."
echo "Final weights: $FINAL_WEIGHTS"
echo "Head6 ONNX: ${ARTIFACT_ROOT}/obstacle8_yolov8n_head6.onnx"
echo "A1 datasets: ${A1_CALIB_ROOT}/datasets.zip"

