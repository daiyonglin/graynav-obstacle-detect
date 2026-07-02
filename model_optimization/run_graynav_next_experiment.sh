#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

COCO_PUB="${COCO_PUB:-/root/autodl-pub/COCO2017}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/datasets/graynav_obstacle8_v2}"
TRAIN_ZIP="${TRAIN_ZIP:-${COCO_PUB}/train2017.zip}"
ANN_ZIP="${ANN_ZIP:-${COCO_PUB}/annotations_trainval2017.zip}"
TRAIN_ANN="${TRAIN_ANN:-/root/autodl-tmp/datasets/coco/annotations/instances_train2017.json}"
VAL_IMAGES="${VAL_IMAGES:-/root/autodl-tmp/datasets/coco/val2017}"
VAL_ANN="${VAL_ANN:-/root/autodl-tmp/datasets/coco/annotations/instances_val2017.json}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"

IMG_SIZE="${IMG_SIZE:-384}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-0.85}"
WORKERS="${WORKERS:-8}"
TRAIN_MAX_IMAGES="${TRAIN_MAX_IMAGES:-30000}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-0}"
NAV_RATIO="${NAV_RATIO:-0.8}"
PROJECT="${PROJECT:-runs/detect}"
CONFIG="${CONFIG:-configs/graynav_next_training.yaml}"
EVAL_DIR="${EVAL_DIR:-artifacts/graynav_next_eval}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/graynav_next_export}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-runs/tensorboard/graynav_next}"

M1_PREFIX="${M1_PREFIX:-M1_ggg_yolov8n_ft}"
M2_PREFIX="${M2_PREFIX:-M2_gmfe_yolov8n_ft}"

mkdir -p "$(dirname "$TRAIN_ANN")"
if [[ ! -f "$TRAIN_ANN" ]]; then
  unzip -j "$ANN_ZIP" annotations/instances_train2017.json -d "$(dirname "$TRAIN_ANN")"
fi
if [[ ! -f "$VAL_ANN" ]]; then
  unzip -j "$ANN_ZIP" annotations/instances_val2017.json -d "$(dirname "$VAL_ANN")"
fi

if [[ "${PREPARE_DATA:-1}" == "1" ]]; then
  python scripts/prepare_graynav_next_dataset.py \
    --train-zip "$TRAIN_ZIP" \
    --train-annotations "$TRAIN_ANN" \
    --val-images "$VAL_IMAGES" \
    --val-annotations "$VAL_ANN" \
    --out "$DATASET_ROOT" \
    --train-max-images "$TRAIN_MAX_IMAGES" \
    --val-max-images "$VAL_MAX_IMAGES" \
    --nav-ratio "$NAV_RATIO" \
    ${OVERWRITE_DATA:+--overwrite}
fi

GGG_YAML="${DATASET_ROOT}/graynav-obstacle8-ggg.yaml"
GMFE_YAML="${DATASET_ROOT}/graynav-obstacle8-gmfe.yaml"

if [[ "${TRAIN_M1:-1}" == "1" ]]; then
  python scripts/train_yolov8n_gray_obstacle8.py \
    --data "$GGG_YAML" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project "$PROJECT" \
    --name-prefix "$M1_PREFIX" \
    --base-weights "$WEIGHTS" \
    ${CACHE_IMAGES:+--cache}
fi

if [[ "${TRAIN_M2:-1}" == "1" ]]; then
  python scripts/train_yolov8n_gray_obstacle8.py \
    --data "$GMFE_YAML" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project "$PROJECT" \
    --name-prefix "$M2_PREFIX" \
    --base-weights "$WEIGHTS" \
    ${CACHE_IMAGES:+--cache}
fi

M1_WEIGHTS="${PROJECT}/${M1_PREFIX}_stabilize/weights/best.pt"
M2_WEIGHTS="${PROJECT}/${M2_PREFIX}_stabilize/weights/best.pt"

python scripts/evaluate_graynav_next.py \
  --dataset-root "$DATASET_ROOT" \
  --m1-weights "$M1_WEIGHTS" \
  --m2-weights "$M2_WEIGHTS" \
  --out-dir "$EVAL_DIR" \
  --imgsz "$IMG_SIZE" \
  --device "$DEVICE" \
  --batch "${EVAL_BATCH:-64}" \
  --workers "$WORKERS" \
  --tensorboard-dir "$TENSORBOARD_DIR"

if [[ "${EXPORT_MODELS:-1}" == "1" ]]; then
  python scripts/export_yolov8_head6.py \
    --weights "$M1_WEIGHTS" \
    --out-dir "${ARTIFACT_ROOT}/M1_ggg" \
    --imgsz "$IMG_SIZE" \
    --num-classes 8

  python scripts/export_yolov8_head6.py \
    --weights "$M2_WEIGHTS" \
    --out-dir "${ARTIFACT_ROOT}/M2_gmfe" \
    --imgsz "$IMG_SIZE" \
    --num-classes 8

  python scripts/make_a1_calibration_dataset.py \
    --input-dir "${DATASET_ROOT}/variants/ggg/images/train" \
    --output-root "${ARTIFACT_ROOT}/M1_ggg/a1_calib" \
    --imgsz "$IMG_SIZE" \
    --input-mode color3 \
    --calib-num "${A1_CALIB_NUM:-80}" \
    --eval-num "${A1_EVAL_NUM:-20}"

  python scripts/make_a1_calibration_dataset.py \
    --input-dir "${DATASET_ROOT}/variants/gmfe/images/train" \
    --output-root "${ARTIFACT_ROOT}/M2_gmfe/a1_calib" \
    --imgsz "$IMG_SIZE" \
    --input-mode color3 \
    --calib-num "${A1_CALIB_NUM:-80}" \
    --eval-num "${A1_EVAL_NUM:-20}"
fi

cat > "${EVAL_DIR}/cloud_pack_command.txt" <<EOF
cd ${ROOT_DIR}
tar -czf /root/autodl-tmp/graynav_next_results.tar.gz \\
  ${PROJECT}/${M1_PREFIX}_stabilize \\
  ${PROJECT}/${M2_PREFIX}_stabilize \\
  ${EVAL_DIR} \\
  ${ARTIFACT_ROOT} \\
  ${TENSORBOARD_DIR} \\
  ${DATASET_ROOT}/dataset_manifest.json \\
  ${DATASET_ROOT}/gmfe_meta.json \\
  ${DATASET_ROOT}/audit
EOF

echo "GrayNav next experiment done."
echo "M1 weights: ${M1_WEIGHTS}"
echo "M2 weights: ${M2_WEIGHTS}"
echo "Eval summary: ${EVAL_DIR}/graynav_next_eval_summary.json"
echo "Pack command: ${EVAL_DIR}/cloud_pack_command.txt"
