#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COCO_ROOT="${GRAYNAV_COCO_ROOT:-/root/autodl-tmp/coco2017}"
COCO_PREPARED="${GRAYNAV_COCO_PREPARED:-/root/autodl-tmp/graynav_coco_indoor8}"
SCENE_PREPARED="${GRAYNAV_SCENE_PREPARED:-/root/autodl-tmp/graynav_surface_depth_prepared_v2}"
RUN_ROOT="${GRAYNAV_UNIFIED_RUN_ROOT:-/root/autodl-tmp/graynav_unified_indoor8_run}"
YOLO_WEIGHTS="${GRAYNAV_YOLO_WEIGHTS:-${ROOT_DIR}/../weights/yolov8n.pt}"
E3_CHECKPOINT="${GRAYNAV_E3_CHECKPOINT:-${ROOT_DIR}/../weights/graynav_surface_depth_e3_epoch49.pt}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "${COCO_PREPARED}" "${RUN_ROOT}/train" "${RUN_ROOT}/tensorboard"

if [[ ! -f "${COCO_PREPARED}/manifest_train.jsonl" ]]; then
  python "${ROOT_DIR}/scripts/prepare_coco_indoor8.py" \
    --annotations "${COCO_ROOT}/annotations/instances_train2017.json" \
    --images "${COCO_ROOT}/train2017" \
    --output "${COCO_PREPARED}" \
    --split train
fi
if [[ ! -f "${COCO_PREPARED}/manifest_val.jsonl" ]]; then
  python "${ROOT_DIR}/scripts/prepare_coco_indoor8.py" \
    --annotations "${COCO_ROOT}/annotations/instances_val2017.json" \
    --images "${COCO_ROOT}/val2017" \
    --output "${COCO_PREPARED}" \
    --split val
fi

test -f "${YOLO_WEIGHTS}"
test -f "${E3_CHECKPOINT}"
test -f "${SCENE_PREPARED}/manifest_train.jsonl"

  python "${ROOT_DIR}/scripts/train_unified.py" \
  --coco "${COCO_PREPARED}" \
  --scene "${SCENE_PREPARED}" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --surface-e3 "${E3_CHECKPOINT}" \
  --output "${RUN_ROOT}/train" \
  --log-dir "${RUN_ROOT}/tensorboard" \
  --epochs 40 \
  --scene-warmup-epochs 5 \
  --steps-per-epoch 1000 \
  --batch-size 32 \
  --workers 8 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --seed 42 \
  --amp

echo "GRAYNAV_UNIFIED_CLOUD_RUN_OK"
