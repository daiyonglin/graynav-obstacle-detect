#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAIN_ZIP="${TRAIN_ZIP:-/root/autodl-pub/COCO2017/train2017.zip}"
ANNOTATION_ZIP="${ANNOTATION_ZIP:-/root/autodl-pub/COCO2017/annotations_trainval2017.zip}"
VAL_ZIP="${VAL_ZIP:-/root/autodl-tmp/val2017.zip}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/graynav_coco8_graystem}"
COCO_ROOT="${COCO_ROOT:-/root/autodl-tmp/datasets/coco}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
M1_WEIGHTS_PATH="${M1_WEIGHTS_PATH:-}"
CONFIG="${CONFIG:-configs/graystem_training.yaml}"
IMG_SIZE="${IMG_SIZE:-384}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-96}"
WORKERS="${WORKERS:-8}"
TARGET_IMAGES="${TARGET_IMAGES:-60000}"
HARD_NEGATIVE="${HARD_NEGATIVE:-2500}"
RUN_M2="${RUN_M2:-1}"

ANN_DIR="$COCO_ROOT/annotations"
TRAIN_ANN="$ANN_DIR/instances_train2017.json"
VAL_ANN="$ANN_DIR/instances_val2017.json"
VAL_IMAGES="$COCO_ROOT/val2017"
STATE_DIR="artifacts/graystem_state"
EXPORT_DIR="artifacts/graystem_export"
EVAL_DIR="artifacts/graystem_eval"
TB_DIR="runs/tensorboard/graystem"

mkdir -p "$ANN_DIR" "$STATE_DIR" "$EXPORT_DIR" "$EVAL_DIR"

if [[ ! -f "$TRAIN_ANN" || ! -f "$VAL_ANN" ]]; then
  unzip -o -j "$ANNOTATION_ZIP" annotations/instances_train2017.json annotations/instances_val2017.json -d "$ANN_DIR"
fi

if [[ ! -d "$VAL_IMAGES" || -z "$(find "$VAL_IMAGES" -maxdepth 1 -name '*.jpg' -print -quit 2>/dev/null)" ]]; then
  mkdir -p "$COCO_ROOT"
  unzip -q "$VAL_ZIP" -d "$COCO_ROOT"
fi

if [[ ! -f "$DATA_ROOT/graynav8.yaml" ]]; then
  python scripts/prepare_graystem_dataset.py \
    --train-zip "$TRAIN_ZIP" \
    --train-annotations "$TRAIN_ANN" \
    --val-images "$VAL_IMAGES" \
    --val-annotations "$VAL_ANN" \
    --out "$DATA_ROOT" \
    --target-images "$TARGET_IMAGES" \
    --hard-negative "$HARD_NEGATIVE" \
    --overwrite
fi

if [[ -n "$M1_WEIGHTS_PATH" ]]; then
  M1_WEIGHTS="$M1_WEIGHTS_PATH"
  echo "$M1_WEIGHTS" > "$STATE_DIR/M1_final_weights.txt"
  echo "Using existing M1 weights: $M1_WEIGHTS"
else
  python scripts/train_yolov8n_gray_obstacle8.py \
    --data "$DATA_ROOT/graynav8.yaml" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project runs/detect \
    --name-prefix M1_graynav_yolov8n_ft \
    --base-weights "$WEIGHTS" \
    --final-weights-file "$STATE_DIR/M1_final_weights.txt"
  M1_WEIGHTS="$(cat "$STATE_DIR/M1_final_weights.txt")"
fi
python scripts/export_yolov8_head6.py \
  --weights "$M1_WEIGHTS" \
  --out-dir "$EXPORT_DIR/M1_graynav_yolov8n_ft" \
  --imgsz "$IMG_SIZE" \
  --num-classes 8

M2_WEIGHTS=""
if [[ "$RUN_M2" == "1" ]]; then
  M2_INIT="$STATE_DIR/M2_graystem_bc_init.pt"
  python scripts/graystem_yolov8.py --mode init-from-m1 --weights "$M1_WEIGHTS" --out "$M2_INIT"
  python scripts/train_yolov8n_gray_obstacle8.py \
    --data "$DATA_ROOT/graynav8.yaml" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project runs/detect \
    --name-prefix M2_graystem_bc \
    --base-weights "$M2_INIT" \
    --stages adapt stabilize \
    --final-weights-file "$STATE_DIR/M2_raw_final_weights.txt"
  M2_RAW="$(cat "$STATE_DIR/M2_raw_final_weights.txt")"
  M2_WEIGHTS="$STATE_DIR/M2_graystem_bc_final.pt"
  python scripts/graystem_yolov8.py --mode tie-bc --weights "$M2_RAW" --out "$M2_WEIGHTS"
  echo "$M2_WEIGHTS" > "$STATE_DIR/M2_final_weights.txt"
  python scripts/export_yolov8_head6.py \
    --weights "$M2_WEIGHTS" \
    --out-dir "$EXPORT_DIR/M2_graystem_bc" \
    --imgsz "$IMG_SIZE" \
    --num-classes 8
fi

EVAL_ARGS=(
  --dataset-root "$DATA_ROOT"
  --annotations "$DATA_ROOT/annotations/instances_val_obstacle8.json"
  --m0-weights "$WEIGHTS"
  --m1-weights "$M1_WEIGHTS"
  --out-dir "$EVAL_DIR"
  --imgsz "$IMG_SIZE"
  --device "$DEVICE"
  --tensorboard-dir "$TB_DIR"
)
if [[ -n "$M2_WEIGHTS" ]]; then
  EVAL_ARGS+=(--m2-weights "$M2_WEIGHTS")
fi
python scripts/evaluate_graystem_obstacle8.py "${EVAL_ARGS[@]}"

cat > "$STATE_DIR/cloud_pack_command.txt" <<'EOF'
cd /root/autodl-tmp/graynav-graystem/model_optimization
tar -czf /root/autodl-tmp/graystem_results_min.tar.gz \
  artifacts/graystem_state \
  artifacts/graystem_eval/graystem_eval_summary.json \
  /root/autodl-tmp/datasets/graynav_coco8_graystem/dataset_manifest.json
EOF

echo "GrayStem experiment done."
echo "M1 weights: $M1_WEIGHTS"
if [[ -n "$M2_WEIGHTS" ]]; then echo "M2 weights: $M2_WEIGHTS"; fi
echo "Eval summary: $EVAL_DIR/graystem_eval_summary.json"
