#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAIN_ZIP="${TRAIN_ZIP:-/root/autodl-pub/COCO2017/train2017.zip}"
ANNOTATION_ZIP="${ANNOTATION_ZIP:-/root/autodl-pub/COCO2017/annotations_trainval2017.zip}"
VAL_ZIP="${VAL_ZIP:-/root/autodl-tmp/val2017.zip}"
COCO_ROOT="${COCO_ROOT:-/root/autodl-tmp/datasets/coco}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/gray_coco80_graystem_bc}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
M1_CONFIG="${M1_CONFIG:-configs/yolov8n80_gray_conservative.yaml}"
M2_CONFIG="${M2_CONFIG:-configs/graystem_bc_yolov8n80.yaml}"
IMG_SIZE="${IMG_SIZE:-384}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-0.85}"
WORKERS="${WORKERS:-12}"
EVAL_BATCH="${EVAL_BATCH:-128}"
CACHE="${CACHE:-ram}"
TARGET_IMAGES="${TARGET_IMAGES:-30000}"
CONTEXT_IMAGES="${CONTEXT_IMAGES:-8000}"
NEGATIVE_IMAGES="${NEGATIVE_IMAGES:-0}"
RUN_M1="${RUN_M1:-1}"
RUN_M2="${RUN_M2:-1}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
TRAIN_VERBOSE="${TRAIN_VERBOSE:-1}"
TERMINAL_QUIET="${TERMINAL_QUIET:-0}"
LOG_DIR="${LOG_DIR:-logs/graystem_bc_$(date +%Y%m%d_%H%M%S)}"

ANN_DIR="$COCO_ROOT/annotations"
TRAIN_ANN="$ANN_DIR/instances_train2017.json"
VAL_ANN="$ANN_DIR/instances_val2017.json"
VAL_IMAGES="$COCO_ROOT/val2017"
STATE_DIR="artifacts/graystem_bc_state"
EXPORT_DIR="artifacts/graystem_bc_export"
EVAL_DIR="artifacts/graystem_bc_eval"
TB_DIR="runs/tensorboard/graystem_bc"

mkdir -p "$ANN_DIR" "$STATE_DIR" "$EXPORT_DIR" "$EVAL_DIR" "$LOG_DIR"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-2}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: base weights not found: $WEIGHTS" >&2
  exit 2
fi
if [[ "$(basename "$WEIGHTS")" != "yolov8n.pt" ]]; then
  echo "ERROR: this experiment must start from official yolov8n.pt, got: $WEIGHTS" >&2
  exit 2
fi

python - <<PY
from pathlib import Path
import hashlib, json
p=Path("$WEIGHTS")
info={"base_weights": str(p), "name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
Path("$STATE_DIR").mkdir(parents=True, exist_ok=True)
Path("$STATE_DIR/base_weights_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print("base_weights_info:", json.dumps(info, indent=2))
PY

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "========== ${name} =========="
  echo "log: ${log}"
  if [[ "$TERMINAL_QUIET" == "1" ]]; then
    "$@" >"$log" 2>&1
  else
    "$@" 2>&1 | tee "$log"
  fi
}

if [[ ! -f "$TRAIN_ANN" || ! -f "$VAL_ANN" ]]; then
  unzip -o -j "$ANNOTATION_ZIP" annotations/instances_train2017.json annotations/instances_val2017.json -d "$ANN_DIR"
fi
if [[ ! -d "$VAL_IMAGES" || -z "$(find "$VAL_IMAGES" -maxdepth 1 -name '*.jpg' -print -quit 2>/dev/null)" ]]; then
  mkdir -p "$COCO_ROOT"
  unzip -q "$VAL_ZIP" -d "$COCO_ROOT"
fi

if [[ "$REBUILD_DATASET" == "1" ]]; then
  echo "REBUILD_DATASET=1, removing existing dataset root: $DATA_ROOT"
  rm -rf "$DATA_ROOT"
fi

if [[ ! -f "$DATA_ROOT/gray_coco80.yaml" ]]; then
  run_step prepare_dataset python scripts/prepare_yolov8n80_gray_dataset.py \
    --train-zip "$TRAIN_ZIP" \
    --train-annotations "$TRAIN_ANN" \
    --val-images "$VAL_IMAGES" \
    --val-annotations "$VAL_ANN" \
    --out "$DATA_ROOT" \
    --target-images "$TARGET_IMAGES" \
    --context-images "$CONTEXT_IMAGES" \
    --negative-images "$NEGATIVE_IMAGES" \
    --overwrite
else
  echo "Reusing existing dataset: $DATA_ROOT"
  if [[ -f "$DATA_ROOT/dataset_manifest.json" ]]; then
    python - <<PY
from pathlib import Path
import json
p=Path("$DATA_ROOT/dataset_manifest.json")
m=json.loads(p.read_text(encoding="utf-8"))
print("dataset_manifest:", json.dumps({
    "dataset": m.get("dataset"),
    "target_images": m.get("target_images"),
    "context_images": m.get("context_images"),
    "negative_images": m.get("negative_images"),
    "splits": m.get("splits"),
}, ensure_ascii=False, indent=2))
PY
  fi
fi

run_step audit_gray_dataset python scripts/audit_gray_dataset.py \
  --roots "$DATA_ROOT/images/train" "$DATA_ROOT/images/val" \
  --out "$STATE_DIR/gray_dataset_audit.json" \
  --sample-images 1200

VERBOSE_ARGS=()
if [[ "$TRAIN_VERBOSE" == "1" ]]; then
  VERBOSE_ARGS+=(--verbose)
fi

if [[ "$RUN_M1" == "1" || ! -f "$STATE_DIR/M1_final_weights.txt" ]]; then
  run_step train_m1_conservative python scripts/train_yolov8n_gray_obstacle8.py \
    --data "$DATA_ROOT/gray_coco80.yaml" \
    --config "$M1_CONFIG" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project runs/detect \
    --name-prefix M1_yolov8n80_gray_conservative \
    --base-weights "$WEIGHTS" \
    --cache "$CACHE" \
    --final-weights-file "$STATE_DIR/M1_final_weights.txt" \
    "${VERBOSE_ARGS[@]}"
fi

M1_WEIGHTS="$(cat "$STATE_DIR/M1_final_weights.txt")"

if [[ "$RUN_M2" == "1" ]]; then
  run_step train_m2_graystem_bc python scripts/train_yolov8n_graystem_bc.py \
    --data "$DATA_ROOT/gray_coco80.yaml" \
    --config "$M2_CONFIG" \
    --base-weights "$M1_WEIGHTS" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project runs/detect \
    --name-prefix M2_graystem_bc_yolov8n80 \
    --cache "$CACHE" \
    --final-weights-file "$STATE_DIR/M2_final_weights.txt" \
    "${VERBOSE_ARGS[@]}"
fi

M2_WEIGHTS=""
if [[ -f "$STATE_DIR/M2_final_weights.txt" ]]; then
  M2_WEIGHTS="$(cat "$STATE_DIR/M2_final_weights.txt")"
fi

run_step export_m1 python scripts/export_yolov8_head6.py \
  --weights "$M1_WEIGHTS" \
  --out-dir "$EXPORT_DIR/M1_yolov8n80_gray_conservative" \
  --imgsz "$IMG_SIZE" \
  --num-classes 80

if [[ -n "$M2_WEIGHTS" ]]; then
  run_step export_m2 python scripts/export_yolov8_head6.py \
    --weights "$M2_WEIGHTS" \
    --out-dir "$EXPORT_DIR/M2_graystem_bc_yolov8n80" \
    --imgsz "$IMG_SIZE" \
    --num-classes 80
fi

EVAL_ARGS=(
  --dataset-root "$DATA_ROOT"
  --annotations "$DATA_ROOT/annotations/instances_val_graynav8.json"
  --m0-weights "$WEIGHTS"
  --m1-weights "$M1_WEIGHTS"
  --out-dir "$EVAL_DIR"
  --imgsz "$IMG_SIZE"
  --device "$DEVICE"
  --batch "$EVAL_BATCH"
  --tensorboard-dir "$TB_DIR"
)
if [[ -n "$M2_WEIGHTS" ]]; then
  EVAL_ARGS+=(--m2-weights "$M2_WEIGHTS")
fi
run_step evaluate python scripts/evaluate_graystem_obstacle8.py "${EVAL_ARGS[@]}"

cat > "$STATE_DIR/cloud_pack_command.txt" <<EOF
cd /root/autodl-tmp/graynav-graystem-bc/model_optimization
tar -czf /root/autodl-tmp/graystem_bc_next_min_results.tar.gz \\
  artifacts/graystem_bc_state \\
  artifacts/graystem_bc_eval/graystem_eval_summary.json \\
  /root/autodl-tmp/datasets/gray_coco80_graystem_bc/dataset_manifest.json \\
  logs
EOF

echo "GrayStem-BC next experiment done."
echo "M1 weights: $M1_WEIGHTS"
echo "M2 weights: ${M2_WEIGHTS:-none}"
echo "Eval summary: $EVAL_DIR/graystem_eval_summary.json"
echo "Logs: $LOG_DIR"
