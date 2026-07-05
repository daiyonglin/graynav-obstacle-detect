#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAIN_ZIP="${TRAIN_ZIP:-/root/autodl-pub/COCO2017/train2017.zip}"
ANNOTATION_ZIP="${ANNOTATION_ZIP:-/root/autodl-pub/COCO2017/annotations_trainval2017.zip}"
VAL_ZIP="${VAL_ZIP:-/root/autodl-tmp/val2017.zip}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/gray_coco80_yolov8n_ft}"
COCO_ROOT="${COCO_ROOT:-/root/autodl-tmp/datasets/coco}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
CONFIG="${CONFIG:-configs/yolov8n80_gray_ft.yaml}"
IMG_SIZE="${IMG_SIZE:-384}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-0.85}"
WORKERS="${WORKERS:-12}"
EVAL_BATCH="${EVAL_BATCH:-128}"
CACHE="${CACHE:-ram}"
TARGET_IMAGES="${TARGET_IMAGES:-30000}"
CONTEXT_IMAGES="${CONTEXT_IMAGES:-3000}"
TERMINAL_QUIET="${TERMINAL_QUIET:-1}"
LOG_DIR="${LOG_DIR:-logs/yolov8n80_gray_$(date +%Y%m%d_%H%M%S)}"

ANN_DIR="$COCO_ROOT/annotations"
TRAIN_ANN="$ANN_DIR/instances_train2017.json"
VAL_ANN="$ANN_DIR/instances_val2017.json"
VAL_IMAGES="$COCO_ROOT/val2017"
STATE_DIR="artifacts/yolov8n80_gray_state"
EXPORT_DIR="artifacts/yolov8n80_gray_export"
EVAL_DIR="artifacts/yolov8n80_gray_eval"
TB_DIR="runs/tensorboard/yolov8n80_gray"

mkdir -p "$ANN_DIR" "$STATE_DIR" "$EXPORT_DIR" "$EVAL_DIR" "$LOG_DIR"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-5}"

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

if [[ ! -f "$DATA_ROOT/gray_coco80.yaml" ]]; then
  run_step prepare_dataset python scripts/prepare_yolov8n80_gray_dataset.py \
    --train-zip "$TRAIN_ZIP" \
    --train-annotations "$TRAIN_ANN" \
    --val-images "$VAL_IMAGES" \
    --val-annotations "$VAL_ANN" \
    --out "$DATA_ROOT" \
    --target-images "$TARGET_IMAGES" \
    --context-images "$CONTEXT_IMAGES" \
    --overwrite
fi

run_step train_m1_v2 python scripts/train_yolov8n_gray_obstacle8.py \
  --data "$DATA_ROOT/gray_coco80.yaml" \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --project runs/detect \
  --name-prefix M1_v2_yolov8n80_gray_ft \
  --base-weights "$WEIGHTS" \
  --cache "$CACHE" \
  --final-weights-file "$STATE_DIR/M1_v2_final_weights.txt"

M1_WEIGHTS="$(cat "$STATE_DIR/M1_v2_final_weights.txt")"
run_step export_m1_v2 python scripts/export_yolov8_head6.py \
  --weights "$M1_WEIGHTS" \
  --out-dir "$EXPORT_DIR/M1_v2_yolov8n80_gray_ft" \
  --imgsz "$IMG_SIZE" \
  --num-classes 80

run_step evaluate python scripts/evaluate_graystem_obstacle8.py \
  --dataset-root "$DATA_ROOT" \
  --annotations "$DATA_ROOT/annotations/instances_val_graynav8.json" \
  --m0-weights "$WEIGHTS" \
  --m1-weights "$M1_WEIGHTS" \
  --out-dir "$EVAL_DIR" \
  --imgsz "$IMG_SIZE" \
  --device "$DEVICE" \
  --batch "$EVAL_BATCH" \
  --tensorboard-dir "$TB_DIR"

cat > "$STATE_DIR/cloud_pack_command.txt" <<'EOF'
cd /root/autodl-tmp/graynav-graystem/model_optimization
tar -czf /root/autodl-tmp/yolov8n80_gray_m1v2_min_results.tar.gz \
  artifacts/yolov8n80_gray_state \
  artifacts/yolov8n80_gray_eval/graystem_eval_summary.json \
  /root/autodl-tmp/datasets/gray_coco80_yolov8n_ft/dataset_manifest.json \
  logs
EOF

echo "YOLOv8n80 gray fine-tuning done."
echo "M1-v2 weights: $M1_WEIGHTS"
echo "Eval summary: $EVAL_DIR/graystem_eval_summary.json"
echo "Logs: $LOG_DIR"

