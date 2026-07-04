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
ALLOW_NON_YOLOV8N="${ALLOW_NON_YOLOV8N:-0}"
CONFIG="${CONFIG:-configs/graystem_training.yaml}"
IMG_SIZE="${IMG_SIZE:-384}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-0.85}"
WORKERS="${WORKERS:-12}"
EVAL_BATCH="${EVAL_BATCH:-64}"
CACHE="${CACHE:-ram}"
TARGET_IMAGES="${TARGET_IMAGES:-60000}"
HARD_NEGATIVE="${HARD_NEGATIVE:-2500}"
RUN_M2="${RUN_M2:-1}"
TERMINAL_QUIET="${TERMINAL_QUIET:-0}"
LOG_DIR="${LOG_DIR:-logs/graystem_$(date +%Y%m%d_%H%M%S)}"

ANN_DIR="$COCO_ROOT/annotations"
TRAIN_ANN="$ANN_DIR/instances_train2017.json"
VAL_ANN="$ANN_DIR/instances_val2017.json"
VAL_IMAGES="$COCO_ROOT/val2017"
STATE_DIR="artifacts/graystem_state"
EXPORT_DIR="artifacts/graystem_export"
EVAL_DIR="artifacts/graystem_eval"
TB_DIR="runs/tensorboard/graystem"

mkdir -p "$ANN_DIR" "$STATE_DIR" "$EXPORT_DIR" "$EVAL_DIR"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-5}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: base weights not found: $WEIGHTS" >&2
  echo "This runner refuses Ultralytics auto-download. Put official yolov8n.pt at /root/autodl-tmp/yolov8n.pt or set WEIGHTS explicitly." >&2
  exit 2
fi

if [[ "$(basename "$WEIGHTS")" != "yolov8n.pt" && "$ALLOW_NON_YOLOV8N" != "1" ]]; then
  echo "ERROR: expected official YOLOv8n weights, got: $WEIGHTS" >&2
  echo "Set ALLOW_NON_YOLOV8N=1 only for an intentional non-YOLOv8n ablation." >&2
  exit 2
fi

python - <<PY
from pathlib import Path
import hashlib, json
p = Path("$WEIGHTS")
info = {
    "base_weights": str(p),
    "name": p.name,
    "bytes": p.stat().st_size,
    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
}
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

if [[ ! -f "$DATA_ROOT/graynav8.yaml" ]]; then
  run_step prepare_dataset python scripts/prepare_graystem_dataset.py \
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
  run_step train_m1 python scripts/train_yolov8n_gray_obstacle8.py \
    --data "$DATA_ROOT/graynav8.yaml" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project runs/detect \
    --name-prefix M1_graynav_yolov8n_ft \
    --base-weights "$WEIGHTS" \
    --cache "$CACHE" \
    --final-weights-file "$STATE_DIR/M1_final_weights.txt"
  M1_WEIGHTS="$(cat "$STATE_DIR/M1_final_weights.txt")"
fi
run_step export_m1 python scripts/export_yolov8_head6.py \
  --weights "$M1_WEIGHTS" \
  --out-dir "$EXPORT_DIR/M1_graynav_yolov8n_ft" \
  --imgsz "$IMG_SIZE" \
  --num-classes 8

M2_WEIGHTS=""
if [[ "$RUN_M2" == "1" ]]; then
  M2_INIT="$STATE_DIR/M2_graystem_bc_init.pt"
  run_step init_m2 python scripts/graystem_yolov8.py --mode init-from-m1 --weights "$M1_WEIGHTS" --out "$M2_INIT"
  run_step train_m2 python scripts/train_yolov8n_gray_obstacle8.py \
    --data "$DATA_ROOT/graynav8.yaml" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --project runs/detect \
    --name-prefix M2_graystem_bc \
    --base-weights "$M2_INIT" \
    --stages adapt stabilize \
    --cache "$CACHE" \
    --final-weights-file "$STATE_DIR/M2_raw_final_weights.txt"
  M2_RAW="$(cat "$STATE_DIR/M2_raw_final_weights.txt")"
  M2_WEIGHTS="$STATE_DIR/M2_graystem_bc_final.pt"
  run_step tie_m2_bc python scripts/graystem_yolov8.py --mode tie-bc --weights "$M2_RAW" --out "$M2_WEIGHTS"
  echo "$M2_WEIGHTS" > "$STATE_DIR/M2_final_weights.txt"
  run_step export_m2 python scripts/export_yolov8_head6.py \
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
  --batch "$EVAL_BATCH"
  --tensorboard-dir "$TB_DIR"
)
if [[ -n "$M2_WEIGHTS" ]]; then
  EVAL_ARGS+=(--m2-weights "$M2_WEIGHTS")
fi
run_step evaluate python scripts/evaluate_graystem_obstacle8.py "${EVAL_ARGS[@]}"

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
echo "Logs: $LOG_DIR"
