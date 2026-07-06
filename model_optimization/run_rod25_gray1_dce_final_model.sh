#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET_SOURCE="${DATASET_SOURCE:-/root/autodl-tmp/archive.zip}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/graynav_rod25_gray1}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
FINAL_WEIGHTS="${FINAL_WEIGHTS:-}"
IMG_SIZE="${IMG_SIZE:-384}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-12}"
CACHE="${CACHE:-ram}"
VAL_BATCH="${VAL_BATCH:-64}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
TRAIN_VERBOSE="${TRAIN_VERBOSE:-1}"
LR0="${LR0:-0.0003}"
LRF="${LRF:-0.01}"
MOSAIC="${MOSAIC:-0.25}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-15}"
CALIB_NUM="${CALIB_NUM:-120}"
EVAL_NUM="${EVAL_NUM:-40}"
LOG_DIR="${LOG_DIR:-logs/rod25_gray1_dce_final_$(date +%Y%m%d_%H%M%S)}"

STATE_DIR="artifacts/rod25_gray1_dce_final_state"
EXPORT_DIR="artifacts/rod25_gray1_dce_final_export"
CALIB_DIR="artifacts/rod25_gray1_dce_final_calib"

mkdir -p "$STATE_DIR" "$EXPORT_DIR" "$CALIB_DIR" "$LOG_DIR"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-2}"

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "========== ${name} =========="
  echo "log: ${log}"
  "$@" 2>&1 | tee "$log"
}

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: yolov8n base weights not found: $WEIGHTS" >&2
  exit 2
fi

if [[ "$REBUILD_DATASET" == "1" ]]; then
  echo "REBUILD_DATASET=1, removing $DATA_ROOT"
  rm -rf "$DATA_ROOT"
fi

if [[ ! -f "$DATA_ROOT/gray_dataset.yaml" ]]; then
  if [[ ! -e "$DATASET_SOURCE" ]]; then
    echo "ERROR: dataset source not found: $DATASET_SOURCE" >&2
    exit 2
  fi
  run_step prepare_rod25_gray1 python scripts/prepare_generic_yolo_gray_dataset.py \
    --source "$DATASET_SOURCE" \
    --out "$DATA_ROOT" \
    --output-mode gray1 \
    --overwrite
else
  echo "Reusing prepared true one-channel gray dataset: $DATA_ROOT"
fi

NC="$(python - <<PY
import yaml
from pathlib import Path
d=yaml.safe_load(Path("$DATA_ROOT/gray_dataset.yaml").read_text(encoding="utf-8"))
names=d["names"]
print(len(names) if isinstance(names, list) else len(names.keys()))
PY
)"
if [[ "$NC" != "25" ]]; then
  echo "ERROR: expected ROD25 25 classes, got $NC" >&2
  exit 2
fi

run_step audit_gray1_dataset python scripts/audit_generic_yolo_dataset.py \
  --dataset-root "$DATA_ROOT" \
  --data-yaml "$DATA_ROOT/gray_dataset.yaml" \
  --out "$STATE_DIR/dataset_integrity_audit.json" \
  --gray-sample 3000 \
  --phash-sample 0

VERBOSE_ARGS=()
if [[ "$TRAIN_VERBOSE" == "1" ]]; then
  VERBOSE_ARGS+=(--verbose)
fi

if [[ -n "$FINAL_WEIGHTS" ]]; then
  echo "$FINAL_WEIGHTS" > "$STATE_DIR/final_gray1_dce_weights.txt"
elif [[ "$RUN_TRAIN" == "1" || ! -f "$STATE_DIR/final_gray1_dce_weights.txt" ]]; then
  run_step train_gray1_dce python scripts/train_graynav_dce_yolov8n.py \
    --model-yaml configs/graynav_dce_yolov8n_1ch.yaml \
    --data "$DATA_ROOT/gray_dataset.yaml" \
    --weights "$WEIGHTS" \
    --out-dir "$STATE_DIR/final_gray1_dce" \
    --project runs/detect \
    --name final_gray1_dce_yolov8n_rod25_ft \
    --imgsz "$IMG_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --cache "$CACHE" \
    --lr0 "$LR0" \
    --lrf "$LRF" \
    --mosaic "$MOSAIC" \
    --close-mosaic "$CLOSE_MOSAIC" \
    --final-weights-file "$STATE_DIR/final_gray1_dce_weights.txt" \
    "${VERBOSE_ARGS[@]}"
else
  echo "Reusing final gray1 DCE weights: $(cat "$STATE_DIR/final_gray1_dce_weights.txt")"
fi

FINAL_GRAY1_WEIGHTS="$(cat "$STATE_DIR/final_gray1_dce_weights.txt")"
if [[ ! -f "$FINAL_GRAY1_WEIGHTS" ]]; then
  echo "ERROR: final gray1 DCE weights not found: $FINAL_GRAY1_WEIGHTS" >&2
  exit 2
fi

run_step validate_gray1_test python scripts/validate_yolo_model.py \
  --weights "$FINAL_GRAY1_WEIGHTS" \
  --data "$DATA_ROOT/gray_dataset.yaml" \
  --out "$STATE_DIR/test_metrics.json" \
  --split test \
  --imgsz "$IMG_SIZE" \
  --batch "$VAL_BATCH" \
  --device "$DEVICE" \
  --project runs/val \
  --name final_gray1_dce_test

run_step export_gray1_dce_head6 python scripts/export_graynav_dce_head6.py \
  --weights "$FINAL_GRAY1_WEIGHTS" \
  --out-dir "$EXPORT_DIR" \
  --imgsz "$IMG_SIZE" \
  --num-classes "$NC"

run_step audit_head6_onnx_a1_ops python scripts/audit_onnx_a1_ops.py \
  --onnx "$EXPORT_DIR/graynav_dce_yolov8n_head6.onnx" \
  --out "$EXPORT_DIR/a1_ops_audit_head6.json" \
  --expect-input-channels 1 \
  --expect-num-classes "$NC"

run_step make_gray1_calibration_dataset python scripts/make_a1_calibration_dataset.py \
  --input-dir "$DATA_ROOT/images/train" \
  --output-root "$CALIB_DIR" \
  --imgsz "$IMG_SIZE" \
  --calib-num "$CALIB_NUM" \
  --eval-num "$EVAL_NUM" \
  --input-mode gray1

python - <<PY
import json, yaml, hashlib
from pathlib import Path
data_yaml = Path("$DATA_ROOT/gray_dataset.yaml")
data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
names = data["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
weights = Path("$FINAL_GRAY1_WEIGHTS")
meta = {
    "model_name": "GrayNav-DCE-YOLOv8n-ROD25-Gray1",
    "base_weights": "$WEIGHTS",
    "final_weights": str(weights),
    "final_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    "input_mode": "true single-channel grayscale input",
    "input_shape": [1, 1, $IMG_SIZE, $IMG_SIZE],
    "first_conv_init": "RGB first-conv weights from yolov8n.pt are summed into one grayscale channel for [G,G,G] equivalence initialization",
    "imgsz": $IMG_SIZE,
    "num_classes": len(names),
    "class_names": names,
    "detect_layer": 24,
    "head6_outputs": {
        "cls_heads": 3,
        "reg_heads": 3,
        "cls_channels": len(names),
        "reg_channels": 64,
        "reg_max": 16
    },
    "board_compile_define": "A1_YOLO_NUM_CLASSES=25",
    "onnx_full": "$EXPORT_DIR/graynav_dce_yolov8n.onnx",
    "onnx_head6": "$EXPORT_DIR/graynav_dce_yolov8n_head6.onnx",
    "calibration_zip": "$CALIB_DIR/datasets.zip",
    "test_metrics_json": "$STATE_DIR/test_metrics.json",
    "operator_audit_head6": "$EXPORT_DIR/a1_ops_audit_head6.json"
}
Path("$STATE_DIR/final_model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False, indent=2))
PY

cat > "$STATE_DIR/cloud_pack_command.txt" <<EOF
cd /root/autodl-tmp/graynav-dce-fair/model_optimization
tar -czf /root/autodl-tmp/rod25_gray1_dce_final_model_artifacts.tar.gz \\
  artifacts/rod25_gray1_dce_final_state \\
  artifacts/rod25_gray1_dce_final_export \\
  artifacts/rod25_gray1_dce_final_calib \\
  logs
EOF

echo "Final true-single-channel GrayNav DCE model artifacts ready."
echo "weights: $FINAL_GRAY1_WEIGHTS"
echo "full onnx: $EXPORT_DIR/graynav_dce_yolov8n.onnx"
echo "head6 onnx: $EXPORT_DIR/graynav_dce_yolov8n_head6.onnx"
echo "calibration zip: $CALIB_DIR/datasets.zip"
echo "metadata: $STATE_DIR/final_model_meta.json"
