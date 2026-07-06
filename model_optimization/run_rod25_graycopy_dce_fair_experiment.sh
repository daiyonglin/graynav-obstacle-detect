#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET_SOURCE="${DATASET_SOURCE:-/root/autodl-tmp/archive.zip}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/graynav_rod25_gray}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
IMG_SIZE="${IMG_SIZE:-384}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-12}"
CACHE="${CACHE:-ram}"
EVAL_BATCH="${EVAL_BATCH:-64}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
RUN_B1="${RUN_B1:-1}"
RUN_B2="${RUN_B2:-1}"
INCLUDE_R0_REFERENCE="${INCLUDE_R0_REFERENCE:-0}"
TRAIN_VERBOSE="${TRAIN_VERBOSE:-1}"
FREEZE="${FREEZE:-0}"
LR0="${LR0:-0.0003}"
LRF="${LRF:-0.01}"
MOSAIC="${MOSAIC:-0.25}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-15}"
LOG_DIR="${LOG_DIR:-logs/rod25_graycopy_dce_fair_$(date +%Y%m%d_%H%M%S)}"

STATE_DIR="artifacts/rod25_graycopy_dce_fair_state"
EVAL_DIR="artifacts/rod25_graycopy_dce_fair_eval"
TB_DIR="runs/tensorboard/rod25_graycopy_dce_fair"

mkdir -p "$STATE_DIR" "$EVAL_DIR" "$LOG_DIR"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-2}"

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
fi

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "========== ${name} =========="
  echo "log: ${log}"
  "$@" 2>&1 | tee "$log"
}

if [[ ! -f "$DATA_ROOT/gray_dataset.yaml" ]]; then
  run_step prepare_rod25_graycopy python scripts/prepare_generic_yolo_gray_dataset.py \
    --source "$DATASET_SOURCE" \
    --out "$DATA_ROOT" \
    --overwrite
else
  echo "Reusing prepared gray-copy dataset: $DATA_ROOT"
fi

NC="$(python - <<PY
import yaml
from pathlib import Path
d=yaml.safe_load(Path("$DATA_ROOT/gray_dataset.yaml").read_text(encoding="utf-8"))
names=d["names"]
print(len(names) if isinstance(names, list) else len(names.keys()))
PY
)"
echo "Detected dataset classes: $NC"

run_step audit_rod25_graycopy python scripts/audit_generic_yolo_dataset.py \
  --dataset-root "$DATA_ROOT" \
  --data-yaml "$DATA_ROOT/gray_dataset.yaml" \
  --out "$STATE_DIR/dataset_integrity_audit.json" \
  --gray-sample 3000 \
  --phash-sample 0

VERBOSE_ARGS=()
if [[ "$TRAIN_VERBOSE" == "1" ]]; then
  VERBOSE_ARGS+=(--verbose)
fi

COMMON_TRAIN_ARGS=(
  --data "$DATA_ROOT/gray_dataset.yaml"
  --weights "$WEIGHTS"
  --imgsz "$IMG_SIZE"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --device "$DEVICE"
  --workers "$WORKERS"
  --cache "$CACHE"
  --lr0 "$LR0"
  --lrf "$LRF"
  --freeze "$FREEZE"
  --mosaic "$MOSAIC"
  --close-mosaic "$CLOSE_MOSAIC"
  "${VERBOSE_ARGS[@]}"
)

if [[ "$RUN_B1" == "1" || ! -f "$STATE_DIR/B1_yolov8n_graycopy_final_weights.txt" ]]; then
  run_step train_B1_yolov8n_graycopy python scripts/train_graynav_dce_yolov8n.py \
    --model-yaml configs/graynav_yolov8n_baseline.yaml \
    --out-dir "$STATE_DIR/B1_yolov8n_graycopy" \
    --project runs/detect \
    --name B1_yolov8n_rod25_graycopy_ft \
    --final-weights-file "$STATE_DIR/B1_yolov8n_graycopy_final_weights.txt" \
    "${COMMON_TRAIN_ARGS[@]}"
else
  echo "Reusing B1 weights: $(cat "$STATE_DIR/B1_yolov8n_graycopy_final_weights.txt")"
fi

if [[ "$RUN_B2" == "1" || ! -f "$STATE_DIR/B2_dce_graycopy_final_weights.txt" ]]; then
  run_step train_B2_dce_graycopy python scripts/train_graynav_dce_yolov8n.py \
    --model-yaml configs/graynav_dce_yolov8n.yaml \
    --out-dir "$STATE_DIR/B2_dce_graycopy" \
    --project runs/detect \
    --name B2_dce_yolov8n_rod25_graycopy_ft \
    --final-weights-file "$STATE_DIR/B2_dce_graycopy_final_weights.txt" \
    "${COMMON_TRAIN_ARGS[@]}"
else
  echo "Reusing B2 weights: $(cat "$STATE_DIR/B2_dce_graycopy_final_weights.txt")"
fi

B1_WEIGHTS="$(cat "$STATE_DIR/B1_yolov8n_graycopy_final_weights.txt")"
B2_WEIGHTS="$(cat "$STATE_DIR/B2_dce_graycopy_final_weights.txt")"

EXTRA_ARGS=()
if [[ "$INCLUDE_R0_REFERENCE" == "1" ]]; then
  EXTRA_ARGS+=(--extra-model "R0_coco_yolov8n_overlap_reference=$WEIGHTS")
fi

run_step evaluate_B1_B2_fair python scripts/evaluate_graynav_dce_generic.py \
  --dataset-root "$DATA_ROOT" \
  --data-yaml "$DATA_ROOT/gray_dataset.yaml" \
  --split test \
  --m0-name B1_yolov8n_rod25_graycopy_ft \
  --m0-weights "$B1_WEIGHTS" \
  --m1-name B2_dce_yolov8n_rod25_graycopy_ft \
  --m1-weights "$B2_WEIGHTS" \
  "${EXTRA_ARGS[@]}" \
  --out-dir "$EVAL_DIR" \
  --imgsz "$IMG_SIZE" \
  --device "$DEVICE" \
  --batch "$EVAL_BATCH" \
  --tensorboard-dir "$TB_DIR"

run_step summarize_B2_vs_B1 python scripts/summarize_graynav_dce_generic_eval.py \
  --summary "$EVAL_DIR/graynav_dce_generic_eval_summary.json" \
  --out-dir "$EVAL_DIR/summary_tables" \
  --reference B1_yolov8n_rod25_graycopy_ft

cat > "$STATE_DIR/experiment_definition.json" <<EOF
{
  "experiment": "ROD25 GrayCopy DCE fair comparison",
  "primary_comparison": "B2_dce_yolov8n_rod25_graycopy_ft vs B1_yolov8n_rod25_graycopy_ft",
  "input_mode": "All train/val/test images are converted from BGR/RGB to single-channel grayscale, then replicated as [G,G,G].",
  "class_space": "Both primary models use the same ROD25 25-class detection head.",
  "base_weights": "$WEIGHTS",
  "dataset_root": "$DATA_ROOT",
  "epochs": $EPOCHS,
  "batch": "$BATCH",
  "lr0": $LR0,
  "lrf": $LRF,
  "imgsz": $IMG_SIZE,
  "note": "Original COCO yolov8n is optional overlap-only reference, not the primary all-class AP baseline."
}
EOF

cat > "$STATE_DIR/cloud_pack_command.txt" <<EOF
cd /root/autodl-tmp/graynav-dce/model_optimization
tar -czf /root/autodl-tmp/rod25_graycopy_dce_fair_min_results.tar.gz \\
  artifacts/rod25_graycopy_dce_fair_state \\
  artifacts/rod25_graycopy_dce_fair_eval/graynav_dce_generic_eval_summary.json \\
  artifacts/rod25_graycopy_dce_fair_eval/summary_tables \\
  logs
EOF

echo "ROD25 GrayCopy DCE fair experiment done."
echo "B1 weights: $B1_WEIGHTS"
echo "B2 weights: $B2_WEIGHTS"
echo "Eval summary: $EVAL_DIR/graynav_dce_generic_eval_summary.json"
echo "Pack command: $STATE_DIR/cloud_pack_command.txt"
