#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/graynav_rod25_gray}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
DCE_WEIGHTS="${DCE_WEIGHTS:-}"
IMG_SIZE="${IMG_SIZE:-384}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-12}"
CACHE="${CACHE:-ram}"
EVAL_BATCH="${EVAL_BATCH:-64}"
TRAIN_VERBOSE="${TRAIN_VERBOSE:-1}"
LOG_DIR="${LOG_DIR:-logs/graynav_dce_ablation_$(date +%Y%m%d_%H%M%S)}"

STATE_DIR="artifacts/graynav_dce_ablation_state"
EVAL_DIR="artifacts/graynav_dce_ablation_eval"
TB_DIR="runs/tensorboard/graynav_dce_ablation"

mkdir -p "$STATE_DIR" "$EVAL_DIR" "$LOG_DIR"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-2}"

if [[ ! -f "$DATA_ROOT/gray_dataset.yaml" ]]; then
  echo "ERROR: prepared gray dataset not found: $DATA_ROOT/gray_dataset.yaml" >&2
  echo "Run run_graynav_dce_generic_experiment.sh once with REBUILD_DATASET=1 first." >&2
  exit 2
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: base weights not found: $WEIGHTS" >&2
  exit 2
fi

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "========== ${name} =========="
  echo "log: ${log}"
  "$@" 2>&1 | tee "$log"
}

NC="$(python - <<PY
import yaml
from pathlib import Path
d=yaml.safe_load(Path("$DATA_ROOT/gray_dataset.yaml").read_text(encoding="utf-8"))
names=d["names"]
print(len(names) if isinstance(names, list) else len(names.keys()))
PY
)"
echo "Detected dataset classes: $NC"

run_step audit_generic_yolo_dataset python scripts/audit_generic_yolo_dataset.py \
  --dataset-root "$DATA_ROOT" \
  --data-yaml "$DATA_ROOT/gray_dataset.yaml" \
  --out "$STATE_DIR/dataset_integrity_audit.json" \
  --gray-sample 3000 \
  --phash-sample 0

VERBOSE_ARGS=()
if [[ "$TRAIN_VERBOSE" == "1" ]]; then
  VERBOSE_ARGS+=(--verbose)
fi

if [[ ! -f "$STATE_DIR/M2_nodce_final_weights.txt" ]]; then
  run_step train_yolov8n_nodce python scripts/train_graynav_dce_yolov8n.py \
    --model-yaml configs/graynav_yolov8n_baseline.yaml \
    --data "$DATA_ROOT/gray_dataset.yaml" \
    --weights "$WEIGHTS" \
    --out-dir "$STATE_DIR" \
    --project runs/detect \
    --name graynav_yolov8n_generic_nodce \
    --imgsz "$IMG_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --cache "$CACHE" \
    --final-weights-file "$STATE_DIR/M2_nodce_final_weights.txt" \
    "${VERBOSE_ARGS[@]}"
else
  echo "Reusing no-DCE weights: $(cat "$STATE_DIR/M2_nodce_final_weights.txt")"
fi

M2_WEIGHTS="$(cat "$STATE_DIR/M2_nodce_final_weights.txt")"

if [[ -z "$DCE_WEIGHTS" ]]; then
  if [[ -f artifacts/graynav_dce_generic_state/M1_dce_final_weights.txt ]]; then
    DCE_WEIGHTS="$(cat artifacts/graynav_dce_generic_state/M1_dce_final_weights.txt)"
  else
    echo "ERROR: set DCE_WEIGHTS=/path/to/dce/best.pt or keep artifacts/graynav_dce_generic_state/M1_dce_final_weights.txt" >&2
    exit 2
  fi
fi
if [[ ! -f "$DCE_WEIGHTS" ]]; then
  echo "ERROR: DCE weights not found: $DCE_WEIGHTS" >&2
  exit 2
fi

run_step evaluate_three_models python scripts/evaluate_graynav_dce_generic.py \
  --dataset-root "$DATA_ROOT" \
  --data-yaml "$DATA_ROOT/gray_dataset.yaml" \
  --split test \
  --m0-weights "$WEIGHTS" \
  --m1-weights "$DCE_WEIGHTS" \
  --extra-model "M2_yolov8n_no_dce=$M2_WEIGHTS" \
  --out-dir "$EVAL_DIR" \
  --imgsz "$IMG_SIZE" \
  --device "$DEVICE" \
  --batch "$EVAL_BATCH" \
  --tensorboard-dir "$TB_DIR"

run_step summarize_eval python scripts/summarize_graynav_dce_generic_eval.py \
  --summary "$EVAL_DIR/graynav_dce_generic_eval_summary.json" \
  --out-dir "$EVAL_DIR/summary_tables"

cat > "$STATE_DIR/cloud_pack_command.txt" <<EOF
cd /root/autodl-tmp/graynav-dce/model_optimization
tar -czf /root/autodl-tmp/graynav_dce_ablation_min_results.tar.gz \\
  artifacts/graynav_dce_ablation_state \\
  artifacts/graynav_dce_ablation_eval/graynav_dce_generic_eval_summary.json \\
  artifacts/graynav_dce_ablation_eval/summary_tables \\
  logs
EOF

echo "GrayNav-DCE ablation done."
echo "DCE weights: $DCE_WEIGHTS"
echo "No-DCE weights: $M2_WEIGHTS"
echo "Eval summary: $EVAL_DIR/graynav_dce_generic_eval_summary.json"
echo "Pack command: $STATE_DIR/cloud_pack_command.txt"
