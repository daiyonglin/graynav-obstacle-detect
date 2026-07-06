#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET_SOURCE="${DATASET_SOURCE:-/root/autodl-tmp/archive.zip}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/graynav_rod25_gray}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/yolov8n.pt}"
MODEL_YAML="${MODEL_YAML:-configs/graynav_dce_yolov8n.yaml}"
IMG_SIZE="${IMG_SIZE:-384}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-64}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-12}"
CACHE="${CACHE:-ram}"
EVAL_BATCH="${EVAL_BATCH:-64}"
RUN_TRAIN="${RUN_TRAIN:-1}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
TERMINAL_QUIET="${TERMINAL_QUIET:-0}"
TRAIN_VERBOSE="${TRAIN_VERBOSE:-1}"
LOG_DIR="${LOG_DIR:-logs/graynav_dce_generic_$(date +%Y%m%d_%H%M%S)}"

STATE_DIR="artifacts/graynav_dce_generic_state"
EXPORT_DIR="artifacts/graynav_dce_generic_export"
EVAL_DIR="artifacts/graynav_dce_generic_eval"
TB_DIR="runs/tensorboard/graynav_dce_generic"

mkdir -p "$STATE_DIR" "$EXPORT_DIR" "$EVAL_DIR" "$LOG_DIR"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-2}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: base weights not found: $WEIGHTS" >&2
  exit 2
fi
if [[ ! -e "$DATASET_SOURCE" ]]; then
  echo "ERROR: dataset source not found: $DATASET_SOURCE" >&2
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

if [[ "$REBUILD_DATASET" == "1" ]]; then
  echo "REBUILD_DATASET=1, removing $DATA_ROOT"
  rm -rf "$DATA_ROOT"
fi

if [[ ! -f "$DATA_ROOT/gray_dataset.yaml" ]]; then
  run_step prepare_generic_gray python scripts/prepare_generic_yolo_gray_dataset.py \
    --source "$DATASET_SOURCE" \
    --out "$DATA_ROOT" \
    --overwrite
else
  echo "Reusing existing generic gray dataset: $DATA_ROOT"
  if [[ -f "$DATA_ROOT/dataset_manifest.json" ]]; then
    python - <<PY
from pathlib import Path
import json
p=Path("$DATA_ROOT/dataset_manifest.json")
m=json.loads(p.read_text(encoding="utf-8"))
print("dataset_manifest:", json.dumps({"dataset": m.get("dataset"), "nc": m.get("nc"), "splits": m.get("splits")}, ensure_ascii=False, indent=2)[:5000])
PY
  fi
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

run_step audit_gray_dataset python scripts/audit_gray_dataset.py \
  --roots "$DATA_ROOT/images/train" "$DATA_ROOT/images/val" \
  --out "$STATE_DIR/audit_gray_report.json" \
  --sample-images 1000

VERBOSE_ARGS=()
if [[ "$TRAIN_VERBOSE" == "1" ]]; then
  VERBOSE_ARGS+=(--verbose)
fi

if [[ "$RUN_TRAIN" == "1" || ! -f "$STATE_DIR/M1_dce_final_weights.txt" ]]; then
  run_step train_dce_generic python scripts/train_graynav_dce_yolov8n.py \
    --model-yaml "$MODEL_YAML" \
    --data "$DATA_ROOT/gray_dataset.yaml" \
    --weights "$WEIGHTS" \
    --out-dir "$STATE_DIR" \
    --project runs/detect \
    --name graynav_dce_yolov8n_generic \
    --imgsz "$IMG_SIZE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --cache "$CACHE" \
    --final-weights-file "$STATE_DIR/M1_dce_final_weights.txt" \
    "${VERBOSE_ARGS[@]}"
fi

M1_WEIGHTS="$(cat "$STATE_DIR/M1_dce_final_weights.txt")"

run_step export_dce_generic python scripts/export_graynav_dce_head6.py \
  --weights "$M1_WEIGHTS" \
  --out-dir "$EXPORT_DIR/M1_graynav_dce_yolov8n_generic" \
  --imgsz "$IMG_SIZE" \
  --num-classes "$NC"

run_step evaluate_generic python scripts/evaluate_graynav_dce_generic.py \
  --dataset-root "$DATA_ROOT" \
  --data-yaml "$DATA_ROOT/gray_dataset.yaml" \
  --split test \
  --m0-weights "$WEIGHTS" \
  --m1-weights "$M1_WEIGHTS" \
  --out-dir "$EVAL_DIR" \
  --imgsz "$IMG_SIZE" \
  --device "$DEVICE" \
  --batch "$EVAL_BATCH" \
  --tensorboard-dir "$TB_DIR"

cat > "$STATE_DIR/cloud_pack_command.txt" <<EOF
cd /root/autodl-tmp/graynav-dce/model_optimization
tar -czf /root/autodl-tmp/graynav_dce_generic_min_results.tar.gz \\
  artifacts/graynav_dce_generic_state \\
  artifacts/graynav_dce_generic_eval/graynav_dce_generic_eval_summary.json \\
  /root/autodl-tmp/datasets/graynav_rod25_gray/dataset_manifest.json \\
  logs
EOF

echo "GrayNav-DCE generic experiment done."
echo "M1 weights: $M1_WEIGHTS"
echo "Eval summary: $EVAL_DIR/graynav_dce_generic_eval_summary.json"
echo "Logs: $LOG_DIR"
