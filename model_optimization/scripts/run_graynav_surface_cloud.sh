#!/usr/bin/env bash
set -Eeuo pipefail

# GrayNav 单卡云端训练入口。数据下载涉及数据集条款，因此本脚本只消费
# 已经由用户放置在持久数据盘上的官方数据和 PaddleSeg 预训练权重。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODEL_ROOT}/.." && pwd)"

: "${GRAYNAV_WORK_ROOT:=/root/autodl-tmp/graynav}"
: "${GRAYNAV_MAPILLARY_ROOT:?set GRAYNAV_MAPILLARY_ROOT to Mapillary Vistas v1.2 root}"
: "${GRAYNAV_STAIR_ROOT:?set GRAYNAV_STAIR_ROOT to the RGB-D stair dataset root}"
: "${GRAYNAV_PADDLE_CHECKPOINT:?set GRAYNAV_PADDLE_CHECKPOINT to official Fast-SCNN .pdparams}"
: "${GRAYNAV_BATCH_SIZE:=16}"
: "${GRAYNAV_WORKERS:=8}"
: "${GRAYNAV_EPOCHS:=80}"
: "${GRAYNAV_WIDTH_MULT:=1.0}"

DATA_ROOT="${GRAYNAV_WORK_ROOT}/datasets/graynav_surface"
RUN_ROOT="${GRAYNAV_WORK_ROOT}/runs/graynav_fast_scnn_w${GRAYNAV_WIDTH_MULT}"
INIT_WEIGHT="${RUN_ROOT}/paddleseg_gray1_init.pt"
LOG_ROOT="${GRAYNAV_WORK_ROOT}/logs"
mkdir -p "${DATA_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}"

cd "${MODEL_ROOT}"
python - <<'PY'
import json
import platform
import torch

report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(report, ensure_ascii=False))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing paid cloud training")
PY

if [[ ! -f "${DATA_ROOT}/dataset_summary.json" ]]; then
    python scripts/prepare_graynav_surface_dataset.py \
        --mapillary-root "${GRAYNAV_MAPILLARY_ROOT}" \
        --stair-root "${GRAYNAV_STAIR_ROOT}" \
        --output "${DATA_ROOT}"
fi

if [[ ! -f "${INIT_WEIGHT}" ]]; then
    python scripts/import_paddleseg_fast_scnn.py \
        --paddle-checkpoint "${GRAYNAV_PADDLE_CHECKPOINT}" \
        --output "${INIT_WEIGHT}" \
        --width-mult "${GRAYNAV_WIDTH_MULT}"
fi

TRAIN_ARGS=(
    --data "${DATA_ROOT}"
    --output "${RUN_ROOT}"
    --epochs "${GRAYNAV_EPOCHS}"
    --batch-size "${GRAYNAV_BATCH_SIZE}"
    --workers "${GRAYNAV_WORKERS}"
    --width-mult "${GRAYNAV_WIDTH_MULT}"
    --amp
)
if [[ -f "${RUN_ROOT}/last.pt" ]]; then
    TRAIN_ARGS+=(--resume "${RUN_ROOT}/last.pt")
else
    TRAIN_ARGS+=(--pretrained "${INIT_WEIGHT}")
fi

python scripts/train_graynav_fast_scnn.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "${LOG_ROOT}/train_w${GRAYNAV_WIDTH_MULT}.log"

ONNX_PATH="${RUN_ROOT}/graynav_fast_scnn_gray1_4cls_256.onnx"
python scripts/export_graynav_fast_scnn.py \
    --checkpoint "${RUN_ROOT}/best.pt" \
    --onnx "${ONNX_PATH}"
python scripts/audit_surface_onnx.py --onnx "${ONNX_PATH}"
python scripts/validate_surface_onnx.py \
    --checkpoint "${RUN_ROOT}/best.pt" \
    --onnx "${ONNX_PATH}" \
    --images "${DATA_ROOT}/images/val" \
    --report "${RUN_ROOT}/onnx_consistency.json"
python scripts/build_surface_calibration_set.py \
    --data "${DATA_ROOT}" \
    --output "${RUN_ROOT}/int8_calibration" \
    --count 200

python - <<PY
from pathlib import Path
for path in sorted(Path("${RUN_ROOT}").glob("*")):
    if path.is_file():
        print(f"artifact={path} bytes={path.stat().st_size}")
PY
