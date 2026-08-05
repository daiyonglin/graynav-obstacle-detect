#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${GRAYNAV_ROOT:-${SCRIPT_DIR}}"
PREPARED="${GRAYNAV_PREPARED:-/root/autodl-tmp/graynav_surface_depth_prepared}"
RUN_ROOT="${GRAYNAV_RUN_ROOT:-/root/autodl-tmp/graynav_surface_depth_run}"
WIDTH_MULT="${GRAYNAV_WIDTH_MULT:-1.0}"
PADDLE_CHECKPOINT="${GRAYNAV_PADDLE_CHECKPOINT:-/root/autodl-tmp/graynav/weights/fast_scnn_cityscapes.pdparams}"
PRETRAINED="${RUN_ROOT}/preflight/fast_scnn_gray1_init.pt"

mkdir -p "${RUN_ROOT}/preflight" "${RUN_ROOT}/train" "${RUN_ROOT}/deploy"

# Mandatory A1 graph preflight before spending GPU hours.
python "${ROOT}/scripts/export_graynav_surface_depth.py" \
  --random-init --width-mult "${WIDTH_MULT}" \
  --onnx "${RUN_ROOT}/preflight/graynav_surface_depth_random.onnx"
python "${ROOT}/scripts/audit_surface_depth_onnx.py" \
  --onnx "${RUN_ROOT}/preflight/graynav_surface_depth_random.onnx" \
  --report "${RUN_ROOT}/preflight/a1_ops.json"

if [[ "${GRAYNAV_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight artifacts are ready. Submit the ONNX to the official A1 converter."
  exit 0
fi
if [[ "${GRAYNAV_A1_PREFLIGHT_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing to train before official A1 preflight confirmation." >&2
  echo "Re-run with GRAYNAV_A1_PREFLIGHT_CONFIRMED=1 after conversion succeeds." >&2
  exit 3
fi

TRAIN_INIT=()
if [[ -f "${RUN_ROOT}/train/last.pt" ]]; then
  echo "Resuming ${RUN_ROOT}/train/last.pt"
  TRAIN_INIT=(--resume "${RUN_ROOT}/train/last.pt")
else
  if [[ ! -f "${PADDLE_CHECKPOINT}" ]]; then
    echo "missing official PaddleSeg checkpoint: ${PADDLE_CHECKPOINT}" >&2
    exit 4
  fi
  python "${ROOT}/scripts/import_paddleseg_fast_scnn.py" \
    --paddle-checkpoint "${PADDLE_CHECKPOINT}" \
    --output "${PRETRAINED}" --width-mult "${WIDTH_MULT}"
  TRAIN_INIT=(--pretrained-fastscnn "${PRETRAINED}")
fi

python "${ROOT}/scripts/train_graynav_surface_depth.py" \
  --data "${PREPARED}" --output "${RUN_ROOT}/train" \
  --log-dir "${RUN_ROOT}/tensorboard" \
  --epochs 50 --batch-size 32 --workers 8 --lr 3e-4 \
  --weight-decay 0.01 --width-mult "${WIDTH_MULT}" --amp \
  "${TRAIN_INIT[@]}"
python "${ROOT}/scripts/export_graynav_surface_depth.py" \
  --checkpoint "${RUN_ROOT}/train/best.pt" \
  --width-mult "${WIDTH_MULT}" \
  --onnx "${RUN_ROOT}/deploy/graynav_surface_depth_gray1.onnx"
python "${ROOT}/scripts/audit_surface_depth_onnx.py" \
  --onnx "${RUN_ROOT}/deploy/graynav_surface_depth_gray1.onnx" \
  --report "${RUN_ROOT}/deploy/a1_ops.json"
python "${ROOT}/scripts/validate_surface_depth_onnx.py" \
  --checkpoint "${RUN_ROOT}/train/best.pt" \
  --onnx "${RUN_ROOT}/deploy/graynav_surface_depth_gray1.onnx" \
  --images "${PREPARED}/images/val" --limit 200 \
  --report "${RUN_ROOT}/deploy/onnx_consistency.json"
python "${ROOT}/scripts/build_a1_surface_depth_datasets.py" \
  --data "${PREPARED}" --output "${RUN_ROOT}/deploy" \
  --calibrate 160 --evaluate 40

sha256sum "${RUN_ROOT}/deploy/graynav_surface_depth_gray1.onnx" \
  "${RUN_ROOT}/deploy/datasets.zip" > "${RUN_ROOT}/deploy/SHA256SUMS.txt"
echo "Deployment bundle: ${RUN_ROOT}/deploy"
