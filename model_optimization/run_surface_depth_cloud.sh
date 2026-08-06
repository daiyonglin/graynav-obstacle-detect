#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${GRAYNAV_ROOT:-${SCRIPT_DIR}}"
PREPARED="${GRAYNAV_PREPARED:-/root/autodl-tmp/graynav_surface_depth_prepared_v2}"
RUN_ROOT="${GRAYNAV_RUN_ROOT:-/root/autodl-tmp/graynav_surface_depth_optimized}"
EXPERIMENT="${GRAYNAV_EXPERIMENT:-e1}"
WIDTH_MULT="${GRAYNAV_WIDTH_MULT:-1.0}"
PRETRAINED="${GRAYNAV_PRETRAINED:-/root/autodl-tmp/graynav_surface_depth_run/preflight/fast_scnn_gray1_init.pt}"
EXPERIMENT_ROOT="${RUN_ROOT}/${EXPERIMENT}"

if [[ ! "${EXPERIMENT}" =~ ^e[123]$ ]]; then
  echo "GRAYNAV_EXPERIMENT must be e1, e2, or e3" >&2
  exit 2
fi
for path in "${PREPARED}/manifest_train.jsonl" "${PREPARED}/manifest_val.jsonl" "${PRETRAINED}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing experiment input: ${path}" >&2
    exit 3
  fi
done
mkdir -p "${EXPERIMENT_ROOT}/train" "${EXPERIMENT_ROOT}/tensorboard" "${RUN_ROOT}/static_audit"

# E1/E2 share one legacy32 graph. E3 receives its own true-64 graph audit.
ARCH="legacy32"
DETAIL_ARGS=()
if [[ "${EXPERIMENT}" == "e3" ]]; then
  ARCH="detail64"
  DETAIL_ARGS=(--detail64)
fi
if [[ ! -f "${RUN_ROOT}/static_audit/${ARCH}.a1_ops.json" ]]; then
  python "${ROOT}/scripts/export_graynav_surface_depth.py" \
    --random-init --width-mult "${WIDTH_MULT}" "${DETAIL_ARGS[@]}" \
    --onnx "${RUN_ROOT}/static_audit/${ARCH}.onnx"
  python "${ROOT}/scripts/audit_surface_depth_onnx.py" \
    --onnx "${RUN_ROOT}/static_audit/${ARCH}.onnx" \
    --report "${RUN_ROOT}/static_audit/${ARCH}.a1_ops.json"
fi

TRAIN_INIT=(--pretrained-fastscnn "${PRETRAINED}")
if [[ -f "${EXPERIMENT_ROOT}/train/last.pt" ]]; then
  TRAIN_INIT=(--resume "${EXPERIMENT_ROOT}/train/last.pt")
fi
BASELINE_ARGS=()
if [[ "${EXPERIMENT}" != "e1" ]]; then
  E0_METRICS="${RUN_ROOT}/e0/evaluation_prepared_v2.json"
  if [[ ! -f "${E0_METRICS}" ]]; then
    echo "missing fair E0 gradient baseline: ${E0_METRICS}" >&2
    exit 4
  fi
  BASELINE_ARGS=(--e0-metrics "${E0_METRICS}")
fi
python "${ROOT}/scripts/train_graynav_surface_depth.py" \
  --experiment "${EXPERIMENT}" \
  --data "${PREPARED}" \
  --output "${EXPERIMENT_ROOT}/train" \
  --log-dir "${EXPERIMENT_ROOT}/tensorboard" \
  --epochs 50 --batch-size 32 --workers 8 --lr 3e-4 \
  --weight-decay 0.01 --seed 42 --width-mult "${WIDTH_MULT}" --amp \
  "${BASELINE_ARGS[@]}" "${TRAIN_INIT[@]}"

python "${ROOT}/scripts/summarize_surface_depth_experiment.py" \
  --experiment "${EXPERIMENT}" \
  --train-dir "${EXPERIMENT_ROOT}/train" \
  --output "${EXPERIMENT_ROOT}/experiment_summary.json"

CANDIDATE="${EXPERIMENT_ROOT}/train/best_overall.pt"
if [[ ! -f "${CANDIDATE}" ]]; then
  CANDIDATE="${EXPERIMENT_ROOT}/train/best_seg.pt"
  echo "Gate did not produce best_overall.pt; visualizing best_seg.pt for diagnosis only."
fi
python "${ROOT}/scripts/evaluate_surface_depth_checkpoint.py" \
  --name "${EXPERIMENT}" --checkpoint "${CANDIDATE}" --data "${PREPARED}" \
  --output "${EXPERIMENT_ROOT}/candidate_evaluation.json" \
  --batch-size 32 --workers 8 --device cuda
if [[ ! -f "${EXPERIMENT_ROOT}/fixed_visualization/visualization_report.json" ]]; then
  python "${ROOT}/scripts/visualize_graynav_surface_depth.py" \
    --checkpoint "${CANDIDATE}" --data "${PREPARED}" \
    --output "${EXPERIMENT_ROOT}/fixed_visualization" \
    --fixed-regression-set --device cuda
fi

echo "Experiment complete: ${EXPERIMENT_ROOT}"
echo "Inspect experiment_summary.json before starting the next gated experiment."
