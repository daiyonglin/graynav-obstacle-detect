#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${GRAYNAV_ROOT:-${SCRIPT_DIR}}"
PUBLIC_ROOT="${GRAYNAV_PUBLIC_ROOT:-/root/autodl-tmp/graynav_public}"
PREPARED="${GRAYNAV_PREPARED:-/root/autodl-tmp/graynav_surface_depth_prepared_v2}"

ADE_ROOT="${ADE20K_ROOT:-${PUBLIC_ROOT}/ADEChallengeData2016}"
NYU_MAT="${NYUV2_MAT:-${PUBLIC_ROOT}/nyu_depth_v2_labeled.mat}"
NYU_SPLITS="${NYUV2_SPLITS:-${PUBLIC_ROOT}/splits.mat}"
STAIR_ROOT="${STAIRNET_ROOT:-${PUBLIC_ROOT}/StairNetV3}"

for path in "${ADE_ROOT}" "${NYU_MAT}" "${NYU_SPLITS}" "${STAIR_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing public dataset path: ${path}" >&2
    echo "Download/accept the official dataset terms, then place it under ${PUBLIC_ROOT}." >&2
    exit 2
  fi
done

python "${ROOT}/scripts/prepare_graynav_surface_depth_dataset.py" \
  --ade-root "${ADE_ROOT}" \
  --nyu-mat "${NYU_MAT}" \
  --nyu-splits "${NYU_SPLITS}" \
  --stair-root "${STAIR_ROOT}" \
  --output "${PREPARED}" \
  --overwrite

python "${ROOT}/scripts/audit_surface_depth_prepared.py" \
  --data "${PREPARED}" \
  --output "${PREPARED}/audit"
