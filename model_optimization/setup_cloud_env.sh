#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${GRAYNAV_ROOT:-${SCRIPT_DIR}}"
WORK_ROOT="${GRAYNAV_WORK_ROOT:-/root/autodl-tmp/graynav}"
ENV_PREFIX="${GRAYNAV_ENV_PREFIX:-${WORK_ROOT}/env}"

mkdir -p "${WORK_ROOT}"/{cache/pip,cache/torch,weights,runs,logs,artifacts}
export PIP_CACHE_DIR="${WORK_ROOT}/cache/pip"
export TORCH_HOME="${WORK_ROOT}/cache/torch"

AVAILABLE_KB="$(df -Pk "${WORK_ROOT}" | awk 'NR==2 {print $4}')"
if [[ -z "${AVAILABLE_KB}" || "${AVAILABLE_KB}" -lt 47185920 ]]; then
  echo "At least 45 GiB free is required on the persistent data disk before setup." >&2
  df -h "${WORK_ROOT}" >&2
  exit 2
fi

source /root/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -p "${ENV_PREFIX}" python=3.10 -y
fi
conda activate "${ENV_PREFIX}"
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "${ROOT}/segmentation/requirements_surface.txt"
python -m pip install ultralytics==8.3.0 paddlepaddle==2.6.2

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("gpu", torch.cuda.get_device_name(0))
PY
