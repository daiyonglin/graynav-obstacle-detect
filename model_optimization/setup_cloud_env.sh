#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv-yolo
source .venv-yolo/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Use the CUDA wheel index matching the cloud image. cu121 is widely available
# and works on most recent NVIDIA training hosts. If the provider image already
# includes torch, this command is harmless but may take a while.
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

