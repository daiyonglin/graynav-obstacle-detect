param(
    [string]$WorkRoot = "E:\GrayNavWorkspace",
    [string]$BasePython = "E:\Anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvRoot = Join-Path $WorkRoot "env"
$PythonExe = Join-Path $EnvRoot "Scripts\python.exe"

$directories = @(
    $WorkRoot,
    (Join-Path $WorkRoot "cache\pip"),
    (Join-Path $WorkRoot "cache\torch"),
    (Join-Path $WorkRoot "cache\conda_pkgs"),
    (Join-Path $WorkRoot "tmp"),
    (Join-Path $WorkRoot "data\raw"),
    (Join-Path $WorkRoot "data\prepared"),
    (Join-Path $WorkRoot "runs"),
    (Join-Path $WorkRoot "artifacts")
)
New-Item -ItemType Directory -Force -Path $directories | Out-Null

$env:PIP_CACHE_DIR = Join-Path $WorkRoot "cache\pip"
$env:TORCH_HOME = Join-Path $WorkRoot "cache\torch"
$env:CONDA_PKGS_DIRS = Join-Path $WorkRoot "cache\conda_pkgs"
$env:TEMP = Join-Path $WorkRoot "tmp"
$env:TMP = $env:TEMP
$env:YOLO_CONFIG_DIR = Join-Path $WorkRoot "cache\ultralytics"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    & $BasePython -m venv $EnvRoot
}

& $PythonExe -m pip install --upgrade pip setuptools wheel
& $PythonExe -m pip install `
    torch==2.5.1 torchvision==0.20.1 `
    --index-url https://download.pytorch.org/whl/cu118
& $PythonExe -m pip install `
    -r (Join-Path $PSScriptRoot "requirements.txt")

& $PythonExe -c @"
import os
import torch
import torchvision
import ultralytics
import onnx
import cv2
print('python=', os.sys.executable)
print('torch=', torch.__version__)
print('torchvision=', torchvision.__version__)
print('ultralytics=', ultralytics.__version__)
print('onnx=', onnx.__version__)
print('opencv=', cv2.__version__)
print('cuda=', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the isolated E-drive environment')
print('gpu=', torch.cuda.get_device_name(0))
print('vram_gib=', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print('GRAYNAV_LOCAL_ENV_OK')
"@
