param(
    [string]$WorkRoot = "E:\GrayNavWorkspace",
    [int]$BatchSize = 16,
    [int]$AccumulationSteps = 2,
    [int]$Workers = 4,
    [int]$Epochs = 35,
    [int]$OptimizerStepsPerEpoch = 728,
    [int]$ValidationBatches = 30,
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $WorkRoot "env\Scripts\python.exe"
$CocoPrepared = Join-Path $WorkRoot "data\detection_compact_v1"
$ScenePrepared = Join-Path $WorkRoot "data\surface_depth_prepared_v2"
$RunRoot = Join-Path $WorkRoot "runs\unified_indoor8_v1"
$YoloWeights = Join-Path $RepoRoot "weights\yolov8n.pt"
$E3Checkpoint = Join-Path $RepoRoot "weights\graynav_surface_depth_e3_epoch49.pt"

$env:PIP_CACHE_DIR = Join-Path $WorkRoot "cache\pip"
$env:TORCH_HOME = Join-Path $WorkRoot "cache\torch"
$env:TEMP = Join-Path $WorkRoot "tmp"
$env:TMP = $env:TEMP
$env:YOLO_CONFIG_DIR = Join-Path $WorkRoot "cache\ultralytics"
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"

$required = @(
    $PythonExe,
    (Join-Path $CocoPrepared "manifest_train.jsonl"),
    (Join-Path $CocoPrepared "manifest_val.jsonl"),
    (Join-Path $ScenePrepared "manifest_train.jsonl"),
    (Join-Path $ScenePrepared "manifest_val.jsonl"),
    $YoloWeights,
    $E3Checkpoint
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required training asset is missing: $path"
    }
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $RunRoot "train"), `
    (Join-Path $RunRoot "tensorboard") | Out-Null

Push-Location (Join-Path $RepoRoot "training")
try {
    $arguments = @(
        "scripts\train_unified.py",
        "--coco", $CocoPrepared,
        "--scene", $ScenePrepared,
        "--yolo-weights", $YoloWeights,
        "--surface-e3", $E3Checkpoint,
        "--output", (Join-Path $RunRoot "train"),
        "--log-dir", (Join-Path $RunRoot "tensorboard"),
        "--epochs", $Epochs,
        "--scene-warmup-epochs", ([Math]::Min(5, $Epochs - 1)),
        "--steps-per-epoch", $OptimizerStepsPerEpoch,
        "--batch-size", $BatchSize,
        "--accumulation-steps", $AccumulationSteps,
        "--workers", $Workers,
        "--validation-batches", $ValidationBatches,
        "--lr", "3e-4",
        "--weight-decay", "0.01",
        "--seed", "42",
        "--amp"
    )
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $Resume)) {
            throw "Resume checkpoint is missing: $Resume"
        }
        $arguments += @("--resume", $Resume)
    }
    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Unified training failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
