param(
    [string]$WorkRoot = "E:\jichuang\graynav_local_training"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $WorkRoot "env\Scripts\python.exe"
$RawRoot = Join-Path $WorkRoot "data\raw\voc2007"
$Archive = Join-Path $RawRoot "VOCtrainval_06-Nov-2007.tar"
$ExtractRoot = Join-Path $WorkRoot "data\public\VOCdevkit"
$Prepared = Join-Path $WorkRoot "data\detection_compact_v1"
$Replay = Join-Path $WorkRoot "data\coco128_indoor8_smoke"
$Url = "https://appcenter-deeplearning.sh1a.qingstor.com/dataset/voc/2007/VOCtrainval_06-Nov-2007.tar"

New-Item -ItemType Directory -Force -Path $RawRoot, $ExtractRoot | Out-Null
if (-not (Test-Path -LiteralPath $Archive) -or
    (Get-Item -LiteralPath $Archive).Length -ne 460032000) {
    & curl.exe -k -L --fail --retry 20 --retry-delay 5 --continue-at - `
        --output $Archive $Url
    if ($LASTEXITCODE -ne 0) {
        throw "VOC 2007 trainval download failed"
    }
}
if ((Get-Item -LiteralPath $Archive).Length -ne 460032000) {
    throw "VOC archive size does not match the official HTTP contract"
}

$VocRoot = Join-Path $ExtractRoot "VOC2007"
if (-not (Test-Path -LiteralPath (Join-Path $VocRoot "ImageSets\Main\train.txt"))) {
    & tar.exe -xf $Archive -C (Split-Path -Parent $ExtractRoot)
    if ($LASTEXITCODE -ne 0) {
        throw "VOC archive extraction failed"
    }
}
if (Test-Path -LiteralPath $Prepared) {
    throw "Prepared detection output already exists: $Prepared"
}

Push-Location (Join-Path $RepoRoot "model_optimization")
try {
    & $PythonExe scripts\prepare_voc2007_indoor8.py `
        --voc-root $VocRoot `
        --output $Prepared `
        --negative-modulus 20 `
        --coco128-replay $Replay
    if ($LASTEXITCODE -ne 0) {
        throw "VOC Indoor8 preparation failed"
    }
} finally {
    Pop-Location
}

Get-FileHash -Algorithm SHA256 -LiteralPath $Archive
Write-Output "GRAYNAV_LOCAL_COMPACT_DETECTION_READY"
