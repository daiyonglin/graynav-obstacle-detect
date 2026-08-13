param(
    [string]$WorkRoot = "E:\jichuang\graynav_local_training"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $WorkRoot "env\Scripts\python.exe"
$RawRoot = Join-Path $WorkRoot "data\raw"
$ExtractRoot = Join-Path $WorkRoot "data\public"
$PreparedRoot = Join-Path $WorkRoot "data\surface_depth_prepared_v2"
$AdeArchive = Join-Path $RawRoot "ADEChallengeData2016.zip"
$StairArchive = Join-Path $RawRoot "RGB-D stair dataset.zip"
$NyuMat = Join-Path $RawRoot "nyu_depth_v2_labeled.mat"
$NyuSplits = Join-Path $RawRoot "splits.mat"

foreach ($path in @($PythonExe, $AdeArchive, $StairArchive, $NyuMat, $NyuSplits)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required public dataset asset is missing: $path"
    }
}

New-Item -ItemType Directory -Force -Path $ExtractRoot | Out-Null
$AdeRoot = Join-Path $ExtractRoot "ADEChallengeData2016"
$StairRoot = Join-Path $ExtractRoot "StairNetV3"

if (-not (Test-Path -LiteralPath (Join-Path $AdeRoot "images\training"))) {
    & $PythonExe -c @"
from pathlib import Path
from zipfile import ZipFile
archive = Path(r'$AdeArchive')
target = Path(r'$ExtractRoot')
with ZipFile(archive) as handle:
    handle.extractall(target)
print('GRAYNAV_ADE_EXTRACT_OK')
"@
}

if (-not (Test-Path -LiteralPath (Join-Path $StairRoot "train\images"))) {
    & $PythonExe -c @"
from pathlib import Path
from zipfile import ZipFile
archive = Path(r'$StairArchive')
target = Path(r'$StairRoot')
prefix = 'RGB-D stair dataset/RGB-D stair dataset/'
allowed = ('train/images/', 'train/segmentations/', 'val/images/', 'val/segmentations/')
copied = 0
with ZipFile(archive) as handle:
    for info in handle.infolist():
        name = info.filename.replace('\\', '/')
        if not name.startswith(prefix):
            continue
        relative = name[len(prefix):]
        if not relative.startswith(allowed) or info.is_dir():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with handle.open(info) as source, destination.open('wb') as output:
            output.write(source.read())
        copied += 1
print('stair_files=', copied)
if copied != 5664:
    raise SystemExit(f'expected 5664 StairNet files, got {copied}')
print('GRAYNAV_STAIR_EXTRACT_OK')
"@
}

if (Test-Path -LiteralPath $PreparedRoot) {
    throw "Prepared output already exists; refusing to overwrite: $PreparedRoot"
}

Push-Location (Join-Path $RepoRoot "model_optimization")
try {
    & $PythonExe scripts\prepare_graynav_surface_depth_dataset.py `
        --ade-root $AdeRoot `
        --nyu-mat $NyuMat `
        --nyu-splits $NyuSplits `
        --stair-root $StairRoot `
        --output $PreparedRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Scene dataset preparation failed with exit code $LASTEXITCODE"
    }
    & $PythonExe scripts\audit_surface_depth_prepared.py `
        --data $PreparedRoot `
        --output (Join-Path $PreparedRoot "audit")
    if ($LASTEXITCODE -ne 0) {
        throw "Prepared dataset audit failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Output "GRAYNAV_LOCAL_SCENE_DATA_READY"
