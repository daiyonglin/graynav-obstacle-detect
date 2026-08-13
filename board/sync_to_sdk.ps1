param(
    [string]$SdkRoot = "E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$appSource = Join-Path $repoRoot "board\obstacle_detect"
$appTarget = Join-Path $SdkRoot "smart_software\src\app_demo\obstacle_detect\ssne_ai_demo"
$packageSource = Join-Path $repoRoot "board\sdk_overlay\ssne_ai_demo.mk"
$packageTarget = Join-Path $SdkRoot "smart_software\package\ssne_ai_demo\ssne_ai_demo.mk"
$startSource = Join-Path $repoRoot "board\rootfs_overlay\usr\smartsoc\smartsoc_start.sh"
$startTarget = Join-Path $SdkRoot "smart_software\board\m1pro\rootfs_overlay\usr\smartsoc\smartsoc_start.sh"

if (-not (Test-Path -LiteralPath $SdkRoot)) { throw "SDK root not found: $SdkRoot" }
if (-not (Test-Path -LiteralPath $appTarget)) { throw "SDK app target not found: $appTarget" }

# Copy the managed application files without deleting SDK-only notes, archived
# experiments or local hardware diagnostics. The Buildroot install whitelist,
# not filesystem mirroring, is responsible for keeping old models out of rootfs.
robocopy $appSource $appTarget /E /XD .git __pycache__ /XF *.pyc | Out-Host
if ($LASTEXITCODE -gt 7) { throw "robocopy app sync failed: $LASTEXITCODE" }

Copy-Item -LiteralPath $packageSource -Destination $packageTarget -Force
Copy-Item -LiteralPath $startSource -Destination $startTarget -Force

$model = Join-Path $appTarget "app_assets\models\graynav_unified_indoor8_scene21.m1model"
$hash = (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash
if ($hash -ne "33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8") {
    throw "Unified model hash mismatch after sync: $hash"
}

Write-Host "GRAYNAV_SDK_SYNC_OK"
Write-Host "app=$appTarget"
Write-Host "model_sha256=$hash"
