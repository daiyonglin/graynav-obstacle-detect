param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$SdkRoot = (Resolve-Path "$ProjectRoot\..\..\..\..\..").Path,
    [string]$OutputRoot = "E:\jichuang\submission",
    [string]$PackageName = "obstacle_detect_technical_data"
)

$ErrorActionPreference = "Stop"

$packageRoot = Join-Path $OutputRoot $PackageName
$zipPath = Join-Path $OutputRoot "$PackageName.zip"

if (Test-Path $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "source") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "binaries") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "images") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "scripts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "docs") | Out-Null

$sourceDst = Join-Path $packageRoot "source\ssne_ai_demo"
New-Item -ItemType Directory -Force -Path $sourceDst | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "*") -Destination $sourceDst -Recurse -Force

Get-ChildItem -LiteralPath $sourceDst -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $sourceDst -Recurse -File -Include "*.pyc","*.bak" | Remove-Item -Force

$buildExe = Join-Path $SdkRoot "output\build\ssne_ai_demo\ssne_ai_demo"
$targetExe = Join-Path $SdkRoot "output\target\app_demo\ssne_ai_demo"
if (Test-Path $targetExe) {
    Copy-Item -LiteralPath $targetExe -Destination (Join-Path $packageRoot "binaries\ssne_ai_demo") -Force
} elseif (Test-Path $buildExe) {
    Copy-Item -LiteralPath $buildExe -Destination (Join-Path $packageRoot "binaries\ssne_ai_demo") -Force
} else {
    Write-Warning "ssne_ai_demo executable not found. Build the SDK before packaging."
}

$imageDir = Join-Path $SdkRoot "output\images"
foreach ($name in @("zImage.smartsens-m1-evb", "rootfs.cpio.gz")) {
    $src = Join-Path $imageDir $name
    if (Test-Path $src) {
        Copy-Item -LiteralPath $src -Destination (Join-Path $packageRoot "images\$name") -Force
    } else {
        Write-Warning "image not found: $src"
    }
}

Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\run.sh") -Destination (Join-Path $packageRoot "scripts\run.sh") -Force
$voiceRunScript = Join-Path $ProjectRoot "scripts\run_voice_both.sh"
if (Test-Path $voiceRunScript) {
    Copy-Item -LiteralPath $voiceRunScript -Destination (Join-Path $packageRoot "scripts\run_voice_both.sh") -Force
}

foreach ($doc in @(
    "SUBMISSION_README.md",
    "OBSTACLE_DETECT_SYSTEM_DESIGN_REPORT.md",
    "OBSTACLE_DETECT_IMPLEMENTATION.md",
    "OBSTACLE_DETECT_SCREEN_OVERLAY_EXPLANATION.md",
    "OBSTACLE_DETECT_QUICK_GUIDE.md",
    "NEXT_STAGE_OPTIMIZATION_IMPLEMENTATION.md",
    "VOICE_UART_INTEGRATION.md",
    "README.md"
)) {
    $src = Join-Path $ProjectRoot $doc
    if (Test-Path $src) {
        Copy-Item -LiteralPath $src -Destination (Join-Path $packageRoot "docs\$doc") -Force
    }
}

Copy-Item -LiteralPath (Join-Path $ProjectRoot "SUBMISSION_README.md") -Destination (Join-Path $packageRoot "README_SUBMISSION.md") -Force

Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -Force

Write-Host "Package created:"
Write-Host $zipPath
