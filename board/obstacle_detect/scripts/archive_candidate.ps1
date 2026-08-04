param(
    [Parameter(Mandatory = $true)][string]$CandidateName,
    [Parameter(Mandatory = $true)][string]$ZImage,
    [Parameter(Mandatory = $true)][string]$DetectorModel,
    [Parameter(Mandatory = $true)][string]$SurfaceDepthModel,
    [Parameter(Mandatory = $true)][string]$OnnxAudit,
    [Parameter(Mandatory = $true)][string]$CalibrationContract,
    [Parameter(Mandatory = $true)][string]$BoardLog,
    [string]$ArchiveRoot = "E:\jichuang\firmware_archive"
)

$ErrorActionPreference = "Stop"
$rollbackHash = "A7976710ECB456CB312D18F0195DCAE496ED652EFC582AB698EBC3EB7B055530"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$inputs = @($ZImage, $DetectorModel, $SurfaceDepthModel, $OnnxAudit, $CalibrationContract, $BoardLog)
foreach ($inputPath in $inputs) {
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "Required candidate artifact is missing: $inputPath"
    }
}

$resolvedRoot = [IO.Path]::GetFullPath($ArchiveRoot)
$target = [IO.Path]::GetFullPath((Join-Path $resolvedRoot $CandidateName))
if (-not $target.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar,
                           [StringComparison]::OrdinalIgnoreCase)) {
    throw "Candidate target escaped archive root: $target"
}
if (Test-Path -LiteralPath $target) {
    throw "Candidate archive already exists and will not be overwritten: $target"
}

$zHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ZImage).Hash
if ($zHash -eq $rollbackHash) {
    throw "The supplied zImage is the protected rollback image, not a new candidate."
}

New-Item -ItemType Directory -Path $target | Out-Null
$copied = @()
foreach ($inputPath in $inputs) {
    $destination = Join-Path $target ([IO.Path]::GetFileName($inputPath))
    Copy-Item -LiteralPath $inputPath -Destination $destination
    $copied += $destination
}

$commit = git -C $repo rev-parse HEAD
$status = git -C $repo status --short
$manifest = [ordered]@{
    candidate = $CandidateName
    created_at = (Get-Date).ToString("o")
    git_commit = $commit.Trim()
    git_status = @($status)
    rollback = [ordered]@{
        size = 8214488
        sha256 = $rollbackHash
    }
    artifacts = @(
        $copied | ForEach-Object {
            [ordered]@{
                file = [IO.Path]::GetFileName($_)
                bytes = (Get-Item -LiteralPath $_).Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_).Hash
            }
        }
    )
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $target "manifest.json") -Encoding UTF8
Write-Host "Archived candidate: $target"
Write-Host "zImage SHA256: $zHash"
