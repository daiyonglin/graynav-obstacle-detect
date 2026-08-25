param(
    [Parameter(Mandatory = $true)][string]$ReleaseName,
    [Parameter(Mandatory = $true)][string]$ZImage,
    [Parameter(Mandatory = $true)][string]$UnifiedModel,
    [Parameter(Mandatory = $true)][string]$ConversionAudit,
    [Parameter(Mandatory = $true)][string]$BoardLog,
    [string]$ArchiveRoot = "E:\GrayNavReleases"
)

$ErrorActionPreference = "Stop"
$expectedModelHash = "33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$inputs = @($ZImage, $UnifiedModel, $ConversionAudit, $BoardLog)
foreach ($inputPath in $inputs) {
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "Required release artifact is missing: $inputPath"
    }
}

$resolvedRoot = [IO.Path]::GetFullPath($ArchiveRoot)
$target = [IO.Path]::GetFullPath((Join-Path $resolvedRoot $ReleaseName))
if (-not $target.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar,
                           [StringComparison]::OrdinalIgnoreCase)) {
    throw "Release target escaped archive root: $target"
}
if (Test-Path -LiteralPath $target) {
    throw "Release already exists and will not be overwritten: $target"
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $UnifiedModel).Hash -ne $expectedModelHash) {
    throw "Unified model hash mismatch"
}
if ((Get-Item -LiteralPath $ZImage).Length -ge 15MB) {
    throw "zImage exceeds the 15 MiB project release gate"
}

New-Item -ItemType Directory -Path $target | Out-Null
$artifacts = foreach ($inputPath in $inputs) {
    $destination = Join-Path $target ([IO.Path]::GetFileName($inputPath))
    Copy-Item -LiteralPath $inputPath -Destination $destination
    [ordered]@{
        file = [IO.Path]::GetFileName($destination)
        bytes = (Get-Item -LiteralPath $destination).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash
    }
}

$manifest = [ordered]@{
    release = $ReleaseName
    created_at = (Get-Date).ToString("o")
    git_commit = (git -C $repo rev-parse HEAD).Trim()
    git_status = @(git -C $repo status --short)
    artifacts = @($artifacts)
}
$manifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $target "manifest.json") -Encoding UTF8
Write-Host "GRAYNAV_RELEASE_ARCHIVED"
Write-Host "path=$target"
