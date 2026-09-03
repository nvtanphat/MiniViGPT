param(
    [Parameter(Mandatory=$true)]
    [string]$KaggleUsername,
    [string]$Slug = "minivigpt-from-scratch"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $RepoRoot "artifacts\kaggle"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

kaggle kernels output "$KaggleUsername/$Slug" -p $Output -o
Write-Host "Downloaded to: $Output"
