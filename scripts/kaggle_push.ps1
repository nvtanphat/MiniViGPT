param(
    [Parameter(Mandatory=$true)]
    [string]$KaggleUsername,
    [string]$Slug = "minivigpt-from-scratch",
    [string]$Config = "configs/minivigpt_20m.yaml",
    [string]$MachineShape = "NvidiaTeslaT4",
    [string[]]$DatasetSource = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

kaggle --version

$ArgsList = @(
    "scripts/build_kaggle_bundle.py",
    "--username", $KaggleUsername,
    "--slug", $Slug,
    "--config", $Config,
    "--machine-shape", $MachineShape
)
foreach ($Source in $DatasetSource) {
    $ArgsList += @("--dataset-source", $Source)
}
python @ArgsList

kaggle kernels push -p dist/kaggle

Write-Host ""
Write-Host "Kernel pushed: $KaggleUsername/$Slug"
Write-Host "Check status with:"
Write-Host ".\scripts\kaggle_status.ps1 -KaggleUsername `"$KaggleUsername`" -Slug `"$Slug`""
