param(
    [Parameter(Mandatory=$true)]
    [string]$KaggleUsername,
    [string]$Slug = "minivigpt-from-scratch"
)

$ErrorActionPreference = "Stop"
kaggle kernels status "$KaggleUsername/$Slug"
