$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location $root
$modelsDir = Join-Path $root "models"
if (-not (Test-Path $modelsDir)) {
  throw "Folder models belum ada. Jalankan scripts/run_experiment.ps1 dulu."
}

$latest = Get-ChildItem -Path $modelsDir -Directory |
  Where-Object { $_.Name -like "model_*" } |
  Sort-Object Name |
  Select-Object -Last 1

if (-not $latest) {
  throw "Belum ada folder model_xxx di models/."
}

$metricsPath = Join-Path $latest.FullName "metrics.json"
if (-not (Test-Path $metricsPath)) {
  throw "metrics.json tidak ditemukan di $($latest.FullName)"
}

Write-Host "[latest] $($latest.Name)"
Get-Content $metricsPath
Pop-Location
