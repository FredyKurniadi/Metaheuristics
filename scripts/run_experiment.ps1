$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location $root
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  throw "Python environment belum siap. Jalankan scripts/setup_all.ps1 dulu."
}

$env:PYTHONPATH = "$root\train\src"
.\.venv\Scripts\python.exe train/src/main.py --config train/configs/experiment.yaml
Pop-Location
