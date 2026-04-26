$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location $root
./scripts/run_experiment.ps1
Pop-Location
