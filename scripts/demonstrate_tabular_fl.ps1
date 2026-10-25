# MedShield-FL tabular federated demonstration (no implicit training)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== 1. Validate config ==="
py -m fl.runner validate-config configs/federations/tabular_cardio_v1_real_smoke.json

Write-Host "=== 2. Readiness ==="
py -m fl.runner readiness --config configs/federations/tabular_cardio_v1_real_smoke.json

Write-Host "=== 3. Verify existing latest run (if any) ==="
py scripts/verify_tabular_federated_run.py
if ($LASTEXITCODE -ne 0) { Write-Host "No valid prior run (optional)." }

Write-Host ""
Write-Host "To execute training, run explicitly:"
Write-Host "  py -m fl.runner run-tabular-experiment configs/federations/tabular_cardio_v1_real_smoke.json --confirm-run"
