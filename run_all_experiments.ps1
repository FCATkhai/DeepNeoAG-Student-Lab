$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Step 1: Building / Verifying 2024-2026 IEDB Temporal Test Set" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
python scripts\step1_build_temporal_dataset.py

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " Step 2: Extracting All 8 Protein Representations (PLMs & Bio)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
python scripts\step2_extract_all_plms.py

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " Step 3: Training & Evaluating 6 Deep Learning Architectures" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
python scripts\step3_train_and_evaluate_models.py

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " Step 4: Generating Figures in figures/" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
python scripts\step4_plot_results.py

Write-Host "`nAll steps completed successfully!" -ForegroundColor Green

