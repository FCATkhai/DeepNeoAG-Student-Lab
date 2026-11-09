@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ==================================================================
echo  Step 1: Building / Verifying 2024-2026 IEDB Temporal Test Set
echo ==================================================================
python scripts\step1_build_temporal_dataset.py
if errorlevel 1 (
    echo [ERROR] Step 1 failed!
    exit /b %errorlevel%
)

echo.
echo ==================================================================
echo  Step 2: Extracting All 8 Protein Representations (PLMs ^& Bio)
echo ==================================================================
python scripts\step2_extract_all_plms.py
if errorlevel 1 (
    echo [ERROR] Step 2 failed!
    exit /b %errorlevel%
)

echo.
echo ==================================================================
echo  Step 3: Training ^& Evaluating 6 Deep Learning Architectures
echo ==================================================================
python scripts\step3_train_and_evaluate_models.py
if errorlevel 1 (
    echo [ERROR] Step 3 failed!
    exit /b %errorlevel%
)

echo.
echo ==================================================================
echo  Step 4: Generating Figures in figures/
echo ==================================================================
python scripts\step4_plot_results.py
if errorlevel 1 (
    echo [ERROR] Step 4 failed!
    exit /b %errorlevel%
)

echo.
echo ==================================================================
echo  All steps completed successfully!
echo ==================================================================

