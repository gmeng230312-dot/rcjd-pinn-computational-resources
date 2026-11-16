@echo off
    setlocal
    cd /d "%~dp0\02_Inversion"
    echo === STEP6 ORIGINAL 30-CASE BENCHMARK ===
    python step6v2b_expanded_synthetic_benchmark.py
    if errorlevel 1 goto :fail
    echo INVERSION_SCRIPT_COMPLETED. Compare numerical results with archived paper reference.
    pause
    exit /b 0
    :fail
    echo INVERSION HOLD.
    pause
    exit /b 1
