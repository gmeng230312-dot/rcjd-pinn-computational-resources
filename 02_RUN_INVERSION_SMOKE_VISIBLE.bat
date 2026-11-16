@echo off
    setlocal
    cd /d "%~dp0\02_Inversion"
    echo === STEP6 INVERSION ONE-CASE SMOKE TEST, NOT FULL BENCHMARK ===
    python step6v2b_expanded_synthetic_benchmark.py --max-cases 1
    if errorlevel 1 goto :fail
    echo INVERSION_SMOKE_COMPLETED. Not proof of 30-case reproduction.
    pause
    exit /b 0
    :fail
    echo INVERSION HOLD.
    pause
    exit /b 1
