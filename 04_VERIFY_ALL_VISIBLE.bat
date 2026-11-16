@echo off
    setlocal
    cd /d "%~dp0"
    echo === FINAL 3 MODULE NUMERICAL VERIFICATION ===
    python 04_VERIFY_RUN_RESULTS.py
    if errorlevel 1 goto :hold
    echo === ALL THREE CORE MODULES PASS ===
    pause
    exit /b 0
    :hold
    echo HOLD. See verification_results\final_gate.json
    pause
    exit /b 1
