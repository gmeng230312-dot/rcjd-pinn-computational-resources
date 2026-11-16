@echo off
    setlocal
    cd /d "%~dp0"
    echo === TG RCJD RELEASE CANDIDATE V0.2 ===
    python --version
    if errorlevel 1 goto :fail
    python -c "import numpy, scipy, pandas, matplotlib, torch; print('DEPENDENCIES_PASS')"
    if errorlevel 1 goto :fail
    python 00_PREFLIGHT.py
    if errorlevel 1 goto :fail
    python 00_VERIFY_RCJD.py
    if errorlevel 1 goto :fail
    echo === RCJD REPLAY PASS ===
    echo Next: run 01_RUN_POSTERIOR_VISIBLE.bat
    pause
    exit /b 0
    :fail
    echo ERROR / HOLD. See traceback above. Do not declare verification PASS.
    pause
    exit /b 1
