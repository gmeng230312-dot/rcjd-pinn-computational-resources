@echo off
    setlocal
    cd /d "%~dp0"
    echo === STEP7 V4B FROZEN PRIOR REPLAY ===
    python "03_Posterior\step7v4b_compute_reliability.py" --config "03_Posterior\configs\step7_v4_config.json" --output-dir "outputs\posterior_replay"
    if errorlevel 1 goto :fail
    echo STEP7_SCRIPT_COMPLETED. Validate output tables against the manuscript before declaring published reproducibility.
    pause
    exit /b 0
    :fail
    echo STEP7 HOLD.
    pause
    exit /b 1
