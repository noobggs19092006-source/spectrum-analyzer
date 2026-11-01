@echo off
title Spectrum Demo Launcher
set PROJECT=C:\spectrum_rf
set RTL=C:\Users\aryan\Downloads\Release\x64\rtl_tcp.exe
set MODEL=C:\spectrum_rf\model_cfo15\classifier_cfo15.onnx
set FREQ=93.5e6

echo Starting RTL-SDR TCP server...
start "1 - rtl_tcp (V4)" cmd /k "%RTL% -a 127.0.0.1 -p 1234"
timeout /t 4 /nobreak >nul

echo Starting classifier service...
start "2 - inference service" cmd /k "cd /d %PROJECT% && call venv\Scripts\activate.bat && python service\inference_service.py --model %MODEL% --verbose"
timeout /t 6 /nobreak >nul

echo Starting live frame cutter...
start "3 - live frame cutter" cmd /k "cd /d %PROJECT% && call venv\Scripts\activate.bat && python live_frame_cutter.py --freq %FREQ% --continuous"
timeout /t 3 /nobreak >nul

echo Starting dashboard...
start "4 - dashboard" cmd /k "cd /d %PROJECT% && call venv\Scripts\activate.bat && python dashboard\dashboard.py"

echo All four windows launched. Press Ctrl+C in each window to stop.
