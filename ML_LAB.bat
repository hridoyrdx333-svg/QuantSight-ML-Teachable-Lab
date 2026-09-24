@echo off
setlocal
cd /d "%~dp0"
title QuantSight ML Lab

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating .venv...
  where py >nul 2>nul
  if %errorlevel%==0 (py -m venv .venv) else (python -m venv .venv)
  if errorlevel 1 exit /b 1
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)

if not exist "config.json" copy /Y "config.example.json" "config.json" >nul
if not exist ".env" (
  copy /Y ".env.example" ".env" >nul
  echo Add MASSIVE_API_KEY in .env, save, then run ML_LAB.bat again.
  notepad ".env"
  exit /b 0
)

echo Starting orchestration...
start "" ".venv\Scripts\python.exe" launcher.py
ping 127.0.0.1 -n 3 >nul
start "" "http://127.0.0.1:8765"
exit /b 0
