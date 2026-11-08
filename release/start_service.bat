@echo off
rem Clickbait Detector - start the local model service (Windows).
rem First run: creates a Python environment in .venv and installs packages (about 1 GB).
rem Later runs: starts straight away. Keep this window open while browsing.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto run

echo [1/3] Creating a Python environment in .venv ...
py -3.11 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" goto nopython

echo [2/3] Installing packages - first run only, this takes a few minutes ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

:run
echo [3/3] Starting the model service on http://127.0.0.1:8000 - keep this window open.
rem everything the model needs is in models\phobert_final, so no downloads are needed
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
".venv\Scripts\python.exe" -m service.app %*
pause
exit /b 0

:nopython
echo.
echo Python was not found. Install Python 3.11 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation, then run this file again.
pause
exit /b 1

:failed
echo.
echo Installing packages failed - see the messages above. Delete the .venv folder and try again.
pause
exit /b 1
