#!/usr/bin/env bash
# Clickbait Detector - start the local model service (macOS / Linux).
# First run: creates a Python environment in .venv and installs packages (about 1 GB).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "[1/3] Creating a Python environment in .venv ..."
  PY=$(command -v python3.11 || command -v python3 || true)
  if [ -z "$PY" ]; then echo "Python 3.11 is required: https://www.python.org/downloads/"; exit 1; fi
  "$PY" -m venv .venv
  echo "[2/3] Installing packages (first run only) ..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/python -m pip install -r requirements.txt
fi

echo "[3/3] Starting the model service on http://127.0.0.1:8000 - keep this terminal open."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec .venv/bin/python -m service.app "$@"
