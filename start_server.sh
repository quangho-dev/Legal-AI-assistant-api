#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting the API server (Uvicorn)..."

if [ -f ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  echo "Virtual environment not found. Run: uv sync"
  exit 1
fi

exec "$PYTHON" -m uvicorn src.server:app --reload --host 0.0.0.0 --port 8003
