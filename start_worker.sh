#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Celery worker..."

if [ -f ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
  POOL="solo"
elif [ -f ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
  # prefork + native libs (torch/onnx/HF) aborts on macOS after fork()
  if [ "$(uname -s)" = "Darwin" ]; then
    POOL="solo"
  else
    POOL="prefork"
  fi
else
  echo "Virtual environment not found. Run: uv sync"
  exit 1
fi

echo "Using pool: ${POOL}"
exec "$PYTHON" -m celery -A src.services.celery:celery_app worker --loglevel=info --pool="$POOL" --concurrency=1
