#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REDIS_PORT="${REDIS_PORT:-6379}"

echo "Starting Redis on port ${REDIS_PORT}..."

if command -v redis-server >/dev/null 2>&1; then
  exec redis-server --port "$REDIS_PORT"
elif command -v docker >/dev/null 2>&1; then
  echo "redis-server not found. Starting Redis with Docker..."
  exec docker run --rm --name legal-ai-redis -p "${REDIS_PORT}:6379" redis:7-alpine
else
  echo "Redis is not available."
  echo "Install Redis locally, install Docker, or point REDIS_URL to an existing Redis instance."
  exit 1
fi
