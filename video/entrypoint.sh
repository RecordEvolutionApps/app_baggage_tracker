#!/bin/bash
set -euo pipefail

UVICORN_ARGS="--host 0.0.0.0 --port 8000 --app-dir /app/video"

echo "Starting Video Service API on port 8000..."

exec uvicorn api:app $UVICORN_ARGS