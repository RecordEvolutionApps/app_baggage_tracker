#!/bin/bash
set -euo pipefail

UVICORN_ARGS="--host 0.0.0.0 --port 8000 --app-dir /app/video"

if [ "${ENV:-}" = "DEV" ]; then
  echo "Starting Video Service API (dev mode with --reload) on port 8000..."
  UVICORN_ARGS="$UVICORN_ARGS --reload"
else
  echo "Starting Video Service API on port 8000..."
fi

exec uvicorn api:app $UVICORN_ARGS