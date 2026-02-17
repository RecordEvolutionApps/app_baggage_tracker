#!/bin/bash
set -euo pipefail

# Source arch-specific environment (e.g. LD_PRELOAD on Jetson/arm64)
if [ -f /etc/video-env.conf ]; then
    set -a
    . /etc/video-env.conf
    set +a
fi

UVICORN_ARGS="--host 0.0.0.0 --port 8000 --app-dir /app/video"

echo "Starting Video Service API on port 8000..."

exec uvicorn api:app $UVICORN_ARGS