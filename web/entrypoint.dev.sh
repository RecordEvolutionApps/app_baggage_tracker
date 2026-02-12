#!/bin/bash
set -euo pipefail

cd /app/backend
bun install

cd /app/frontend
bun install

cd /app
bunx concurrently -k \
  "cd /app/backend && bun run dev" \
  "cd /app/frontend && bun run dev -- --host 0.0.0.0"
