FROM oven/bun:1.3-debian

WORKDIR /app

# --- Backend deps (cached unless package.json changes) ---
COPY backend/package.json /app/backend/
RUN cd /app/backend && bun i

# --- Frontend deps (cached unless package.json/lockfile changes) ---
COPY frontend/package.json frontend/bun.lockb /app/frontend/
RUN cd /app/frontend && bun i

# --- Copy source code (changes frequently) ---
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# --- Build frontend ---
RUN cd /app/frontend && bun run build

COPY entrypoint.sh .

ENTRYPOINT ["/app/entrypoint.sh"]