set dotenv-load := true
set ignore-comments := true

# Start dev stack (frontend + backend) with bind mounts
dev:
  docker compose -f docker-compose.local.yml up --build -d

# Stop dev stack
dev-down:
  docker compose -f docker-compose.local.yml down -v
