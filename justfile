set dotenv-load := true
set ignore-comments := true

# Start dev stack (frontend + backend) with bind mounts
dev:
  docker compose -f docker-compose.dev.yml up --build

# Stop dev stack
dev-down:
  docker compose -f docker-compose.dev.yml down
