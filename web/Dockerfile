FROM oven/bun:1.3-debian

RUN apt-get update && apt-get install -y openssh-client

WORKDIR /app

COPY backend/package.json /app/backend/
RUN cd /app/backend && bun i
COPY backend/* /app/backend/
COPY entrypoint.sh .
COPY frontend /app/frontend
RUN cd frontend && bun i 
RUN cd frontend && bun run build

ENTRYPOINT ["/app/entrypoint.sh"]


# ssh -o StrictHostKeyChecking=no video