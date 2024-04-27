FROM oven/bun:debian

RUN apt-get update && apt-get install -y openssh-client python3

WORKDIR /app

COPY backend/package.json /app/backend/
RUN cd /app/backend && bun i --production
COPY backend/* /app/backend/
COPY entrypoint.sh .
COPY frontend /app/frontend
RUN cd frontend && bun i 
RUN cd frontend && bun run build

ENTRYPOINT ["/app/entrypoint.sh"]


# ssh -o StrictHostKeyChecking=no video