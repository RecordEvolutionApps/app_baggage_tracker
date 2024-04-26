FROM oven/bun:1.1.5-alpine as base

RUN apk update && apk add --no-cache openssh-client

RUN mkdir -p /data/ssh
RUN ssh-keygen -t rsa -b 4096 -f /data/ssh/mykey -N "" -O exist && \
    cp -a /data/ssh/* ~/.ssh

WORKDIR /app

COPY backend/package.json /app/backend/
RUN cd /app/backend && bun i --production
COPY backend/* /app/backend/

COPY frontend /app/frontend
RUN cd frontend && . /root/.bashrc && bun i --production && bun run build

ENTRYPOINT [ "bun", "run", "/app/backend/index.ts" ]