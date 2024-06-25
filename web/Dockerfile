FROM oven/bun:1.0-debian

RUN apt-get update && apt-get install -y openssh-client

WORKDIR /app

COPY backend/package.json /app/backend/
RUN cd /app/backend && bun i
COPY backend/* /app/backend/
COPY entrypoint.sh .
COPY frontend /app/frontend
RUN cd frontend && bun i 
RUN cd frontend && bun build ./src/camera-shell.ts --outdir ./dist
RUN cp -a frontend/src/modules/janus.js frontend/dist/janus.js
RUN cp -a frontend/index.html frontend/dist/index.html

ENTRYPOINT ["/app/entrypoint.sh"]


# ssh -o StrictHostKeyChecking=no video