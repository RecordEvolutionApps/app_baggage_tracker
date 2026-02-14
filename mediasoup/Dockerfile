FROM node:22-bookworm-slim

# mediasoup needs python3, pip, make, g++ for its C++ worker build
# On arm64 there is no prebuilt worker binary, so mediasoup compiles it
# from source using pip+invoke.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json .
RUN npm install --omit=dev

COPY src/ src/

EXPOSE 1200
# UDP ports for RTP ingest from video container
EXPOSE 5004/udp 5005/udp 5006/udp 5007/udp
# UDP port range for WebRTC
EXPOSE 2000-2999/udp

CMD ["node", "src/server.js"]
