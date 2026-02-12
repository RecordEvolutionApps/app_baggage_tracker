# Copilot Instructions

## Goal
This repository runs an end-to-end smart baggage tracking system with camera configuration, live low-latency streaming, and a cloud-backed dashboard of detections.

## High-level architecture
- Docker Compose orchestrates three services: Janus (WebRTC gateway), Video (edge inference + stream pipeline), and Web (UI + API).
- The Video service performs object detection on camera frames, applies per-zone rules, and publishes live video to Janus.
- The Web service exposes two experiences:
  - Camera UI for live viewing and configuring detection zones and model parameters.
  - Dashboard view for aggregated detection history stored in shared data volume.

## Service map
- janus/: WebRTC gateway (Janus) used for low-latency video streaming.
- video/: Inference and stream pipeline (YOLO, optional SAHI, per-zone counting, camera input).
- web/:
  - backend/: Bun + Elysia API and static assets.
  - frontend/: Lit-based web components for the camera UI.

## Shared storage and data flow
- A shared Docker volume named "data" is mounted into video/ and web/ containers.
- Video writes detection snapshots and history to /data; Web reads from /data to render dashboard and status.

## Key files
- docker-compose.yml: Brings up janus, video, and web.
- video/requirements.txt: Python dependencies for inference and stream pipeline.
- web/backend/src/index.ts: HTTP API and static asset serving.
- web/frontend/src/: Camera UI components and live player.

## How to run (local)
- docker compose up --build
- Web UI available at port 1100; Janus at 1200.

## Copilot guidance
- Prefer small, scoped changes within a single service at a time.
- Keep cross-service contracts stable (data formats written to /data, API endpoints, and Janus stream names).
- When modifying detection logic, ensure per-zone counting and class filtering remain consistent with COCO classes.
- When editing frontend components, maintain Lit patterns and ESM module imports.
