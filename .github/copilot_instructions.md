# Copilot Instructions

## Goal
This repository is a **generic, broadly applicable Vision AI streaming platform**. It connects any camera source to a real-time object detection pipeline with low-latency WebRTC streaming, an interactive web UI for configuring models and detection zones, and cloud-backed dashboarding of detection data. It is not limited to any single use case — it supports any vision AI scenario from traffic monitoring to warehouse safety to retail analytics.

## High-level architecture
- Docker Compose orchestrates three services: **mediasoup** (WebRTC SFU), **video** (edge inference + stream pipeline), and **web** (UI + API).
- The **video** service captures frames from any camera source (USB, MIPI CSI, GMSL, RTSP, HTTP, YouTube), runs object detection with a configurable model, applies per-zone/per-line counting rules, encodes annotated frames to H.264 via GStreamer (HW-accelerated on Jetson, software x264 fallback), and sends RTP to mediasoup.
- The **mediasoup** service is a Node.js WebRTC SFU (Selective Forwarding Unit) that receives RTP ingests from the video service and relays them to browser clients over WebRTC for sub-second latency live viewing.
- The **web** service exposes:
  - A **camera UI** (Lit web components) for live WebRTC viewing, multi-stream management, drawing detection zones/lines on the video canvas, and configuring models + inference parameters per stream.
  - A **dashboard view** for aggregated detection history stored in cloud storage via IronFlock.

## Service map
- `mediasoup/`: Node.js WebRTC SFU built on the mediasoup library. Exposes HTTP + WebSocket on port 1200. Creates plain RTP transports on demand (`POST /ingest`) and relays video to browser WebRTC consumers.
- `video/`: Python (FastAPI + OpenCV + GStreamer) inference and stream pipeline.
  - `api.py` — FastAPI app entrypoint; includes route modules, lifespan cleanup.
  - `videoStream.py` — Per-stream inference loop only: bootstrap, frame read, inference dispatch, annotation, publish, GStreamer write. All domain logic is delegated to the modules below.
  - `config.py` — `StreamConfig` dataclass centralising all mutable pipeline state (resolution, model, thresholds, class filtering, IronFlock identity). Also `parse_args()` and `create_config()` bootstrap helpers.
  - `video_source.py` — Video source management: opening USB, RTSP, HTTP, image, YouTube, and demo sources; `reopen_source()` handles looping awareness internally.
  - `model_loader.py` — `getModel()` dispatcher, `get_mmdet_model()`, `get_tensorrt_model()`, and monkey-patches for MMDet/MMCV compatibility.
  - `model_zoo.py` — `MMDET_MODEL_ZOO` dict, model download/cache CRUD, `prepare_model()`, `build_trt_for_model()`, `write_backend_status()`.
  - `inference.py` — `infer()`, `infer_mmdet()`, `infer_tensorrt()`, `empty_detections()`, `move_detections()`.
  - `sahi.py` — SAHI (Slicing Aided Hyper Inference): `initSliceInferer()`, `run_sahi_inference()`, `get_extreme_points()`, `SahiGridInfo` dataclass.
  - `frame_processing.py` — `processFrame()`: ByteTrack tracker, smoother, supervision annotators, per-zone/line counting.
  - `gstreamer.py` — `build_rtp_pipeline()` (HW nvv4l2h264enc on Jetson, x264enc fallback), `open_video_writer()`.
  - `publisher.py` — `Publisher` class wrapping IronFlock WAMP client for image/count publishing; `StubIronFlock` for dev mode.
  - `masks.py` — `prepMasks()`, `count_polygon_zone()`, `count_detections()`, `get_contrast_color()`.
  - `watchers.py` — Poll-based async file watchers for mask and settings files.
  - `overlay.py` — `overlay_text()`, `draw_sahi_grid()`.
  - `youtube.py` — `get_youtube_video()` via yt-dlp binary + Python library fallback.
  - `model_catalog.py` — Architecture knowledge base, MMDetection model discovery, dataset class lists, config notation decoder. Supports 30+ architectures (RTMDet, YOLO family, DETR, Grounding DINO, GLIP, Detic, Mask R-CNN, etc.).
  - `trt_backend.py` — TensorRT FP16 backend: ONNX export → engine build → cached inference on NVIDIA GPUs.
  - `routes/streams.py` — Start/stop/list stream processes; mediasoup ingest lifecycle.
  - `routes/cameras.py` — List local cameras (USB, MIPI CSI, GMSL) with supported resolutions.
  - `routes/models.py` — Model listing, class queries, cache management, model preparation with SSE progress.
  - `patch/polygon_zone.py` — Custom supervision polygon zone patch.
- `web/`:
  - `backend/` — Bun + Elysia HTTP API. Proxies video service endpoints, serves frontend assets, manages mask/zone data and per-stream settings.
  - `frontend/` — Lit-based web components (Material Web): stream gallery, stream editor, camera player, inference setup (model picker with tags/filters, SAHI, confidence, NMS IoU, class filtering, open-vocabulary class names), polygon/line zone drawing on a canvas overlay, WebRTC player via mediasoup-client.

## Camera input support
- USB cameras (UVC via `/dev/videoN`)
- MIPI CSI / GMSL cameras (via Jetson ISP, privileged mode)
- RTSP streams (GStreamer HW decode on Jetson, FFmpeg fallback)
- HTTP/HTTPS video streams (same decode strategy)
- YouTube live/VOD (via yt-dlp URL resolution)
- Built-in demo video loop

## Inference backends
- **MMDetection** (default on CPU): full MMDet inferencer with the entire model zoo (~300+ models across 30+ architectures).
- **TensorRT** (default on NVIDIA GPU): auto-exports MMDet models to ONNX → builds optimised FP16 TensorRT engines, cached on disk for instant reload.
- **SAHI** (optional): Sliced inference for small-object detection on high-res frames.

## Shared storage and data flow
- A shared Docker volume named `data` is mounted into video and web containers at `/data`.
- Video writes: detection snapshots, per-zone/line counts, stream resolution status, backend status, mask/settings files.
- Web reads from `/data` to render dashboard and stream configuration state.
- IronFlock SDK publishes images, detection counts, and line crossing events to cloud tables for dashboard aggregation.

## Key files
- `docker-compose.yml` — Production stack (NVIDIA runtime, Jetson).
- `docker-compose.dev.yml` — Local dev stack (CPU-only, bind mounts).
- `justfile` — Dev commands (`just dev`, `just dev-down`).
- `mediasoup/src/server.js` — WebRTC SFU server.
- `video/api.py` — FastAPI entrypoint.
- `video/videoStream.py` — Inference loop (lean orchestrator — delegates to modules below).
- `video/config.py` — `StreamConfig` dataclass + bootstrap helpers.
- `video/video_source.py` — Camera/source open, retry, loop/reopen logic.
- `video/model_loader.py` — Model loading dispatcher (MMDet, TensorRT).
- `video/model_zoo.py` — Model zoo dict, download/cache, TRT build, status.
- `video/inference.py` — Inference dispatch (`infer`, `infer_mmdet`, `infer_tensorrt`).
- `video/sahi.py` — SAHI slicer creation and tiled inference runner.
- `video/frame_processing.py` — Tracker, smoother, annotators, zone/line counting.
- `video/gstreamer.py` — GStreamer RTP pipeline builder.
- `video/publisher.py` — IronFlock publish wrapper.
- `video/masks.py` — Mask preparation and polygon zone counting.
- `video/watchers.py` — Async file watchers for masks/settings.
- `video/overlay.py` — Text overlay and SAHI grid drawing.
- `video/youtube.py` — YouTube URL resolution via yt-dlp.
- `video/model_catalog.py` — Model discovery and architecture knowledge base.
- `video/trt_backend.py` — TensorRT engine builder and runner.
- `video/requirements.txt` — Python dependencies.
- `web/backend/src/index.ts` — Elysia HTTP API.
- `web/frontend/src/` — Camera UI components (Lit + Material Web).

## How to run
- **Production (Jetson):** `docker compose up --build` — Web UI at `:1100`, mediasoup at `:1200`.
- **Local dev (CPU):** `just dev` or `docker compose -f docker-compose.dev.yml up --build` — Web at `:1100`, Vite dev server at `:5173`.

## Copilot guidance
- Prefer small, scoped changes within a single service at a time.
- Keep cross-service contracts stable: data formats written to `/data`, REST API endpoints between web↔video, RTP port allocation between video↔mediasoup, WebSocket protocol between mediasoup↔browser.
- The video service spawns one `videoStream.py` subprocess per camera stream. Each process is independent.
- `videoStream.py` is a thin inference loop — all domain logic lives in dedicated modules (`config.py`, `video_source.py`, `inference.py`, `sahi.py`, `frame_processing.py`, etc.). Keep it that way: push source-specific, model-specific, or SAHI-specific knowledge into the appropriate module, not into the loop.
- `StreamConfig` is the single source of mutable pipeline state. Functions accept it as a parameter — avoid cross-module global mutation.
- When modifying detection logic, ensure per-zone counting, per-line crossing, class filtering (by ID or open-vocab names), and ByteTrack tracking remain consistent.
- When editing frontend components, maintain Lit patterns, ESM module imports, and Material Web component usage.
- The model catalog (`model_catalog.py`) is a pure data/logic module with no FastAPI dependency — keep it that way.
- TensorRT engine builds are cached in `/data/tensorrt` and take 1-3 min on Jetson. Don't break the caching logic.
