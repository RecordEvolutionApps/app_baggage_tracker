# Migration to MMDetection Backend

This document summarizes the migration from Ultralytics to a pure MMDetection-based object detection pipeline.

## Changes Made

### 1. Docker Images

**Jetson AGX Production** ([video/Dockerfile](video/Dockerfile))
- Base: `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` (JetPack 5.x, PyTorch 2.0, Python 3.10)
- Includes CUDA/TensorRT support for GPU acceleration
- MMDetection installed via openmim (mim) for optimal compatibility
- Multi-layer build for better Docker cache utilization
- Removed all Ultralytics dependencies

**Local Development** ([video/Dockerfile.cpu](video/Dockerfile.cpu))
- Base: `openmmlabbase/mmdetection:latest` (pre-built with PyTorch, mmcv, mmdet)
- Much faster build times since MMDetection ecosystem is pre-installed
- Only installs application-specific dependencies
- Lighter and faster for Mac/Linux dev containers

### 2. Docker Compose Files

- [docker-compose.yml](docker-compose.yml): Production (Jetson with NVIDIA runtime)
- [docker-compose.dev.yml](docker-compose.dev.yml): Local dev (CPU-only, no NVIDIA runtime, with live code reload)

### 3. Model Backend ([video/model_utils.py](video/model_utils.py))

**Removed:**
- All Ultralytics/YOLO imports and functions
- `get_ultralytics_model()`, `infer_ultralytics()`, `download_ultralytics_model()`
- OBB-specific annotation logic

**Added:**
- MMDetection model zoo mapping (`MMDET_MODEL_ZOO`) with RTMDet variants
- `get_mmdet_model()`: Downloads config + checkpoint on demand, caches in `/data/mmdet`
- `infer_mmdet()`: Runs MMDetection inference and converts to `sv.Detections`
- Unified backend abstraction (`model_bundle` dict) for future extensibility

**Backend Selection:**
- `DETECT_BACKEND` removed from config (always `mmdet` now)
- `OBJECT_MODEL` defaults to `rtmdet_tiny_8xb32-300e_coco`

### 4. Dependencies ([video/requirements.txt](video/requirements.txt))

**Added:**
- `mmdet>=3.3.0`
- `mmengine>=0.10.0`
- `mmcv>=2.1.0`
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`

**Removed:**
- No explicit Ultralytics dependency

### 5. Configuration ([.ironflock/env-template.yml](.ironflock/env-template.yml))

- `OBJECT_MODEL` dropdown now lists RTMDet variants (tiny/s/m)
- Removed `DETECT_BACKEND` parameter
- Updated descriptions to reflect MMDetection

### 6. Documentation ([README.md](README.md))

- Updated parameter table to reflect MMDetection backend
- Changed hardware requirements to mention CPU fallback for dev
- Removed YOLO/Ultralytics references

## Model Zoo

Currently available models (COCO-trained):

| Model | Size | Speed | Description |
|-------|------|-------|-------------|
| `rtmdet_tiny_8xb32-300e_coco` | Tiny | Fastest | Max FPS, good for Jetson |
| `rtmdet_s_8xb32-300e_coco` | Small | Fast | Balanced speed/accuracy |
| `rtmdet_m_8xb32-300e_coco` | Medium | Moderate | Higher accuracy |

Models are downloaded at runtime on first use and cached in `/data/mmdet/`.

## Hardware Acceleration

- **Jetson AGX**: CUDA inference via PyTorch on GPU
- **Local Dev**: CPU inference via PyTorch (slower but functional)
- **Future**: TensorRT export for RTMDet (via MMDeploy/torch2trt) can be added

## Usage

**Jetson (production):**
```bash
docker compose up --build
```

**Local dev (Mac/Linux):**
```bash
docker compose -f docker-compose.dev.yml up --build
```

## Migration Notes

- All existing zone/line/mask logic remains unchanged
- `supervision` Detections format is consistent across backends
- FPS monitoring, tracking, and smoothing still work
- WebRTC stream and mask updates via stdin unchanged

## Next Steps

1. **TensorRT Export**: Add TensorRT conversion for RTMDet on Jetson for max performance
2. **Extended Zoo**: Add more MMDetection models (YOLOX, Faster R-CNN, etc.)
3. **Custom Weights**: Support loading custom-trained MMDetection checkpoints via env var
