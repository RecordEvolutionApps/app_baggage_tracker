# Quick Start Guide

## Building for Different Platforms

### Jetson AGX (Production with CUDA/TensorRT)

```bash
# Use the default docker-compose.yml
docker compose up --build

# Or explicitly
docker compose -f docker-compose.yml up --build
```

**Requirements:**
- NVIDIA Jetson AGX with JetPack 5.x
- Docker with NVIDIA runtime configured
- USB camera or IP camera

### Local Development (Mac/Linux CPU-only)

```bash
# Use the development compose file with live code reload
docker compose -f docker-compose.dev.yml up --build
```

**Note:** The video container stays running but doesn't auto-start the video stream. To test video inference:

```bash
# In another terminal, exec into the video container
docker compose -f docker-compose.dev.yml exec video bash

# Run the video stream with the demo video
python3 /app/video/videoStream.py demoVideo frontCam

# Or with a USB camera
python3 /app/video/videoStream.py /dev/video0 frontCam
```

**Requirements:**
- Docker Desktop (Mac) or Docker Engine (Linux)
- No GPU required
- USB camera or IP camera (optional - demo video available)

## Configuration

Set environment variables in the IronFlock app settings or in a `.env` file:

```bash
# Model selection (required)
OBJECT_MODEL=rtmdet_tiny_8xb32-300e_coco

# Detection settings
CLASS_LIST=24,26,28  # COCO class IDs (leave empty for all classes)
CONF=0.1             # Confidence threshold (0-1)
IOU=0.5              # IoU threshold for NMS (0-1)

# Video settings
RESOLUTION_X=1280
RESOLUTION_Y=720
FRAMERATE=30

# Performance tuning
USE_SAHI=false       # Enable image slicing for small objects
SMOOTHING=false      # Enable detection smoothing
FRAME_BUFFER=180     # Extra pixels around detection zones when using SAHI
```

## Available Models

| Model Name | Speed | Use Case |
|------------|-------|----------|
| `rtmdet_tiny_8xb32-300e_coco` | Fastest | Default, best for real-time |
| `rtmdet_s_8xb32-300e_coco` | Fast | Balanced |
| `rtmdet_m_8xb32-300e_coco` | Moderate | Higher accuracy |

## Camera Setup

### USB Camera
Set device path in app settings:
```bash
/dev/video0  # First USB camera
/dev/video1  # Second USB camera
```

### IP Camera (RTSP)
```bash
rtsp://username:password@192.168.1.100:554/stream
```

### Demo Video
Use the built-in demo:
```bash
demoVideo
```

## Accessing the Web Interface

After starting the containers:

- **Live Stream**: http://localhost:1100
- **Video Configuration**: Available via the web UI
- **Zone/Line Drawing**: Interactive canvas on the live stream page

## Troubleshooting

### Models Not Downloading

If models fail to download, check:
1. Container has internet access
2. `/data` volume is mounted and writable
3. Check logs: `docker compose logs video`

### Low FPS

1. Use smaller model: `rtmdet_tiny_8xb32-300e_coco`
2. Reduce resolution: `RESOLUTION_X=640 RESOLUTION_Y=480`
3. Disable SAHI: `USE_SAHI=false`
4. On Jetson: Ensure NVIDIA runtime is active (`docker info | grep nvidia`)

### Camera Not Detected

1. Check USB devices: `docker compose exec video v4l2-ctl --list-devices`
2. Verify devices are mounted in docker-compose.yml
3. Try privileged mode (already enabled in compose files)

## Development

### Rebuilding After Code Changes

```bash
# Production (Jetson)
docker compose up --build

# Development (local with live reload)
docker compose -f docker-compose.dev.yml up --build
```

### SSH Access (Production Only)

The production container runs an SSH server:
1. SSH keys are generated in `/data/ssh/`
2. Use `docker compose exec video bash` for direct access

## Data Persistence

All data is stored in the `data` Docker volume:
- `/data/mmdet/`: Model configs and checkpoints
- `/data/mask.json`: Zone and line definitions
- `/data/ssh/`: SSH keys (production only)
