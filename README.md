
# Vision AI Streaming System

**See everything. Understand everything. In real time.**

Turn any camera into an intelligent eye. This app brings the power of cutting-edge computer vision AI directly to your cameras — detecting, counting, and tracking objects in a live video stream with stunning accuracy, all viewable from your browser in real time.

Whether you're monitoring a warehouse floor, counting vehicles on a highway, keeping an airport baggage belt running smoothly, or watching wildlife in the field — this is the only app you need.


<div style="display:flex;flex-direction:row;align-items:center;justify-content:center;">
    <img src="https://res.cloudinary.com/dotw7ar1m/image/upload/v1717493714/baggaeAppBanner.png" width="650px">
</div>


---

## What Can You Do With This App?

### Watch AI-Powered Live Video

Open your browser and see your camera feed — enhanced with real-time AI detections. Every object is identified, labelled, and highlighted with bounding boxes, right on top of the live video. The stream is ultra low-latency, so what you see is what's happening *right now*.

### Detect Anything

Choose from **over 500 pre-trained AI models** spanning 30+ detection architectures. From people and vehicles to animals, furniture, food, and industrial equipment — there's a model for virtually every scenario. Browse models by category, read descriptions of what each one is best at, and switch models on the fly.

Need to detect something unusual? With **open-vocabulary models** like Grounding DINO and YOLO-World, simply type what you're looking for — *"red hard hat"*, *"forklift"*, *"unattended bag"* — and the AI will find it. No retraining required.

### Draw Smart Zones and Counting Lines

Don't just detect — **focus**. Draw polygon zones directly on the live video to count objects only in the areas that matter to you. Set up counting lines to track how many objects cross a boundary and in which direction. Perfect for:

- Counting people entering or leaving a room
- Monitoring items on a conveyor belt
- Tracking vehicles passing through an intersection
- Watching inventory levels in a storage zone

### Run Multiple Cameras

Add as many camera streams as you want. Each stream gets its own model, its own settings, and its own detection zones. Manage them all from a clean, intuitive gallery view. USB cameras, network cameras (RTSP/HTTP), YouTube streams — they all work.

### Fine-Tune to Perfection

Every stream is fully configurable:

- **Confidence threshold** — control how certain the AI needs to be before reporting a detection.
- **Class filtering** — choose exactly which object types to detect and ignore the rest.
- **SAHI sliced inference** — enable tiled detection for spotting tiny objects in high-resolution footage.
- **Object smoothing** — reduce jitter with temporal averaging across frames.
- **NMS tuning** — eliminate duplicate detections with adjustable overlap thresholds.

### Dashboard Your Insights

All detection data — zone counts, line crossings, annotated snapshots — is automatically collected and published to a shared cloud dashboard. See trends over time, compare cameras, and get a bird's-eye view of what's happening across your entire camera fleet. Data from all devices in your swarm flows into one unified view.

---

## Works With Any Camera

| Camera Type | Examples |
|-------------|----------|
| USB | Any standard webcam or industrial USB camera |
| MIPI CSI / GMSL | Jetson-native camera modules and multi-cam setups |
| Network (RTSP/HTTP) | IP cameras, NVRs, security camera streams |
| YouTube | Live streams and recorded videos via URL |
| Demo Video | Built-in test footage — no hardware needed to try it out |

---

## AI That Runs on the Edge

This app runs entirely on your device — your video never leaves your network unless you choose to publish snapshots to the dashboard. AI inference happens locally on the NVIDIA GPU with hardware-accelerated TensorRT, delivering real-time performance even on compact Jetson devices.

For development and testing, the app also runs on standard laptops and desktops without a GPU.

---

## Built for Scale

Deploy this app on one device or a hundred. Each device in your IronFlock fleet runs independently with its own cameras and models, while all detection data flows into a shared cloud dashboard. Add new cameras and devices at any time — the system grows with your needs.

---

## Key Specs

| Feature | Detail |
|---------|--------|
| Detection Models | 300+ from the MMDetection model zoo |
| Model Architectures | RTMDet, YOLO family, Faster R-CNN, DETR, Grounding DINO, YOLO-World, and 25+ more |
| Open-Vocabulary Detection | Yes — detect objects by typing text descriptions |
| GPU Acceleration | NVIDIA TensorRT FP16 (automatic engine caching) |
| Streaming Protocol | WebRTC via mediasoup SFU — sub-second browser latency |
| Detection Zones | Interactive polygon zones and directional counting lines |
| Object Tracking | ByteTrack multi-object tracker |
| Sliced Inference (SAHI) | Overlapping tile detection for small objects on high-res video |
| Camera Support | USB, MIPI CSI, GMSL, RTSP, HTTP, YouTube |
| Cloud Dashboard | Aggregated detection history, snapshots, and analytics |
| Multi-Device Fleet | Unlimited cameras across unlimited devices via IronFlock |

---

## Get Started

Install this app on your NVIDIA Jetson device through IronFlock, connect a camera, and open the web interface. You'll be watching AI-enhanced live video in minutes — no configuration files, no command lines, no complexity.

> **No camera? No problem.** The app includes a built-in demo video so you can explore every feature right away.
