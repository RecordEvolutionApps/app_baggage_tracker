import sys
import os

# ── Jetson TLS diagnostics & fix ───────────────────────────────────────────
# On Jetson (L4T), the static TLS block is limited. PyTorch + CUDA can exhaust
# it, preventing GStreamer NVIDIA plugins from loading libGLdispatch.so.0.
# We force-load these .so files before any heavy imports (torch, cv2).
import ctypes
_tls_libs = [
    '/lib/aarch64-linux-gnu/libGLdispatch.so.0',
    '/usr/lib/aarch64-linux-gnu/libGLESv2.so.2',
    '/usr/lib/aarch64-linux-gnu/libEGL.so.1',
    '/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvidconv.so',
    '/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so',
]
for _lib in _tls_libs:
    if os.path.exists(_lib):
        try:
            ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
            print(f'[TLS] Pre-loaded: {_lib}', flush=True)
        except OSError as e:
            print(f'[TLS] FAILED to pre-load {_lib}: {e}', flush=True)
    else:
        print(f'[TLS] Not found (skipped): {_lib}', flush=True)

print(f'[TLS] LD_PRELOAD={os.environ.get("LD_PRELOAD", "<not set>")}', flush=True)
del _lib, _tls_libs
# ────────────────────────────────────────────────────────────────────────────

import json
from asyncio import get_event_loop, sleep
import supervision as sv
import cv2
import argparse
import time
import logging
from datetime import datetime
from ironflock import IronFlock
from concurrent.futures import ThreadPoolExecutor

from pprint import pprint
import base64
import torch
import functools

from model_utils import getModel, processFrame, initSliceInferer, move_detections, get_extreme_points, infer, get_youtube_video, overlay_text, count_polygon_zone, count_detections, watchMaskFile, watchSettingsFile, empty_detections, FRAME_BUFFER, write_backend_status
import model_utils  # for updating CLASS_LIST dynamically

# Configure logging for the whole video service
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('videoStream')

OBJECT_MODEL = os.environ.get('OBJECT_MODEL')
RESOLUTION_X = int(os.environ.get('RESOLUTION_X', 640))
RESOLUTION_Y = int(os.environ.get('RESOLUTION_Y', 480))
FRAMERATE = int(os.environ.get('FRAMERATE', 30))
USE_SAHI = (os.environ.get('USE_SAHI', 'true') == "true")

DEVICE_KEY = os.environ.get('DEVICE_KEY')
DEVICE_URL = os.environ.get('DEVICE_URL')
CLASS_LIST = os.environ.get('CLASS_LIST', '')
CLASS_LIST = CLASS_LIST.split(',')
try:
    CLASS_LIST = [int(num.strip()) for num in CLASS_LIST]
except Exception as err:
    logger.warning('Invalid Class list given: %s', CLASS_LIST)
    CLASS_LIST = []

if len(CLASS_LIST) <= 1:
    CLASS_LIST = []

logger.info('Using CLASS_LIST: %s', CLASS_LIST)

saved_masks = []
stream_settings = { 'model': OBJECT_MODEL }

parser = argparse.ArgumentParser(description='Start a Video Stream for the given Camera Device')

parser.add_argument('device', type=str, help='A device path like e.g. /dev/video0')
parser.add_argument('camStream', type=str, help='Stream name (e.g. frontCam)')
parser.add_argument('--port', type=int, required=True, help='RTP port assigned by mediasoup')

args = parser.parse_args()

device = args.device
logger.info('Camera: %s', device)

# Check OpenCV video backends
logger.info('OpenCV backends — FFmpeg: %s, GStreamer: %s',
    'YES' if cv2.getBuildInformation().find('FFMPEG') > 0 else 'NO',
    'YES' if cv2.getBuildInformation().find('GStreamer') > 0 else 'NO')

def setVideoSource(device):
    global RESOLUTION_X
    global RESOLUTION_Y
    if device.startswith('http'):
        if device.startswith('https://youtu') or device.startswith('https://www.youtube.com'):
            # Get YouTube stream URL via yt_dlp, then use OpenCV
            try:
                video = get_youtube_video(device, RESOLUTION_Y)
                RESOLUTION_X = video.get("width", RESOLUTION_X)
                RESOLUTION_Y = video.get("height", RESOLUTION_Y)
                stream_url = video.get('url', device)
                logger.info('YouTube resolved to %dx%d, streaming with OpenCV', RESOLUTION_X, RESOLUTION_Y)
                cap = cv2.VideoCapture(stream_url)
            except Exception as err:
                logger.warning('YouTube resolution failed: %s, trying direct URL', err)
                cap = cv2.VideoCapture(device)
        else:
            # Direct HTTPS/HTTP streaming with OpenCV (GStreamer preferred)
            logger.info('Opening HTTP(S) video with OpenCV: %s', device)
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                logger.warning('FFmpeg backend failed, trying without explicit backend: %s', device)
                cap = cv2.VideoCapture(device)
            if cap.isOpened():
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width > 0 and actual_height > 0:
                    RESOLUTION_X = actual_width
                    RESOLUTION_Y = actual_height
                    logger.info('HTTP video opened: %dx%d', RESOLUTION_X, RESOLUTION_Y)
            else:
                logger.error('Failed to open HTTP video: %s', device)

    elif device.startswith('rtsp:'):
        cap = cv2.VideoCapture(device)
    elif device.startswith('demoVideo'):
        # Use FFmpeg backend — GStreamer can't reliably re-open m4v files
        # (fails with "unable to query duration of stream" on subsequent opens)
        cap = cv2.VideoCapture('/app/video/luggagebelt.m4v', cv2.CAP_FFMPEG)
        time.sleep(0.3)
        RESOLUTION_X = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        RESOLUTION_Y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info('Demo video opened: %dx%d (FFmpeg backend)', RESOLUTION_X, RESOLUTION_Y)
        # RESOLUTION_X = 1280
        # RESOLUTION_Y = 720
    else:
        device = int(device[-1])
        cap = cv2.VideoCapture(device)
    return cap

cap = setVideoSource(device)

logger.info('Resolution: %dx%d, source: %s', RESOLUTION_X, RESOLUTION_Y, device)
cap.set(3, RESOLUTION_X)
cap.set(4, RESOLUTION_Y)

model = getModel(OBJECT_MODEL)
current_model_name = OBJECT_MODEL
logger.info('Model native input: %s', model.get('native_input_wh', 'unknown'))
write_backend_status(args.camStream, model)

# print("CUDA available:", torch.cuda.is_available(), 'GPUs', torch.cuda.device_count())

async def main(_saved_masks):
    global cap
    try:
        fps_monitor = sv.FPSMonitor()
        start_time = time.time()
        start_time1 = time.time()
        start_time2 = time.time()
        
        # CPU encoding
        # outputFormat = " videoconvert ! vp8enc deadline=2 threads=4 keyframe-max-dist=10 ! video/x-vp8 ! rtpvp8pay pt=96"

        if torch.cuda.is_available():
            # Hardware h264 encoding on Jetson
            # profile=0 (Baseline) matches mediasoup's declared profile-level-id 42e01f
            # idrinterval=1 makes every I-frame a true IDR (NAL type 5) — mediasoup
            # only detects NAL type 5 as keyframes, NOT non-IDR I-frames (type 1).
            # h264parse normalises NAL unit boundaries for clean RTP packetisation.
            # config-interval=-1 inserts SPS/PPS with every IDR (not every N seconds).
            outputFormat = ("queue ! videoconvert ! nvvidconv ! queue"
                " ! video/x-raw(memory:NVMM), format=I420"
                " ! nvv4l2h264enc maxperf-enable=true preset-level=1 profile=0"
                "   insert-sps-pps=true insert-vui=true iframeinterval=10 idrinterval=1"
                " ! h264parse"
                " ! rtph264pay pt=96 ssrc=11111111 config-interval=-1")
        else:
            # CPU software encoding (x264)
            # key-int-max=15 ensures an IDR every ~0.5s at 30fps.
            # h264parse + config-interval=-1: same rationale as the HW path above.
            outputFormat = ("videoconvert ! video/x-raw, format=I420"
                " ! x264enc tune=zerolatency key-int-max=15 bframes=0 speed-preset=veryfast"
                " ! h264parse"
                " ! rtph264pay pt=96 ssrc=11111111 config-interval=-1")

        logger.info('Streaming %s to port %d (127.0.0.1)', args.camStream, args.port)

        if args.port < 1024 or args.port > 65535:
            logger.error('Port %d out of range', args.port)
            sys.exit(1)

        # Send to localhost since we share network namespace with mediasoup
        writerStream = "appsrc ! " + outputFormat + " ! udpsink host=127.0.0.1 port=" + str(args.port) + " sync=false async=false"
        logger.info('GStreamer pipeline: %s', writerStream)
        out = cv2.VideoWriter(writerStream, cv2.CAP_GSTREAMER, 0, FRAMERATE, (RESOLUTION_X, RESOLUTION_Y), True)
        if out.isOpened():
            logger.info('GStreamer VideoWriter opened successfully (hw encoder: %s)', 'yes' if torch.cuda.is_available() else 'no')
        else:
            logger.error('GStreamer VideoWriter FAILED to open — check GStreamer plugin availability')

        logger.info('Starting main video loop...')
        success = True
        frame = None
        loop_start = time.monotonic()
        frame_interval = 1.0 / FRAMERATE  # seconds per frame
        next_frame_time = time.monotonic()
        aggCounts = []
        slicer = initSliceInferer(model, stream_settings) if USE_SAHI else None

        while cap.isOpened():
            await sleep(0) # Give other task time to run, not a hack: https://superfastpython.com/what-is-asyncio-sleep-zero/#:~:text=You%20can%20force%20the%20current,before%20resuming%20the%20current%20task.
            elapsed_time = time.time() - start_time
            elapsed_time1 = time.time() - start_time1
            elapsed_time2 = time.time() - start_time2

            skip_inference = stream_settings.get('model', '') == 'none'

            # Dynamically update class filter from stream settings
            settings_class_list = stream_settings.get('classList', None)
            if settings_class_list is not None and len(settings_class_list) > 0:
                new_list = [int(c) for c in settings_class_list]
                if new_list != model_utils.CLASS_LIST:
                    model_utils.CLASS_LIST = new_list
                    CLASS_LIST[:] = new_list
                    logger.info('Updated CLASS_LIST from settings: %s', new_list)
            elif settings_class_list is not None and len(settings_class_list) == 0:
                # Empty list means "all classes" (no filter)
                if model_utils.CLASS_LIST != []:
                    model_utils.CLASS_LIST = []
                    CLASS_LIST[:] = []
                    logger.info('Reset CLASS_LIST to all classes (no filter)')

            # Dynamically update open-vocab class names from stream settings
            settings_class_names = stream_settings.get('classNames', None)
            if settings_class_names is not None and isinstance(settings_class_names, list):
                if hasattr(model_utils, 'CLASS_NAMES') and settings_class_names != model_utils.CLASS_NAMES:
                    model_utils.CLASS_NAMES = settings_class_names
                    logger.info('Updated CLASS_NAMES from settings: %s', settings_class_names)
                elif not hasattr(model_utils, 'CLASS_NAMES'):
                    model_utils.CLASS_NAMES = settings_class_names
                    logger.info('Set CLASS_NAMES from settings: %s', settings_class_names)

            # --- Fixed-timestep pacing (standard game-loop / video-player pattern) ---
            # Always wait for the next frame slot before reading
            wait_time = next_frame_time - time.monotonic()
            if wait_time > 0:
                await sleep(wait_time)

            # Read one frame
            success, frame = cap.read()

            # When inference is active and we fell behind, drop frames to catch up
            if success and not skip_inference:
                video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if video_ms > 0:
                    real_ms = (time.monotonic() - loop_start) * 1000
                    while real_ms >= video_ms and video_ms > 0:
                        success, frame = cap.read()
                        if not success: break
                        real_ms = (time.monotonic() - loop_start) * 1000
                        video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Schedule next frame relative to previous target (absorbs jitter)
            next_frame_time += frame_interval
            # Only hard-reset if we're very far behind (e.g. mode switch, video restart).
            # Small spikes (publishImage etc.) self-correct: the next wait_time will be
            # negative, so we skip the sleep and process immediately — catching up in 1-2 frames.
            if next_frame_time < time.monotonic() - 0.5:
                next_frame_time = time.monotonic() + frame_interval

            if not success:
                if device.startswith('demoVideo'):
                    # Demo video ended — re-open the file to loop.
                    # Seek (CAP_PROP_POS_FRAMES=0) is unreliable on m4v with FFmpeg;
                    # it can silently fail, causing a tight read-fail loop.
                    cap.release()
                    cap = cv2.VideoCapture('/app/video/luggagebelt.m4v', cv2.CAP_FFMPEG)
                    loop_start = time.monotonic()
                    next_frame_time = time.monotonic()
                    logger.info('Demo video looped to start')
                    continue
                logger.warning('Restarting video source: %s', device)
                cap.release()
                await sleep(1)
                cap = setVideoSource(device)
                loop_start = time.monotonic()
                next_frame_time = time.monotonic()
                if elapsed_time >= 2.0:
                    logger.error('Video frame could not be read from source: %s', device)
                    start_time = time.time()
                continue

            zoneCounts = {}
            lineCounts = {}
            detections = False
            fps_monitor.tick()

            if not skip_inference:
                use_sahi = stream_settings.get('useSahi', USE_SAHI)
                # Lazily create or clear slicer when the setting changes
                if use_sahi and slicer is None:
                    slicer = initSliceInferer(model, stream_settings)
                elif not use_sahi:
                    slicer = None

                if use_sahi and slicer is not None:
                    frame_buf = int(stream_settings.get('frameBuffer', FRAME_BUFFER))
                    low_x, low_y, high_x, high_y = get_extreme_points(_saved_masks, frame_buf)
                    # print('CROPPING', low_x, low_y, high_x, high_y)
                    frame_ = frame[low_y:high_y, low_x:high_x]
                    detections = slicer(frame_)
                    detections = move_detections(detections, low_x, low_y)
                    cv2.rectangle(frame, (low_x, low_y), (high_x, high_y), (255, 0, 0), 2)
                    # print('SLICER detections:', detections)
                else:
                    conf = float(stream_settings.get('confidence', model_utils.CONF))
                    detections = infer(frame, model, confidence=conf)

            start_time2 = time.time()

            if detections:
                frame, zoneCounts, lineCounts = processFrame(frame, detections, _saved_masks)
            elif skip_inference:
                # Still draw zone/line overlays even without inference
                frame, zoneCounts, lineCounts = processFrame(frame, empty_detections(), _saved_masks)

            # Draw FPS and Timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            overlay_text(frame, f'Timestamp: {current_time}', position=(10, 30))
            overlay_text(frame, f'FPS: {fps_monitor.fps:.2f}', position=(10, 60))

            # aggregate line counts for later publish
            for item in lineCounts:
                agg = next((agg for agg in aggCounts if agg["label"] == item["label"]), None)
                if not agg:
                    agg = {"label": item["label"]}
                    aggCounts.append(agg)
                agg["num_in"] = agg.get("num_in", 0) + item["num_in"] 
                agg["num_out"] = agg.get("num_out", 0) + item["num_out"] 

            # Publish data
            if elapsed_time1 >= 2.0:
                # print('FPS:', str(fps_monitor.fps))
                publishImage(frame)
                for item in zoneCounts:
                    publishClassCount(item["label"], item["count"])
                for item in aggCounts:
                    publishLineCount(item["label"], item["num_in"], item["num_out"])
                aggCounts = []

                start_time1 = time.time()

            if elapsed_time > 10.0:
                publishCameras()
                start_time = time.time()

            out.write(frame)


        logger.warning('Video source is not open: %s', device)
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            logger.debug('cv2.destroyAllWindows() not supported in headless mode')
    except Exception as e:
        logger.critical('Fatal error in video loop: %s', e, exc_info=True)
        sys.exit(1)

_publish_pool = ThreadPoolExecutor(max_workers=1)

def _encode_image(frame):
    """CPU-intensive JPEG + base64 encoding — runs in a thread to avoid blocking the event loop."""
    _, encoded_frame = cv2.imencode('.jpg', frame)
    return base64.b64encode(encoded_frame.tobytes()).decode('utf-8')

def publishImage(frame):
    async def _publish():
        loop = get_event_loop()
        base64_encoded_frame = await loop.run_in_executor(_publish_pool, _encode_image, frame.copy())
        now = datetime.now().astimezone().isoformat()
        await ironflock.publish_to_table('images', {"tsp": now, "image": 'data:image/jpeg;base64,' + base64_encoded_frame})
    get_event_loop().create_task(_publish())

def publishCameras():
    now = datetime.now().astimezone().isoformat()
    payload = {"tsp": now}
    payload["videolink"] = f"https://{DEVICE_KEY}-baggagetracker-1100.app.ironflock.com"
    payload["devicelink"] = DEVICE_URL
    get_event_loop().create_task(ironflock.publish_to_table('cameras', payload))

def publishClassCount(zone_name, result):
    now = datetime.now().astimezone().isoformat()
    payload = {
        "tsp": now,
        "zone_name": zone_name
        }

    for class_id in CLASS_LIST:
        payload[str(class_id)] = result.get(class_id, 0)

    get_event_loop().create_task(ironflock.publish_to_table('detections', payload))

def publishLineCount(line_name, num_in, num_out):
    now = datetime.now().astimezone().isoformat()
    payload = {
        "tsp": now,
        "line_name": line_name,
        "num_in": num_in,
        "num_out": num_out
    }

    get_event_loop().create_task(ironflock.publish_to_table('linecounts', payload))
    


if __name__ == "__main__":
    ENV = os.environ.get('ENV', '')

    if ENV == 'DEV':
        # Stub IronFlock for local dev — no WAMP connection needed
        class _StubIronFlock:
            async def publish_to_table(self, table, data):
                pass
        ironflock = _StubIronFlock()

        loop = get_event_loop()
        loop.create_task(watchMaskFile(saved_masks, args.camStream))
        loop.create_task(watchSettingsFile(stream_settings, args.camStream))
        loop.create_task(main(saved_masks))
        loop.run_forever()
    else:
        ironflock = IronFlock()
        get_event_loop().create_task(watchMaskFile(saved_masks, args.camStream))
        get_event_loop().create_task(watchSettingsFile(stream_settings, args.camStream))
        get_event_loop().create_task(main(saved_masks))
        ironflock.run()
