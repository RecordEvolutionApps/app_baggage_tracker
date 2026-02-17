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

from model_utils import getModel, processFrame, initSliceInferer, move_detections, get_extreme_points, infer, get_youtube_video, overlay_text, draw_sahi_grid, count_polygon_zone, count_detections, watchMaskFile, watchSettingsFile, empty_detections, FRAME_BUFFER, write_backend_status
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
parser.add_argument('--width', type=int, default=None, help='Requested capture width (overrides RESOLUTION_X)')
parser.add_argument('--height', type=int, default=None, help='Requested capture height (overrides RESOLUTION_Y)')

args = parser.parse_args()

# Per-stream resolution overrides (from --width/--height CLI args)
if args.width:
    RESOLUTION_X = args.width
if args.height:
    RESOLUTION_Y = args.height

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
            # yt_dlp resolves the YouTube page URL into a direct CDN stream URL.
            # We request video-only DASH streams (bestvideo) for the highest
            # resolution; muxed "best" is often only 360p on Shorts / newer videos.
            # Only cap resolution if the user explicitly set --height; otherwise
            # fetch the best available quality from YouTube.
            yt_height = args.height  # None when no explicit override was given
            try:
                video = get_youtube_video(device, yt_height)
                stream_url = video.get('url', device)
                if video.get('width'):
                    RESOLUTION_X = video['width']
                if video.get('height'):
                    RESOLUTION_Y = video['height']
                logger.info('YouTube resolved to %dx%d, opening stream...', RESOLUTION_X, RESOLUTION_Y)
                # Force FFmpeg backend — GStreamer mishandles googlevideo.com URLs
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.info('FFmpeg backend failed, trying auto backend')
                    cap = cv2.VideoCapture(stream_url)
                if cap.isOpened():
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info('YouTube opened: %dx%d', actual_w, actual_h)
                else:
                    logger.error('YouTube VideoCapture FAILED to open stream URL')
            except Exception as err:
                logger.warning('YouTube yt-dlp failed: %s — trying direct URL', err)
                cap = cv2.VideoCapture(device)
        else:
            # Direct HTTPS/HTTP streaming — try GStreamer HW decode on Jetson
            logger.info('Opening HTTP(S) video: %s', device)
            cap = None
            if torch.cuda.is_available():
                gst_pipe = (
                    f"souphttpsrc location={device} ! queue"
                    " ! h264parse ! nvv4l2decoder"
                    " ! nvvidconv ! video/x-raw, format=BGRx"
                    " ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
                )
                logger.info('Trying GStreamer HW decode for HTTP: %s', gst_pipe)
                cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
                if not cap.isOpened():
                    logger.warning('GStreamer HTTP HW decode failed, falling back')
                    cap = None

            if cap is None:
                # Explicitly request FFmpeg so OpenCV doesn't fall back to
                # CAP_IMAGES (which misinterprets URLs as image-sequence paths).
                cap = cv2.VideoCapture(device, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.warning('FFmpeg backend failed, trying auto backend: %s', device)
                    cap = cv2.VideoCapture(device, cv2.CAP_ANY)

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
        # Use GStreamer pipeline with HW decoder on Jetson for zero-copy GPU decode.
        # Falls back to FFmpeg backend if GStreamer is unavailable.
        cap = None
        if torch.cuda.is_available():
            gst_pipe = (
                f"rtspsrc location={device} latency=200 ! queue"
                " ! rtph264depay ! h264parse ! nvv4l2decoder"
                " ! nvvidconv ! video/x-raw, format=BGRx"
                " ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
            )
            logger.info('RTSP via GStreamer HW decode: %s', gst_pipe)
            cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                logger.warning('GStreamer RTSP pipeline failed, falling back to FFmpeg')
                cap = None

        if cap is None:
            # Set RTSP connection timeout (10s) via FFmpeg options to avoid
            # blocking indefinitely on unreachable / slow RTSP servers.
            # stimeout is in microseconds for FFmpeg's RTSP demuxer.
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;10000000'
            logger.info('Opening RTSP via FFmpeg: %s', device)
            cap = cv2.VideoCapture(device, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.warning('FFmpeg RTSP failed, trying auto backend: %s', device)
                cap = cv2.VideoCapture(device, cv2.CAP_ANY)

        # Read actual resolution from RTSP stream
        if cap.isOpened():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w > 0 and actual_h > 0:
                RESOLUTION_X = actual_w
                RESOLUTION_Y = actual_h
                logger.info('RTSP stream opened: %dx%d', RESOLUTION_X, RESOLUTION_Y)
        else:
            logger.error('Failed to open RTSP stream: %s', device)
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
        # USB camera: expect /dev/videoN path
        if device.startswith('/dev/video'):
            dev_index = int(device.replace('/dev/video', ''))
        elif device[-1].isdigit():
            dev_index = int(device[-1])
        else:
            logger.error('Unrecognised device string: %s', device)
            sys.exit(1)
        cap = cv2.VideoCapture(dev_index)
        # Read actual resolution from USB camera after setting requested resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_X)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_Y)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w > 0 and actual_h > 0:
            RESOLUTION_X = actual_w
            RESOLUTION_Y = actual_h
            logger.info('USB camera actual resolution: %dx%d', RESOLUTION_X, RESOLUTION_Y)
    return cap

cap = setVideoSource(device)

# If the source failed to open, retry in a loop before giving up
_open_retries = 0
while not cap.isOpened() and _open_retries < 12:
    _open_retries += 1
    logger.warning('Source not open, retrying in 5s (%d/12)...', _open_retries)
    time.sleep(5)
    cap = setVideoSource(device)

if not cap.isOpened():
    logger.error('Failed to open video source after retries: %s', device)

logger.info('Resolution: %dx%d, source: %s', RESOLUTION_X, RESOLUTION_Y, device)
cap.set(3, RESOLUTION_X)
cap.set(4, RESOLUTION_Y)

# Write the actual resolved resolution so the web backend / frontend can read it
_status_dir = os.path.join('/data', 'status')
os.makedirs(_status_dir, exist_ok=True)
_resolution_file = os.path.join(_status_dir, f'{args.camStream}.resolution.json')
with open(_resolution_file, 'w') as _rf:
    import json as _json
    _json.dump({'width': RESOLUTION_X, 'height': RESOLUTION_Y}, _rf)
logger.info('Wrote resolution status: %dx%d → %s', RESOLUTION_X, RESOLUTION_Y, _resolution_file)

model = getModel(OBJECT_MODEL)
current_model_name = OBJECT_MODEL
logger.info('Model native input: %s', model.get('native_input_wh', 'unknown'))
write_backend_status(args.camStream, model)

# print("CUDA available:", torch.cuda.is_available(), 'GPUs', torch.cuda.device_count())

async def main(_saved_masks):
    global cap, model, current_model_name
    try:
        fps_monitor = sv.FPSMonitor()
        start_time = time.time()
        start_time1 = time.time()
        
        if torch.cuda.is_available():
            # Hardware h264 encoding on Jetson
            # profile=0 (Baseline) matches mediasoup's declared profile-level-id 42e01f
            # idrinterval=1 makes every I-frame a true IDR (NAL type 5) — mediasoup
            # only detects NAL type 5 as keyframes, NOT non-IDR I-frames (type 1).
            # h264parse normalises NAL unit boundaries for clean RTP packetisation.
            # config-interval=-1 inserts SPS/PPS with every IDR (not every N seconds).
            # NOTE: videoconvert (CPU) is required because nvvidconv cannot accept
            # BGR directly from OpenCV's appsrc — it only handles I420/NV12/RGBA.
            # videoconvert converts BGR→I420, then nvvidconv uploads to NVMM for HW encode.
            # Bitrate scales with pixel count (base: 8 Mbps @ 1280x720).
            base_bps = 8_000_000
            base_pixels = 1280 * 720
            actual_pixels = RESOLUTION_X * RESOLUTION_Y
            hw_bitrate = max(4_000_000, int(base_bps * actual_pixels / base_pixels))
            logger.info('HW encoder bitrate: %d bps (%dx%d)', hw_bitrate, RESOLUTION_X, RESOLUTION_Y)
            outputFormat = ("queue ! videoconvert ! video/x-raw, format=I420"
                " ! nvvidconv"
                " ! video/x-raw(memory:NVMM), format=I420"
                " ! nvv4l2h264enc maxperf-enable=true preset-level=1 profile=0"
                "   insert-sps-pps=true insert-vui=true iframeinterval=10 idrinterval=1"
                f"   control-rate=1 bitrate={hw_bitrate}"
                " ! h264parse"
                " ! rtph264pay pt=96 ssrc=11111111 config-interval=-1")
        else:
            # CPU software encoding (x264)
            # Bitrate scales with pixel count to maintain quality across
            # different resolutions (USB 640x480 vs YouTube 1080x1920).
            # Base: 8 Mbps for 1280x720 (~921k pixels), scaled linearly.
            base_bitrate = 8000  # kbps at 1280x720
            base_pixels = 1280 * 720
            actual_pixels = RESOLUTION_X * RESOLUTION_Y
            scaled_bitrate = max(4000, int(base_bitrate * actual_pixels / base_pixels))
            logger.info('x264 bitrate: %d kbps (%dx%d = %d pixels)',
                        scaled_bitrate, RESOLUTION_X, RESOLUTION_Y, actual_pixels)
            # key-int-max=1: every frame is an IDR keyframe — at low actual
            # fps the overhead is negligible and the browser can start
            # decoding instantly on the very first frame after (re)connecting.
            # h264parse + config-interval=-1: same rationale as the HW path above.
            outputFormat = ("videoconvert ! video/x-raw, format=I420"
                f" ! x264enc tune=zerolatency bitrate={scaled_bitrate}"
                "   key-int-max=1 bframes=0 speed-preset=faster"
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

        # SAHI slicer (created lazily)
        slicer = initSliceInferer(model, stream_settings) if USE_SAHI else None

        logger.info('Starting main video loop...')
        success = True
        frame = None
        loop_start = time.monotonic()
        frame_interval = 1.0 / FRAMERATE  # seconds per frame
        next_frame_time = time.monotonic()
        aggCounts = []
        zoneCounts = []
        lineCounts = []

        while cap.isOpened():
            await sleep(0) # Give other task time to run
            elapsed_time = time.time() - start_time
            elapsed_time1 = time.time() - start_time1

            sahi_rect = None
            sahi_slice_wh = None
            sahi_overlap_wh = None
            use_sahi = False
            nms_iou = model_utils.NMS_IOU
            sahi_iou = model_utils.SAHI_IOU
            overlap_ratio = 0.2

            skip_inference = stream_settings.get('model', '') == 'none'

            # ── Hot-reload model when the user picks a different one ──
            new_model_name = stream_settings.get('model', current_model_name)
            if new_model_name and new_model_name != 'none' and new_model_name != current_model_name:
                logger.info('Model changed: %s -> %s — reloading...', current_model_name, new_model_name)
                try:
                    model = getModel(new_model_name)
                    current_model_name = new_model_name
                    slicer = None  # force SAHI re-init with new model
                    logger.info('Model reloaded: %s, native input: %s',
                                current_model_name, model.get('native_input_wh', 'unknown'))
                    write_backend_status(args.camStream, model)
                except Exception as e:
                    logger.error('Failed to reload model %s: %s', new_model_name, e, exc_info=True)

            # Dynamically update class filter from stream settings
            settings_class_list = stream_settings.get('classList', None)
            if settings_class_list is not None and len(settings_class_list) > 0:
                new_list = [int(c) for c in settings_class_list]
                if new_list != model_utils.CLASS_LIST:
                    model_utils.CLASS_LIST = new_list
                    CLASS_LIST[:] = new_list
                    logger.info('Updated CLASS_LIST from settings: %s', new_list)
            elif settings_class_list is not None and len(settings_class_list) == 0:
                if model_utils.CLASS_LIST != []:
                    model_utils.CLASS_LIST = []
                    CLASS_LIST[:] = []
                    logger.info('Reset CLASS_LIST to all classes (no filter)')

            # Dynamically update open-vocab class names from stream settings
            settings_class_names = stream_settings.get('classNames', None)
            if settings_class_names is not None and isinstance(settings_class_names, list):
                if len(settings_class_names) > 0:
                    if hasattr(model_utils, 'CLASS_NAMES') and settings_class_names != model_utils.CLASS_NAMES:
                        model_utils.CLASS_NAMES = settings_class_names
                        logger.info('Updated CLASS_NAMES from settings: %s', settings_class_names)
                    elif not hasattr(model_utils, 'CLASS_NAMES'):
                        model_utils.CLASS_NAMES = settings_class_names
                        logger.info('Set CLASS_NAMES from settings: %s', settings_class_names)

            # --- Fixed-timestep pacing ---
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
            if next_frame_time < time.monotonic() - 0.5:
                next_frame_time = time.monotonic() + frame_interval

            if not success:
                if device.startswith('demoVideo'):
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

            fps_monitor.tick()

            # ── Run inference on this frame ──────────────────────────────
            # Single-threaded: read → infer → annotate → write.
            # Bounding boxes are always computed on the exact frame shown.
            if skip_inference:
                detections = empty_detections()
            else:
                use_sahi = stream_settings.get('useSahi', USE_SAHI)
                nms_iou = float(stream_settings.get('nmsIou', model_utils.NMS_IOU))
                sahi_iou = float(stream_settings.get('sahiIou', model_utils.SAHI_IOU))
                overlap_ratio = float(stream_settings.get('overlapRatio', 0.2))
                try:
                    # Lazily create or clear slicer when the setting changes
                    if use_sahi and (slicer is None or getattr(slicer, '_sahi_iou', None) != sahi_iou or getattr(slicer, '_sahi_overlap_ratio', None) != overlap_ratio):
                        slicer = initSliceInferer(model, stream_settings)
                    elif not use_sahi:
                        slicer = None

                    if use_sahi and slicer is not None:
                        frame_buf = int(stream_settings.get('frameBuffer', FRAME_BUFFER))
                        low_x, low_y, high_x, high_y = get_extreme_points(_saved_masks, frame_buf)
                        sahi_rect = (low_x, low_y, high_x, high_y)
                        if hasattr(slicer, '_sahi_slice_wh'):
                            sahi_slice_wh = slicer._sahi_slice_wh
                        if hasattr(slicer, '_sahi_overlap_wh'):
                            sahi_overlap_wh = slicer._sahi_overlap_wh
                        crop_w, crop_h = high_x - low_x, high_y - low_y
                        logger.debug('[SAHI] crop=(%d,%d)-(%d,%d) size=%dx%d, masks=%d',
                                    low_x, low_y, high_x, high_y, crop_w, crop_h, len(_saved_masks))
                        frame_ = frame[low_y:high_y, low_x:high_x]
                        t_sahi = time.monotonic()
                        if hasattr(slicer, '_sahi_slice_count'):
                            slicer._sahi_slice_count[0] = 0
                        detections = slicer(frame_)
                        dt_sahi = time.monotonic() - t_sahi
                        det_count = len(detections) if detections else 0
                        n_slices = slicer._sahi_slice_count[0] if hasattr(slicer, '_sahi_slice_count') else '?'
                        logger.debug('[SAHI] %s slices in %.1fms, %d detections (pre-offset)',
                                    n_slices, dt_sahi * 1000, det_count)
                        detections = move_detections(detections, low_x, low_y)
                    else:
                        conf = float(stream_settings.get('confidence', model_utils.CONF))
                        detections = infer(frame, model, confidence=conf, iou=nms_iou)

                    if not detections:
                        detections = empty_detections()
                except Exception as e:
                    logger.error('Inference error: %s', e, exc_info=True)
                    detections = empty_detections()

            # ── Overlay detections & annotate ───────────────────────────

            frame, zoneCounts, lineCounts = processFrame(frame, detections, _saved_masks, stream_settings)

            if use_sahi and sahi_rect and sahi_slice_wh and sahi_overlap_wh:
                frame = draw_sahi_grid(frame, sahi_rect, sahi_slice_wh, sahi_overlap_wh)

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
    """WebP + base64 encoding — ~30% smaller than JPEG at equivalent quality."""
    _, encoded_frame = cv2.imencode('.webp', frame, [cv2.IMWRITE_WEBP_QUALITY, 80])
    return base64.b64encode(encoded_frame.tobytes()).decode('utf-8')

def publishImage(frame):
    async def _publish():
        loop = get_event_loop()
        base64_encoded_frame = await loop.run_in_executor(_publish_pool, _encode_image, frame.copy())
        now = datetime.now().astimezone().isoformat()
        await ironflock.publish_to_table('images', {"tsp": now, "image": 'data:image/webp;base64,' + base64_encoded_frame})
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
