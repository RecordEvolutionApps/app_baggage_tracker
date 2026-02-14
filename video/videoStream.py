import sys
import json
from asyncio import get_event_loop, sleep
import supervision as sv
import cv2
import os
import argparse
import time
from datetime import datetime
from ironflock import IronFlock
from concurrent.futures import ThreadPoolExecutor

from pprint import pprint
import base64
import torch
import functools

import traceback

from model_utils import getModel, processFrame, initSliceInferer, move_detections, get_extreme_points, infer, get_youtube_video, overlay_text, count_polygon_zone, count_detections, watchMaskFile, watchSettingsFile, empty_detections, FRAME_BUFFER
import model_utils  # for updating CLASS_LIST dynamically

print = functools.partial(print, flush=True)

with open('/app/video/coco_classes.json', 'r') as f:
  class_id_topic = json.load(f)

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
    print('Invalid Class list given', CLASS_LIST)
    CLASS_LIST = []

if len(CLASS_LIST) <= 1:
    CLASS_LIST = list(class_id_topic.keys())
    CLASS_LIST = [int(item) for item in CLASS_LIST]

print('########## USING CLASS LIST:', CLASS_LIST)

saved_masks = []
stream_settings = { 'model': OBJECT_MODEL }

parser = argparse.ArgumentParser(description='Start a Video Stream for the given Camera Device')

parser.add_argument('device', type=str, help='A device path like e.g. /dev/video0')
parser.add_argument('camStream', type=str, help='Stream name (e.g. frontCam)')
parser.add_argument('--port', type=int, required=True, help='RTP port assigned by mediasoup')

args = parser.parse_args()

device = args.device
print('CAMERA USED:' + device)

# Check OpenCV video backends
print('OpenCV build info:')
print('  FFmpeg:', 'YES' if cv2.getBuildInformation().find('FFMPEG') > 0 else 'NO')
print('  GStreamer:', 'YES' if cv2.getBuildInformation().find('GStreamer') > 0 else 'NO')

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
                print(f'YouTube resolved to {RESOLUTION_X}x{RESOLUTION_Y}, streaming with OpenCV')
                cap = cv2.VideoCapture(stream_url)
            except Exception as err:
                print(f'YouTube resolution failed: {err}, trying direct URL')
                cap = cv2.VideoCapture(device)
        else:
            # Direct HTTPS/HTTP streaming with OpenCV (GStreamer preferred)
            print(f'Opening HTTP(S) video with OpenCV: {device}')
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                print(f'FFmpeg backend failed, trying without explicit backend: {device}')
                cap = cv2.VideoCapture(device)
            if cap.isOpened():
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width > 0 and actual_height > 0:
                    RESOLUTION_X = actual_width
                    RESOLUTION_Y = actual_height
                    print(f'HTTP video opened successfully: {RESOLUTION_X}x{RESOLUTION_Y}')
            else:
                print(f'Failed to open HTTP video: {device}')

    elif device.startswith('rtsp:'):
        cap = cv2.VideoCapture(device)
    elif device.startswith('demoVideo'):
        # Use default backend (GStreamer) for best performance
        cap = cv2.VideoCapture('/app/video/luggagebelt.m4v')
        # Give GStreamer pipeline time to start (prevents immediate read failures)
        time.sleep(0.5)
        RESOLUTION_X = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        RESOLUTION_Y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Demo video opened: {RESOLUTION_X}x{RESOLUTION_Y}')
        # RESOLUTION_X = 1280
        # RESOLUTION_Y = 720
    else:
        device = int(device[-1])
        cap = cv2.VideoCapture(device)
    return cap

cap = setVideoSource(device)

print('RESOLUTION', RESOLUTION_X, RESOLUTION_Y, device)
cap.set(3, RESOLUTION_X)
cap.set(4, RESOLUTION_Y)

model = getModel(OBJECT_MODEL)
current_model_name = OBJECT_MODEL
print(f'Model native input: {model.get("native_input_wh", "unknown")}')

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
            # Hardware h264 encoding of jetson
            outputFormat = "queue ! videoconvert ! nvvidconv ! queue ! video/x-raw(memory:NVMM), format=I420 ! nvv4l2h264enc maxperf-enable=true preset-level=1 insert-sps-pps=true insert-vui=true iframeinterval=10 idrinterval=10 ! rtph264pay pt=96 ssrc=11111111 config-interval=1"
        else:
            # veryfast preset: good balance of speed/quality (ultrafast was too low quality)
            # NOTE: Do NOT use intra-refresh=true — it replaces real IDR keyframes with gradual
            # refresh, which prevents the browser H264 decoder from starting after a reconnect.
            # key-int-max=15 ensures an IDR every ~0.5s at 30fps for fast recovery on browser refresh.
            outputFormat = "videoconvert ! video/x-raw, format=I420 ! x264enc tune=zerolatency key-int-max=15 bframes=0 speed-preset=veryfast ! rtph264pay pt=96 ssrc=11111111 config-interval=1"

        print(f"Streaming to {args.camStream} on port {args.port} (127.0.0.1)")

        if args.port not in range(10000, 10101):  # Just check if valid
            print("Port error")
            sys.exit(1)

        # Send to localhost since we share network namespace with mediasoup
        writerStream = "appsrc ! " + outputFormat + " ! udpsink host=127.0.0.1 port=" + str(args.port) + " sync=false async=false"
        print('-------------CREATING WRITE STREAM:', writerStream)
        out = cv2.VideoWriter(writerStream, cv2.CAP_GSTREAMER, 0, FRAMERATE, (RESOLUTION_X, RESOLUTION_Y), True)

        print('starting main video loop...')
        success = True
        frame = None
        loop_start = time.monotonic()
        frame_interval = 1.0 / FRAMERATE  # seconds per frame
        next_frame_time = time.monotonic()
        aggCounts = []
        slicer = initSliceInferer(model) if USE_SAHI else None

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
                    print('Updated CLASS_LIST from settings:', new_list)
            elif settings_class_list is not None and len(settings_class_list) == 0:
                # Empty list means "all classes"
                all_classes = [int(k) for k in class_id_topic.keys()]
                if model_utils.CLASS_LIST != all_classes:
                    model_utils.CLASS_LIST = all_classes
                    CLASS_LIST[:] = all_classes
                    print('Reset CLASS_LIST to all classes')

            # Dynamically update open-vocab class names from stream settings
            settings_class_names = stream_settings.get('classNames', None)
            if settings_class_names is not None and isinstance(settings_class_names, list):
                if hasattr(model_utils, 'CLASS_NAMES') and settings_class_names != model_utils.CLASS_NAMES:
                    model_utils.CLASS_NAMES = settings_class_names
                    print('Updated CLASS_NAMES from settings:', settings_class_names)
                elif not hasattr(model_utils, 'CLASS_NAMES'):
                    model_utils.CLASS_NAMES = settings_class_names
                    print('Set CLASS_NAMES from settings:', settings_class_names)

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
                print("################ RESTART VIDEO ####################")
                await sleep(1)
                cap = setVideoSource(device)
                loop_start = time.monotonic()
                next_frame_time = time.monotonic()
                if elapsed_time >= 2.0:
                    print('Video Frame could not be read from source', device)
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
                    slicer = initSliceInferer(model)
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
                    detections = infer(frame, model)

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


        print('WARNING: Video source is not open', device)
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            print('cv2.destroyAllWindows() not supported in headless mode')
    except Exception as e:
        print(f'############################# Error in video loop: {str(e)}')
        traceback.print_exc()
        import sys
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
        payload[class_id_topic[str(class_id)]] = result.get(class_id, 0)

    print(payload)

    get_event_loop().create_task(ironflock.publish_to_table('detections', payload))

def publishLineCount(line_name, num_in, num_out):
    now = datetime.now().astimezone().isoformat()
    payload = {
        "tsp": now,
        "line_name": line_name,
        "num_in": num_in,
        "num_out": num_out
    }

    print(payload)
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
