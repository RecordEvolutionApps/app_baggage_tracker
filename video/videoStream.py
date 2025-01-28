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

from pprint import pprint
import base64
import torch
import functools

import traceback

from model_utils import getModel, processFrame, initSliceInferer, move_detections, get_extreme_points, infer, get_youtube_video, overlay_text, count_polygon_zone, count_detections, readMasksFromStdin

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

parser = argparse.ArgumentParser(description='Start a Video Stream for the given Camera Device')

parser.add_argument('device', type=str, help='A device path like e.g. /dev/video0')
parser.add_argument('camStream', type=str, help='One of frontCam, leftCam, rightCam, backCam')

args = parser.parse_args()

portMap = {"frontCam": 5004,
           "leftCam": 5005,
           "rightCam": 5006,
           "backCam": 5007}

device = args.device
print('CAMERA USED:' + device)

def setVideoSource(device):
    global RESOLUTION_X
    global RESOLUTION_Y
    if device.startswith('http'):
        if device.startswith('https://youtu') or device.startswith('https://www.youtube.com'):
            video = get_youtube_video(device, RESOLUTION_Y)
            RESOLUTION_X = video["width"]
            RESOLUTION_Y = video["height"]
            device = video.get('url')
        cap = cv2.VideoCapture(device)
        # device = f"uridecay url='{device}' ! nvenc-hwaccel=true ! nvdec_hwaccel ! videoconvert ! appsink"
        # cap = cv2.VideoCapture(device, cv2.CAP_GSTREAMER)

    elif device.startswith('rtsp:'):
        cap = cv2.VideoCapture(device)
    elif device.startswith('demoVideo'):
        cap = cv2.VideoCapture('/app/video/luggagebelt.m4v')
        RESOLUTION_X = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        RESOLUTION_Y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

MODEL_RESX = (RESOLUTION_X // 32) * 32 # must be multiple of max stride 32
MODEL_RESY = (RESOLUTION_Y // 32) * 32
if USE_SAHI:
    model = getModel(OBJECT_MODEL)
else:
    model = getModel(OBJECT_MODEL, MODEL_RESX, MODEL_RESY)

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
            outputFormat = "queue ! videoconvert ! nvvidconv ! queue ! video/x-raw(memory:NVMM), format=I420 ! nvv4l2h264enc maxperf-enable=true preset-level=1 insert-sps-pps=true insert-vui=true iframeinterval=10 idrinterval=10 ! rtph264pay pt=96 config-interval=1"
        else:
            outputFormat = "videoconvert ! video/x-raw, format=I420 ! x264enc tune=zerolatency ! rtph264pay pt=96 config-interval=1"

        writerStream = "appsrc ! " + outputFormat + " ! udpsink host=janus port=" + str(portMap[args.camStream]) + " sync=false async=false"
        # writerStream = "appsrc ! " + outputFormat + " ! webrtcsink name=webrtcsink stun-server=stun.l.google.com:19302 signaling-server=ws://localhost:1200"

        print('-------------CREATING WRITE STREAM:', writerStream)
        out = cv2.VideoWriter(writerStream, cv2.CAP_GSTREAMER, 0, FRAMERATE, (RESOLUTION_X, RESOLUTION_Y), True)

        print('starting main video loop...')
        success = True
        frame = None
        start = time.time()
        real_ms = 0
        video_ms = 0
        aggCounts = []
        if USE_SAHI: slicer = initSliceInferer(model)

        while cap.isOpened():
            elapsed_time = time.time() - start_time
            elapsed_time1 = time.time() - start_time1
            elapsed_time2 = time.time() - start_time2

            real_ms = (time.time() - start) * 1000
            while real_ms >= video_ms:
                success, frame = cap.read()
                if not success: break
                real_ms = (time.time() - start) * 1000
                video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                # print('real > vs', int(real_ms), int(video_ms), real_ms > video_ms)
            video_ms = 0

            if not success:
                print("################ RESTART VIDEO ####################")
                await sleep(1)
                cap = setVideoSource(device)
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start = time.time()
                if elapsed_time >= 2.0:
                    print('Video Frame could not be read from source', device)
                    start_time = time.time()
                continue

            zoneCounts = {}
            lineCounts = {}
            detections = False
            fps_monitor.tick()

            if USE_SAHI:
                low_x, low_y, high_x, high_y = get_extreme_points(_saved_masks)
                # print('CROPPING', low_x, low_y, high_x, high_y)
                frame_ = frame[low_y:high_y, low_x:high_x]
                detections = slicer(frame_)
                detections = move_detections(detections, low_x, low_y)
                cv2.rectangle(frame, (low_x, low_y), (high_x, high_y), (255, 0, 0), 2)
                # print('SLICER detections:', detections)
            else:
                detections = infer(frame, model, MODEL_RESX, MODEL_RESY)

            start_time2 = time.time()

            if detections:
                frame, zoneCounts, lineCounts = processFrame(frame, detections, _saved_masks)

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

            await sleep(0) # Give other task time to run, not a hack: https://superfastpython.com/what-is-asyncio-sleep-zero/#:~:text=You%20can%20force%20the%20current,before%20resuming%20the%20current%20task.

        print('WARNING: Video source is not open', device)
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f'############################# Error in video loop: {str(e)}')
        traceback.print_exc()
        import sys
        sys.exit(1)

def publishImage(frame):
    _, encoded_frame = cv2.imencode('.jpg', frame)
    base64_encoded_frame = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')
    now = datetime.now().astimezone().isoformat()

    get_event_loop().create_task(rw.publish_to_table('images', {"tsp": now, "image": 'data:image/jpeg;base64,' + base64_encoded_frame}))

def publishCameras():
    now = datetime.now().astimezone().isoformat()
    payload = {"tsp": now}
    payload["videolink"] = f"https://{DEVICE_KEY}-baggagetracker-1100.app.ironflock.com"
    payload["devicelink"] = DEVICE_URL
    get_event_loop().create_task(rw.publish_to_table('cameras', payload))

def publishClassCount(zone_name, result):
    now = datetime.now().astimezone().isoformat()
    payload = {
        "tsp": now,
        "zone_name": zone_name
        }

    for class_id in CLASS_LIST:
        payload[class_id_topic[str(class_id)]] = result.get(class_id, 0)

    print(payload)

    get_event_loop().create_task(rw.publish_to_table('detections', payload))

def publishLineCount(line_name, num_in, num_out):
    now = datetime.now().astimezone().isoformat()
    payload = {
        "tsp": now,
        "line_name": line_name,
        "num_in": num_in,
        "num_out": num_out
    }

    print(payload)
    get_event_loop().create_task(rw.publish_to_table('linecounts', payload))


rw = IronFlock()

if __name__ == "__main__":
    # run the main coroutine
    task1 = get_event_loop().create_task(readMasksFromStdin(saved_masks))
    task2 = get_event_loop().create_task(main(saved_masks))

    # run the reswarm component
    rw.run()
