import sys
import json
from asyncio import get_event_loop, sleep
import supervision as sv
import cv2
import os
import argparse
import time
from datetime import datetime
from reswarm import Reswarm

from pprint import pprint
import base64
import torch
import functools

import traceback

from model_utils import getModel, processFrame, get_youtube_video, overlay_text, count_polygon_zone, count_detections, prepMasks, readMasksFromStdin

print = functools.partial(print, flush=True)

with open('/app/video/coco_classes.json', 'r') as f:
  class_id_topic = json.load(f)

OBJECT_MODEL = os.environ.get('OBJECT_MODEL')
RESOLUTION_X = int(os.environ.get('RESOLUTION_X', 640))
RESOLUTION_Y = int(os.environ.get('RESOLUTION_Y', 480))
FRAMERATE = int(os.environ.get('FRAMERATE', 30))

DEVICE_KEY = os.environ.get('DEVICE_KEY')
DEVICE_URL = os.environ.get('DEVICE_URL')
CONF = float(os.environ.get('CONF', '0.1'))
IOU = float(os.environ.get('IOU', '0.8'))
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

if device.startswith('http'):
    if device.startswith('https://youtu'):
        video = get_youtube_video(device, RESOLUTION_Y)
        RESOLUTION_X = video["width"]
        RESOLUTION_Y = video["height"]
        device = video.get('url')
    cap = cv2.VideoCapture(device)
    # device = f"uridecay url='{device}' ! nvenc-hwaccel=true ! nvdec_hwaccel ! videoconvert ! appsink"
    # cap = cv2.VideoCapture(device, cv2.CAP_GSTREAMER)

elif device.startswith('rtsp:'):
    cap = cv2.VideoCapture(device)
else:
    device = int(device[-1])
    cap = cv2.VideoCapture(device)

print('RESOLUTION', RESOLUTION_X, RESOLUTION_Y, device)
cap.set(3, RESOLUTION_X)
cap.set(4, RESOLUTION_Y)

MODEL_RESX = (RESOLUTION_X // 32) * 32 # must be multiple of max stride 32
MODEL_RESY = (RESOLUTION_Y // 32) * 32
model = getModel(OBJECT_MODEL, MODEL_RESX, MODEL_RESY)

# print("CUDA available:", torch.cuda.is_available(), 'GPUs', torch.cuda.device_count())

async def main(_saved_masks):
    try:
        print('starting main video loop...')
        fps_monitor = sv.FPSMonitor()
        start_time = time.time()
        start_time1 = time.time()
        start_time2 = time.time()

        prev_frame_time = time.time()
        frame_skip_threshold = 1.0 / FRAMERATE  # Maximum allowed processing time per frame

        outputFormat = " videoconvert ! vp8enc deadline=2 threads=4 keyframe-max-dist=6 ! video/x-vp8 ! rtpvp8pay pt=96"
        # outputFormat = "nvvidconv ! nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true insert-vui=true ! h264parse ! rtph264pay"

        writerStream = "appsrc do-timestamp=true ! " + outputFormat + " ! udpsink host=janus port=" + str(portMap[args.camStream])
        # print(writerStream)

        out = cv2.VideoWriter(writerStream, 0, FRAMERATE, (RESOLUTION_X, RESOLUTION_Y))

        while cap.isOpened():
            elapsed_time = time.time() - start_time
            elapsed_time1 = time.time() - start_time1
            elapsed_time2 = time.time() - start_time2

            success, frame = cap.read()
            if not success:
                continue

            curr_frame_time = time.time()

            # Check if processing time for previous frame exceeded threshold
            if curr_frame_time - prev_frame_time > frame_skip_threshold:
                # print("Skipping frame!", curr_frame_time - prev_frame_time)
                prev_frame_time = curr_frame_time
                continue

            # print("process frame", curr_frame_time - prev_frame_time)
            # Update previous frame time for next iteration
            prev_frame_time = curr_frame_time
            
            counts = {}
            results = []
            fps_monitor.tick()
            results = model(frame, imgsz=(MODEL_RESY, MODEL_RESX), conf=CONF, iou=IOU, verbose=False, classes=CLASS_LIST)
            start_time2 = time.time()
            
            if len(results) > 0:
                frame, counts = processFrame(frame, results, CLASS_LIST, _saved_masks)

            # Draw FPS and Timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            overlay_text(frame, f'Timestamp: {current_time}', position=(10, 30))
            overlay_text(frame, f'FPS: {fps_monitor.fps:.2f}', position=(10, 60))

            # Publish data
            if elapsed_time1 >= 2.0:
                for item in counts:
                    publishImage(frame)
                    publishClassCount(item["count"], item["label"])
                    start_time1 = time.time()
            
            if elapsed_time > 10.0:
                publishCameras()
                start_time = time.time()

            out.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break

            await sleep(0) # Give other ask time to run, not a hack: https://superfastpython.com/what-is-asyncio-sleep-zero/#:~:text=You%20can%20force%20the%20current,before%20resuming%20the%20current%20task.

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
    payload["videolink"] = f"https://{DEVICE_KEY}-traffic-1100.app.record-evolution.com"
    payload["devicelink"] = DEVICE_URL
    get_event_loop().create_task(rw.publish_to_table('cameras', payload))

def publishClassCount(result, zone_name):
    now = datetime.now().astimezone().isoformat()
    payload = {"tsp": now}
    payload["zone_name"] = zone_name

    for class_id in CLASS_LIST:
        payload[class_id_topic[str(class_id)]] = result.get(class_id, 0)

    print(payload)

    get_event_loop().create_task(rw.publish_to_table('detections', payload))

rw = Reswarm()

if __name__ == "__main__":
    # run the main coroutine
    task1 = get_event_loop().create_task(readMasksFromStdin(saved_masks))
    task2 = get_event_loop().create_task(main(saved_masks))

    # run the reswarm component
    rw.run()
