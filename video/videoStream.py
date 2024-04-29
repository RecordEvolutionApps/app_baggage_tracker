import sys
import json
from asyncio import get_event_loop, sleep
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse
import time
from datetime import datetime
from reswarm import Reswarm
from collections import Counter
from datetime import datetime
import shutil
from pathlib import Path
from pprint import pprint
import base64
import torch
import functools
import urllib.request
print = functools.partial(print, flush=True)

with open('/app/video/coco_classes.json', 'r') as f:
  class_id_topic = json.load(f)

OBJECT_MODEL = os.environ.get('OBJECT_MODEL')
RESOLUTION_X = int(os.environ.get('RESOLUTION_X', 640))
RESOLUTION_Y = int(os.environ.get('RESOLUTION_Y', 480))
FRAMERATE = int(os.environ.get('FRAMERATE', 30))

DEVICE_KEY = os.environ.get('DEVICE_KEY')
DEVICE_NAME = os.environ.get('DEVICE_NAME')
DEVICE_URL = os.environ.get('DEVICE_URL')
TUNNEL_PORT = os.environ.get('TUNNEL_PORT')
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

MODEL_RESX = (RESOLUTION_X // 32) * 32 # must be multiple of max stride 32
MODEL_RESY = (RESOLUTION_Y // 32) * 32

saved_masks = []

def downloadModel(model_name, model_path):
    print(f'Downloading Pytorch model {model_name}...')
    urllib.request.urlretrieve(f'https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}.pt', model_path)
    print(f'Download complete!')

def getModel(model_name):
    pytorch_model_path = f'/app/{model_name}.pt'
    tensorrt_initial_model_path = f'/app/{model_name}.engine'

    stored_pytorch_model_path = f'/data/{model_name}.pt'
    stored_tensorrt_model_path = f'/data/{model_name}-{MODEL_RESY}-{MODEL_RESX}.engine'
    model_download_path = f'/app/download/{model_name}.pt'

    stored_tensorrt_file = Path(stored_tensorrt_model_path)
    if stored_tensorrt_file.is_file():
        print(f'Found existing TensorRT Model for {model_name}')
        return YOLO(stored_tensorrt_model_path)

    pytorch_model_file = Path(stored_pytorch_model_path)
    if not pytorch_model_file.is_file():
        print('Original Pytorch model was not found, will download model')

        downloadModel(model_name, model_download_path)

        print('Copying downloaded Pytorch model to /app directory')
        # Move to /app directory to then export it, in case the export fails we don't have any bad data in the /data folder
        shutil.copy(model_download_path, pytorch_model_path)

        print('Moving downloaded Pytorch model to /data directory')
        shutil.move(model_download_path, stored_pytorch_model_path)
    else:
        print('Original Pytorch model was found, copying to main directory to avoid corrupted items in /data')

        print('Copying existing Pytorch model to /app directory')
        # Copy to /app directory to then export it, in case the export fails we don't have any bad data in the /data folder
        shutil.copyfile(stored_pytorch_model_path, pytorch_model_path)
    
    print("Exporting Pytorch model from /app directory into TensorRT....")
    pytorch_model = YOLO(pytorch_model_path)
    pytorch_model.export(format='engine', imgsz=(MODEL_RESY, MODEL_RESX))
    print("Model exported!")

    print(f'Moving exported TensorRT model {model_name} to data folder...')
    shutil.move(tensorrt_initial_model_path, stored_tensorrt_model_path)

    return YOLO(stored_tensorrt_model_path)

parser = argparse.ArgumentParser(description='Start a Video Stream for the given Camera Device')

parser.add_argument('device', type=str, help='A device path like e.g. /dev/video0')
parser.add_argument('camStream', type=str, help='One of frontCam, leftCam, rightCam, backCam')

args = parser.parse_args()

portMap = {"frontCam": 5004,
           "leftCam": 5005,
           "rightCam": 5006,
           "backCam": 5007}

device = args.device

model = getModel(OBJECT_MODEL)
# bounding_box_annotator = sv.BoundingBoxAnnotator()
bounding_box_annotator = sv.DotAnnotator(radius=6)
label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3)

print('CAMERA USED:' + device)
cap = cv2.VideoCapture(int(device[-1]))
cap.set(3, RESOLUTION_X)
cap.set(4, RESOLUTION_Y)

# print("CUDA available:", torch.cuda.is_available(), 'GPUs', torch.cuda.device_count())

outputFormat = " videoconvert ! vp8enc deadline=2 threads=4 keyframe-max-dist=6 ! video/x-vp8 ! rtpvp8pay pt=96"
# outputFormat = "nvvidconv ! nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true insert-vui=true ! h264parse ! rtph264pay"

writerStream = "appsrc do-timestamp=true ! " + outputFormat + " ! udpsink host=janus port=" + str(portMap[args.camStream])
# print(writerStream)

out = cv2.VideoWriter(writerStream, 0, FRAMERATE, (RESOLUTION_X, RESOLUTION_Y))

# Function to display frame rate and timestamp on the frame
def overlay_text(frame, text, position=(10, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def count_polygon_zone(zone):
    count_dict = {}
    for class_id in CLASS_LIST:
        count = zone.class_in_current_count.get(class_id, 0)
        count_dict[class_id] = count
    return count_dict

def count_detections(detections):
    count_dict = {}
    for xyxy, mask, conf, class_id, tracker_id, data in detections:
        if class_id in count_dict:
            count_dict[class_id] += 1
        else:
            count_dict[class_id] = 1
    return count_dict

async def main():
    try:
        print('starting main video loop...')
        fps_monitor = sv.FPSMonitor()
        start_time = time.time()
        global out
        global saved_masks

        while cap.isOpened():
            elapsed_time = time.time() - start_time
            fps_monitor.tick()

            success, frame = cap.read()
            if not success:
                continue


            results = model(frame, imgsz=(MODEL_RESY, MODEL_RESX), conf=CONF, iou=IOU, half=True, verbose=False, classes=CLASS_LIST)
            detections = sv.Detections.from_ultralytics(results[0])
            counts = []

            for saved_mask in saved_masks:
                zone = saved_mask['zone']
                count = zone.current_count

                label = str(count) + ' - ' + saved_mask['label']
                color = saved_mask['color']


                zone_mask = zone.trigger(detections=detections)
                filtered_detections = detections[zone_mask]

                zone_annotator = sv.PolygonZoneAnnotator(
                    zone=zone,
                    color=sv.Color.from_hex(color),
                )

                count_dict = count_polygon_zone(zone)
                counts.append({'label': saved_mask['label'], 'count': count_dict})

                frame = bounding_box_annotator.annotate(scene=frame, detections=filtered_detections)
                frame = label_annotator.annotate(scene=frame, detections=filtered_detections)
                frame = zone_annotator.annotate(scene=frame, label=label)


            # Annotate all detections if no masks are defined
            if len(saved_masks) == 0:
                frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections)

                count_dict = count_detections(detections)
                counts.append({'label': DEVICE_NAME + "_" + "default", 'count': count_dict})

            # Draw FPS and Timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            overlay_text(frame, f'Timestamp: {current_time}', position=(10, 30))
            overlay_text(frame, f'FPS: {fps_monitor.fps:.2f}', position=(10, 60))

            if elapsed_time >= 1.0:
                for item in counts:
                    publishImage(frame)
                    publishClassCount(item["count"], item["label"])
                
                publishCameras()

                start_time = time.time()

            out.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break

            await sleep(0) # Give other ask time to run, not a hack: https://superfastpython.com/what-is-asyncio-sleep-zero/#:~:text=You%20can%20force%20the%20current,before%20resuming%20the%20current%20task.

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print('error in the video loop, exiting', e)
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

async def readMasksFromStdin():
    print("Reading STDIN")

    while True:
        try:
            masksJSON = await get_event_loop().run_in_executor(None, sys.stdin.readline)

            print("Got masks from stdin:", masksJSON)

            stdin_masks = json.loads(masksJSON)

            global saved_masks
            saved_masks = []
            for mask in stdin_masks:
                polygon = np.array(mask['points'])
                polygon.astype(int)

                polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(RESOLUTION_X, RESOLUTION_Y))
                mask['zone'] = polygon_zone

                saved_masks.append(mask)
        except Exception as e:
            print(e)

            

if __name__ == "__main__":
    # run the main coroutine
    task = get_event_loop().create_task(main())
    task2 = get_event_loop().create_task(readMasksFromStdin())

    # run the reswarm component
    rw.run()
