from ultralytics import YOLO
import cv2
import numpy as np
import math
import os
import argparse
import time
from datetime import datetime
from reswarm import Reswarm
from asyncio import sleep
from collections import Counter
from asyncio import get_event_loop, sleep
from datetime import datetime
import polars as pl
from pprint import pprint
rw = Reswarm()

DEVICE_KEY = os.environ.get('DEVICE_KEY')
TUNNEL_PORT = os.environ.get('TUNNEL_PORT')

parser = argparse.ArgumentParser(description='Start a Video Stream for the given Camera Device')

parser.add_argument('device', type=str, help='A device path like e.g. /dev/video0')
parser.add_argument('cam', type=str, help='One of frontCam, leftCam, rightCam, backCam')

args = parser.parse_args()

portMap = {"frontCam": 5004,
           "leftCam": 5005,
           "rightCam": 5006,
           "backCam": 5007}


CAM_NUM = os.environ.get('CAM_NUM', 1)

RESOLUTION_X = int(os.environ.get('RESOLUTION_X', 640))
RESOLUTION_Y = int(os.environ.get('RESOLUTION_Y', 480))
FRAMERATE = int(os.environ.get('FRAMERATE', 20))

device = args.device
gstring = f"v4l2src device={device} ! video/x-raw, width={RESOLUTION_X},height={RESOLUTION_Y} ! videoconvert ! appsink "
print('CAMERA USED:' + gstring)
cap = cv2.VideoCapture(gstring)
cap.set(3, RESOLUTION_X)
cap.set(4, RESOLUTION_Y)

# model
model = YOLO("/app/yolov8.pt")
model.fuse()
pprint('----------Model Device -----------')
pprint(model.device)

outputFormat = " videoconvert ! vp8enc deadline=2 threads=4 keyframe-max-dist=6 ! video/x-vp8 ! rtpvp8pay pt=96"
# outputFormat = "nvvidconv ! nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true insert-vui=true ! h264parse ! rtph264pay"

writerStream = "appsrc do-timestamp=true ! " + outputFormat + " ! udpsink host=127.0.0.1 port=" + str(portMap[args.cam])
print(writerStream)

out = cv2.VideoWriter(writerStream, 0, FRAMERATE, (RESOLUTION_X, RESOLUTION_Y))

# Function to display frame rate and timestamp on the frame
def overlay_text(frame, text, position=(10, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


async def main():
    start_time = time.time()
    frame_count = 0
    fps = 0
    global out
    global pub
    while True:
        success, img = cap.read()
        results = model(img, stream=True, imgsz=RESOLUTION_X, conf=0.1, iou=0.7, classes=[2, 3, 5, 7])
        frame_count += 1
        elapsed_time = time.time() - start_time
        for r in results:
            annotated_frame = r.plot(line_width=1, probs=False, font_size=14)
            # Update frame rate every second
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                await publishClassCount(r)
                start_time = time.time()
                frame_count = 0  

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        overlay_text(annotated_frame, f'Timestamp: {current_time}', position=(10, 30))
            
        overlay_text(annotated_frame, f'FPS: {fps:.2f}', position=(10, 60))
        sleep(2)
        out.write(annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        await sleep(1)

    cap.release()
    cv2.destroyAllWindows()


async def publishClassCount(result):
    classes = []
    boxes = result.boxes
    # pprint(boxes.cls)
    classes = [result.names[int(cls)] for cls in boxes.cls]
    # pprint(classes)

    df = pl.DataFrame({"class": classes})
    agg = df.group_by("class").count()
    # pprint(agg)
    payload = agg.to_dicts()
    now = datetime.now().astimezone().isoformat()
    # pprint('sennding')
    for d in payload:
        d["tsp"] = now
        d["videolink"] = f"https://{DEVICE_KEY}-traffic-{TUNNEL_PORT}.app.record-evolution.com"
        pprint(d)
        await rw.publish_to_table('detections', d)

if __name__ == "__main__":
    # run the main coroutine
    get_event_loop().create_task(main())
    # run the reswarm component
    rw.run()


