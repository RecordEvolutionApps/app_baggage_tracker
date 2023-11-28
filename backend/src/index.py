from ultralytics import YOLO
import cv2
import numpy as np
import math
import os
import argparse
import time
from datetime import datetime

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

outputFormat = " videoconvert ! vp8enc deadline=2 threads=4 keyframe-max-dist=6 ! video/x-vp8 ! rtpvp8pay pt=96"
# outputFormat = "nvvidconv ! nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true insert-vui=true ! h264parse ! rtph264pay"

writerStream = "appsrc do-timestamp=true ! " + outputFormat + " ! udpsink host=127.0.0.1 port=" + str(portMap[args.cam])
print(writerStream)

out = cv2.VideoWriter(writerStream, 0, FRAMERATE, (RESOLUTION_X, RESOLUTION_Y))

start_time = time.time()
frame_count = 0
fps = 0

# Function to display frame rate and timestamp on the frame
def overlay_text(frame, text, position=(10, 30), font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

while True:
    success, img = cap.read()
    results = model(img, stream=True, imgsz=RESOLUTION_X, conf=0.1, iou=0.7, classes=[2, 3, 5, 7])
    for r in results:
        annotated_frame = r.plot(line_width=1, probs=False, font_size=14)
        # boxes = r.boxes.xywh.cpu()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    overlay_text(annotated_frame, f'Timestamp: {current_time}', position=(10, 30))

    frame_count += 1
    elapsed_time = time.time() - start_time

    # Update frame rate every second
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0    
        
    overlay_text(annotated_frame, f'FPS: {fps:.2f}', position=(10, 60))

    out.write(annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()