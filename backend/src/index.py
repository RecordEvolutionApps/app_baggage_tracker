from ultralytics import YOLO
import cv2
import numpy as np
import math
import os
import argparse

parser = argparse.ArgumentParser(description='Start a Video Stream for the given Camera Device')

parser.add_argument('device', type=str, help='A device path like e.g. /dev/video0')

args = parser.parse_args()


CAM_NUM = os.environ.get('CAM_NUM', 1)
device = args.device or f"/dev/video{CAM_NUM}"

RESOLUTION_X = int(os.environ.get('RESOLUTION_X', 640))
RESOLUTION_Y = int(os.environ.get('RESOLUTION_Y', 480))
FRAMERATE = int(os.environ.get('FRAMERATE', 20))

print('CAMERA USED:' + device)

#gstring = f"v4l2src device={device} ! videorate ! image/jpeg,format=I420,width={RESOLUTION_X},height={RESOLUTION_Y},framerate={FRAMERATE}/1"
cap = cv2.VideoCapture(f"v4l2src device={device}")
cap.set(3, RESOLUTION_X)
cap.set(4, RESOLUTION_Y)

# model
model = YOLO("./yolov8n.pt")

framerate = 25.0

out = cv2.VideoWriter('appsrc ! videoconvert ! '
                      'vp8enc deadline=2 threads=2 keyframe-max-dist=60 ! video/x-vp8 ! '
                      'rtpvp8pay !'
                      'udpsink host=127.0.0.1 port=5004',
                      0, framerate, (RESOLUTION_X, RESOLUTION_Y))

while True:
    success, img = cap.read()
    results = model(img, stream=True, imgsz=RESOLUTION_X, classes=[2, 3, 5, 7])
    
    # coordinates
    for r in results:
        annotated_frame = r.plot(line_width=1, probs=False, font_size=14)
        boxes = r.boxes.xywh.cpu()
 
    out.write(annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()