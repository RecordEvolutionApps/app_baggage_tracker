from ultralytics import YOLO
import supervision as sv
import numpy as np
import urllib.request
from pathlib import Path
import shutil
import json
import os
from asyncio import get_event_loop, sleep
import sys
import cv2
import traceback
import torch

OBJECT_MODEL = os.environ.get('OBJECT_MODEL')
RESOLUTION_X = int(os.environ.get('RESOLUTION_X', 640))
RESOLUTION_Y = int(os.environ.get('RESOLUTION_Y', 480))
DEVICE_NAME = os.environ.get('DEVICE_NAME')
CONF = float(os.environ.get('CONF', '0.1'))
IOU = float(os.environ.get('IOU', '0.8'))
SMOOTHING = (os.environ.get('SMOOTHING', 'true') == 'true')
FRAME_BUFFER = int(os.environ.get('FRAME_BUFFER', 64))
CLASS_LIST = os.environ.get('CLASS_LIST', '')
CLASS_LIST = CLASS_LIST.split(',')

with open('/app/video/coco_classes.json', 'r') as f:
  class_id_topic = json.load(f)

try:
    CLASS_LIST = [int(num.strip()) for num in CLASS_LIST]
except Exception as err:
    print('Invalid Class list given', CLASS_LIST)
    CLASS_LIST = []

if len(CLASS_LIST) <= 1:
    CLASS_LIST = list(class_id_topic.keys())
    CLASS_LIST = [int(item) for item in CLASS_LIST]

# Supervision Annotations
if (OBJECT_MODEL.endswith('obb')):
    bounding_box_annotator = sv.OrientedBoxAnnotator()
else:
    bounding_box_annotator = sv.BoundingBoxAnnotator()
# bounding_box_annotator = sv.DotAnnotator(radius=6)
label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3)

tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother(length=5)

def downloadModel(model_name, model_path):
    print(f'Downloading Pytorch model {model_name}...')
    urllib.request.urlretrieve(f'https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}.pt', model_path)
    print(f'Download complete!')

def getModel(model_name, model_resx=640, model_resy=640):
    pytorch_model_path = f'/app/{model_name}.pt'
    tensorrt_initial_model_path = f'/app/{model_name}.engine'

    stored_pytorch_model_path = f'/data/{model_name}.pt'
    stored_tensorrt_model_path = f'/data/{model_name}-{model_resy}-{model_resx}.engine'
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
    if not torch.cuda.is_available():
        return pytorch_model
    else:    
        pytorch_model.export(format='engine', imgsz=(model_resy, model_resx))
        print("Model exported!")

        print(f'Moving exported TensorRT model {model_name} to data folder...')
        shutil.move(tensorrt_initial_model_path, stored_tensorrt_model_path)

        return YOLO(stored_tensorrt_model_path)

def get_youtube_video(url, height):
    import yt_dlp
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(url, download=False)
        formats = info_dict.get('formats', [])
        output_resolution = {"height": 0, "width": 0}
        for format in formats:
            resolution = format.get('height')
            if resolution == None:
                continue
            if height and height == int(resolution):
                output_resolution = format
                break
            elif output_resolution is None or resolution > output_resolution['height']:
                output_resolution = format
        return output_resolution

# Function to display frame rate and timestamp on the frame
def overlay_text(frame, text, position=(10, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def count_polygon_zone(zone, class_list):
    count_dict = {}
    for class_id in class_list:
        count = zone.class_in_current_count.get(class_id, 0)
        count_dict[class_id] = count
    return count_dict

def count_detections(detections):
    count_dict = {}
    try:
        for xyxy, mask, conf, class_id, tracker_id, data in detections:
            if class_id in count_dict:
                count_dict[class_id] += 1
            else:
                count_dict[class_id] = 1
    except Exception as e:
        print('failed to count', e)
    return count_dict


async def readMasksFromStdin(saved_masks):
    print("Reading STDIN")
    try:
        with open('/data/mask.json', 'r') as f:
            loaded_masks = json.load(f)

        saved_masks[:] = []
        saved_masks.extend(prepMasks(loaded_masks))

    except Exception as e:
        traceback.print_exc()
        print('Failed to load masks initially', e)

    while True:
        try:
            masksJSON = await get_event_loop().run_in_executor(None, sys.stdin.readline)

            print("Got masks from stdin:", masksJSON)

            stdin_masks = json.loads(masksJSON)
    
            saved_masks[:] = []
            saved_masks.extend(prepMasks(stdin_masks))
            
        except Exception as e:
            print(e)
        await sleep(0)

def get_contrast_color(hex_color):
    c = sv.Color.from_hex(hex_color)
    luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b

    if luminance < 128:
        return sv.Color(255, 255, 255)  # White for dark colors
    else:
        return sv.Color(0, 0, 0)  # Black for light colors

def prepMasks(in_masks):

    pre_masks = [
        {
            'label': mask['label'],
            'type': mask.get('type', 'ZONE'),
            'points': [(int(point['x']), int(point['y'])) for point in mask['points'][:-1]],
            'color': mask['lineColor']
        }
        for mask in in_masks['polygons']
    ]
    out_masks = []

    for mask in pre_masks:
        polygon = np.array(mask['points'])
        polygon.astype(int)

        if mask['type'] == 'ZONE':
            polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(RESOLUTION_X, RESOLUTION_Y), triggering_position=sv.Position.CENTER)
            mask['zone'] = polygon_zone
            zone_annotator = sv.PolygonZoneAnnotator(
                zone=polygon_zone,
                text_color=get_contrast_color(mask['color']),
                color=sv.Color.from_hex(mask['color']),
            )
            mask['annotator'] = zone_annotator
        elif mask['type'] == 'LINE':
            START = sv.Point(polygon[0][0], polygon[0][1])
            END = sv.Point(polygon[1][0], polygon[1][1])
            line_zone = sv.LineZone(start=START, end=END, triggering_anchors=[sv.Position.CENTER])

            mask['line'] = line_zone

            line_annotator = sv.LineZoneAnnotator(
                thickness=1,
                text_thickness=1,
                text_scale=0.5,
                text_color=get_contrast_color(mask['color']),
                color=sv.Color.from_hex(mask['color']))
            mask['annotator'] = line_annotator

        out_masks.append(mask)
    print('Refreshed Masks', out_masks)
    return out_masks

def get_extreme_points(masks):
    if len(masks) == 0:
        return 0, 0, RESOLUTION_X, RESOLUTION_Y
    low_x = RESOLUTION_X - 1
    low_y = RESOLUTION_Y - 1
    high_x = -1
    high_y = -1
    buf = FRAME_BUFFER
    for mask in masks:
        points = np.array(mask["points"])
        points.astype(int)
        for point in points:
            low_x = point[0] if point[0] < low_x else low_x
            low_y = point[1] if point[1] < low_y else low_y
            high_x = point[0] if point[0] > high_x else high_x
            high_y = point[1] if point[1] > high_y else high_y

    return max(0, low_x - buf), max(0, low_y - buf), min(RESOLUTION_X - 1, high_x + buf), min(RESOLUTION_Y - 1, high_y + buf)


def initSliceInferer(model):
    computer = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def inferSlice(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice, device=computer, conf=CONF, iou=IOU, verbose=False, classes=CLASS_LIST)
        return sv.Detections.from_ultralytics(result[0])
    
    slicer = sv.InferenceSlicer(
        callback=inferSlice, 
        slice_wh=[640, 640], 
        overlap_ratio_wh=[0.2, 0.2],
        iou_threshold=IOU,
        thread_workers=6
        )
    return slicer

def infer(frame, model, model_resx, model_resy):
    computer = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    results = model(frame, device=computer, imgsz=(model_resy, model_resx), conf=CONF, iou=IOU, verbose=False, classes=CLASS_LIST)
    if len(results) == 0:
        return false
    try:
        detections = sv.Detections.from_ultralytics(results[0])
    except Exception as e:
        print('Failed to extract detections from model result', e)
        traceback.print_exc()
        return false
    return detections

def move_detections(detections: sv.Detections, offset_x: int, offset_y: int) -> sv.Detections:
  for i in range(len(detections.xyxy)):
    box = detections.xyxy[i]
    box[0] += offset_x  # xmin
    box[1] += offset_y  # ymin
    box[2] += offset_x  # xmax
    box[3] += offset_y  # ymax

  return detections

def processFrame(frame, detections, saved_masks):

    try:
        detections = tracker.update_with_detections(detections)
        if SMOOTHING:
            detections = smoother.update_with_detections(detections)
    except Exception as e:
        print('Error when smoothing detections or updating tracker', str(e))
        traceback.print_exc()

    zoneCounts = []
    lineCounts = []

    # line_zone.trigger(detections)
    # frame = line_zone_annotator.annotate(frame, line_counter=line_zone)

    for saved_mask in saved_masks:
        if saved_mask['type'] == 'ZONE':
            zone = saved_mask['zone']
            zone_annotator = saved_mask['annotator']
            try:
                zone_mask = zone.trigger(detections=detections)
            except Exception as e:
                print('Failed to get detections')
                continue

            count = zone.current_count
            zone_label = str(count) + ' - ' + saved_mask['label']

            filtered_detections = detections[zone_mask]
            
            # labels = [f"{class_id_topic[str(class_id)]} #{tracker_id}" for class_id, tracker_id in zip(filtered_detections.class_id, filtered_detections.tracker_id)]

            count_dict = count_polygon_zone(zone, CLASS_LIST)
            zoneCounts.append({'label': saved_mask['label'], 'count': count_dict})

            frame = bounding_box_annotator.annotate(scene=frame, detections=filtered_detections)
            frame = label_annotator.annotate(scene=frame, detections=filtered_detections)
            frame = zone_annotator.annotate(scene=frame, label=zone_label)
        elif saved_mask['type'] == 'LINE':
            lineZone = saved_mask['line']
            line_annotator = saved_mask['annotator']
            try:
                crossed_in, crossed_out = lineZone.trigger(detections=detections)
                detections_in = detections[crossed_in]
                detections_out = detections[crossed_out]
                num_in = len(detections_in.xyxy)
                num_out = len(detections_out.xyxy)
                if num_in > 0 or num_out > 0:
                    lineCounts.append({'label': saved_mask['label'], 'num_in': num_in, 'num_out': num_out})
            except Exception as e:
                traceback.print_exc()
                print('Failed to get line counts')
                # continue

            frame = line_annotator.annotate(frame, line_counter=lineZone)

    # Annotate all detections if no zones are defined
    if len([m for m in saved_masks if m['type'] == 'ZONE']) == 0:
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        # labels = [f"{class_id_topic[str(class_id)]} #{tracker_id}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]
        frame = label_annotator.annotate(scene=frame, detections=detections)

        count_dict = count_detections(detections)
        zoneCounts.append({'label': DEVICE_NAME + "_" + "default", 'count': count_dict})
    return frame, zoneCounts, lineCounts