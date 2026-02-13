import supervision as sv
import numpy as np
import urllib.request
from pathlib import Path
import shutil
import json
import os
import subprocess
from asyncio import get_event_loop, sleep
import sys
import cv2
import traceback
import torch
from typing import Dict, Any, Optional

OBJECT_MODEL = os.environ.get('OBJECT_MODEL', 'rtmdet_tiny_8xb32-300e_coco')
DETECT_BACKEND = os.environ.get('DETECT_BACKEND', 'mmdet')
RESOLUTION_X = int(os.environ.get('RESOLUTION_X', 640))
RESOLUTION_Y = int(os.environ.get('RESOLUTION_Y', 480))
DEVICE_NAME = os.environ.get('DEVICE_NAME', 'UNKNOWN_DEVICE')
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

# Supervision Annotations (RTMDet does not use OBB, always use BoxAnnotator)
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3)

tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother(length=5)

def empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int64),
    )

MMDET_MODEL_ZOO = {
    "rtmdet_tiny_8xb32-300e_coco": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    },
    "rtmdet_s_8xb32-300e_coco": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    },
    "rtmdet_m_8xb32-300e_coco": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    },
}

def download_file(url: str, destination: str) -> None:
    print(f'Downloading {url}...')
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    print('Download complete!')

def getModel(model_name: str, model_resx: int = 640, model_resy: int = 640) -> Dict[str, Any]:
    backend = DETECT_BACKEND.lower()
    if backend == 'mmdet':
        return get_mmdet_model(model_name)
    raise ValueError(f'Unsupported DETECT_BACKEND: {backend}. Only mmdet is supported.')

def get_mmdet_model(model_name: str) -> Dict[str, Any]:
    if model_name not in MMDET_MODEL_ZOO:
        raise ValueError(f'Unknown MMDetection model: {model_name}. Available: {", ".join(MMDET_MODEL_ZOO.keys())}')

    cache_root = Path('/data/mmdet')
    checkpoint_path = cache_root / 'checkpoints' / f'{model_name}.pth'

    # Pre-download checkpoint for caching (configs are resolved by DetInferencer
    # using the model name + metafile, which properly handles _base_ imports)
    if not checkpoint_path.is_file():
        download_file(MMDET_MODEL_ZOO[model_name]['checkpoint'], str(checkpoint_path))

    try:
        from mmdet.apis import DetInferencer
    except Exception as exc:
        raise RuntimeError('MMDetection is not installed. Install mmdet, mmengine, and mmcv for DETECT_BACKEND=mmdet.') from exc

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Pass the model name (not a config file path) so DetInferencer resolves
    # the full config from its metafile, properly handling _base_ inheritance.
    inferencer = DetInferencer(model=model_name, weights=str(checkpoint_path), device=device)
    return {
        'backend': 'mmdet',
        'inferencer': inferencer,
        'model_name': model_name,
    }

def get_youtube_video(url, height):
    import yt_dlp
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['web'],
            },
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
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
        if output_resolution is None:
            output_resolution = {}
        output_resolution['http_headers'] = info_dict.get('http_headers', {})
        if 'Referer' not in output_resolution['http_headers']:
            output_resolution['http_headers']['Referer'] = 'https://www.youtube.com/'
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


def initSliceInferer(model_bundle: Dict[str, Any]):
    def inferSlice(image_slice: np.ndarray) -> Optional[sv.Detections]:
        detections = infer_frame(image_slice, model_bundle, model_resx=None, model_resy=None)
        return detections if detections is not False else empty_detections()

    slicer = sv.InferenceSlicer(
        callback=inferSlice,
        slice_wh=[640, 640],
        overlap_wh=[int(0.2 * 640), int(0.2 * 640)],
        iou_threshold=IOU,
        thread_workers=6
    )
    return slicer

def infer(frame, model_bundle, model_resx, model_resy):
    return infer_frame(frame, model_bundle, model_resx=model_resx, model_resy=model_resy)

def infer_frame(frame: np.ndarray, model_bundle: Dict[str, Any], model_resx: Optional[int], model_resy: Optional[int]):
    backend = model_bundle.get('backend')
    if backend == 'mmdet':
        return infer_mmdet(frame, model_bundle['inferencer'])
    raise ValueError(f'Unsupported backend in model bundle: {backend}')

def infer_mmdet(frame: np.ndarray, inferencer) -> Optional[sv.Detections]:
    try:
        results = inferencer(frame, return_vis=False)
        predictions = results.get('predictions', [])
        if not predictions:
            return False
        pred = predictions[0]
        bboxes = np.array(pred.get('bboxes', []), dtype=np.float32)
        scores = np.array(pred.get('scores', []), dtype=np.float32)
        labels = np.array(pred.get('labels', []), dtype=np.int64)
        if bboxes.size == 0:
            return False

        keep = scores >= CONF
        if CLASS_LIST:
            class_mask = np.isin(labels, CLASS_LIST)
            keep = np.logical_and(keep, class_mask)

        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if bboxes.size == 0:
            return False

        return sv.Detections(
            xyxy=bboxes,
            confidence=scores,
            class_id=labels,
        )
    except Exception as e:
        print('Failed to extract detections from MMDetection result', e)
        traceback.print_exc()
        return False

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

    zoneMasks = [m for m in saved_masks if m['type'] == 'ZONE']
    lineMasks = [m for m in saved_masks if m['type'] == 'LINE']

    # Annotate all detections if no zones are defined
    if len(zoneMasks) == 0:
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        # labels = [f"{class_id_topic[str(class_id)]} #{tracker_id}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]
        frame = label_annotator.annotate(scene=frame, detections=detections)

        count_dict = count_detections(detections)
        zoneCounts.append({'label': DEVICE_NAME + "_" + "default", 'count': count_dict})
    else:
        for saved_mask in zoneMasks:
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

    for saved_mask in lineMasks:
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

    
    return frame, zoneCounts, lineCounts