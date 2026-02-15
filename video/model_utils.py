from __future__ import annotations

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
import logging
import torch
from typing import Dict, Any, Optional

# Jetson L4T PyTorch lacks full distributed support — stub out ReduceOp
# so that mmengine doesn't crash on import.
if not hasattr(torch.distributed, 'ReduceOp'):
    class _ReduceOpStub:
        SUM = 0; PRODUCT = 1; MIN = 2; MAX = 3; BAND = 4; BOR = 5; BXOR = 6
    torch.distributed.ReduceOp = _ReduceOpStub

logger = logging.getLogger('model_utils')

# --------------------------------------------------------------------------- #
# Monkey-patch mmcv NMS: fall back to torchvision when the compiled CUDA
# kernel is unavailable (common on Jetson where mmcv was not compiled with
# the correct TORCH_CUDA_ARCH_LIST).
# --------------------------------------------------------------------------- #
def _patch_mmcv_nms():
    try:
        from mmcv.ops import nms as _nms_mod
        _orig_apply = _nms_mod.NMSop.apply

        @staticmethod
        def _safe_nms(*args, **kwargs):
            try:
                return _orig_apply(*args, **kwargs)
            except RuntimeError as e:
                if 'nms_impl' not in str(e):
                    raise
                # Fall back to torchvision NMS (CUDA-accelerated on Jetson)
                import torchvision
                bboxes, scores, iou_threshold = args[0], args[1], args[2]
                max_num = args[5] if len(args) > 5 else -1
                keep = torchvision.ops.nms(bboxes, scores, float(iou_threshold))
                if max_num > 0:
                    keep = keep[:max_num]
                return keep

        _nms_mod.NMSop.apply = _safe_nms
        logger.info('Patched mmcv NMS with torchvision fallback')
    except Exception:
        pass  # mmcv not installed or structure changed — skip silently

_patch_mmcv_nms()

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

try:
    CLASS_LIST = [int(num.strip()) for num in CLASS_LIST]
except Exception as err:
    logger.warning('Invalid Class list given: %s', CLASS_LIST)
    CLASS_LIST = []

if len(CLASS_LIST) <= 1:
    CLASS_LIST = []

# Open-vocabulary class names (text labels for models like Grounding DINO)
CLASS_NAMES: list[str] = []

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
        "native_input_wh": (640, 640),
    },
    "rtmdet_s_8xb32-300e_coco": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
        "native_input_wh": (640, 640),
    },
    "rtmdet_m_8xb32-300e_coco": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
        "native_input_wh": (640, 640),
    },
}

def download_file(url: str, destination: str) -> None:
    logger.info('Downloading %s...', url)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    logger.info('Download complete')

def is_model_cached(model_name: str) -> bool:
    """Check whether a model's checkpoint is already available locally."""
    if model_name in ('none', ''):
        return True
    cache_root = Path('/data/mmdet')
    checkpoint_path = cache_root / 'checkpoints' / f'{model_name}.pth'
    if checkpoint_path.is_file():
        return True
    # Check mmengine / torch hub default cache locations
    for cache_dir in [
        Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints',
        Path.home() / '.cache' / 'mim',
    ]:
        if cache_dir.is_dir():
            # Any file containing the model name is a hit
            for f in cache_dir.iterdir():
                if model_name.replace('-', '_') in f.name.replace('-', '_'):
                    return True
    return False


def prepare_model(model_name: str, progress_callback=None):
    """Download a model checkpoint if not cached. Returns when ready.

    progress_callback(status, progress, message) is called with:
      - ("checking", 0, "...")
      - ("downloading", 0-100, "...")
      - ("ready", 100, "...")
      - ("error", 0, "...")
    """
    def _report(status, progress=0, message=''):
        if progress_callback:
            progress_callback(status, progress, message)

    _report('checking', 0, f'Checking cache for {model_name}...')

    if model_name in ('none', ''):
        _report('ready', 100, 'No model needed')
        return

    if is_model_cached(model_name):
        _report('ready', 100, f'{model_name} already cached')
        return

    _report('downloading', 0, f'Downloading {model_name}...')

    cache_root = Path('/data/mmdet')
    checkpoint_path = cache_root / 'checkpoints' / f'{model_name}.pth'

    # For curated models, download with progress tracking
    if model_name in MMDET_MODEL_ZOO:
        url = MMDET_MODEL_ZOO[model_name]['checkpoint']
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _download_with_progress(url, str(checkpoint_path), _report)
        _report('ready', 100, f'{model_name} downloaded')
        return

    # For all other models, resolve the checkpoint URL from openmim metafiles
    # (DetInferencer's internal registry is smaller and may not find the model)
    weight_url = _resolve_weight_url(model_name)
    if weight_url:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _report('downloading', 5, f'Downloading {model_name} from {weight_url[:80]}...')
        _download_with_progress(weight_url, str(checkpoint_path), _report)
        _report('ready', 100, f'{model_name} downloaded')
        return

    # Last resort: try DetInferencer auto-download
    try:
        from mmdet.apis import DetInferencer
        device = 'cpu'  # Use CPU for download-only; saves GPU memory
        _report('downloading', 10, f'Resolving {model_name} from DetInferencer...')
        DetInferencer(model=model_name, device=device)
        _report('ready', 100, f'{model_name} downloaded and cached')
    except Exception as e:
        _report('error', 0, str(e))
        raise


def _download_with_progress(url: str, dest: str, report_fn):
    """Download a file with progress reporting."""
    import urllib.request
    Path(dest).parent.mkdir(parents=True, exist_ok=True)

    response = urllib.request.urlopen(url)
    total = int(response.headers.get('Content-Length', 0))
    downloaded = 0
    block_size = 1024 * 256  # 256 KB
    last_pct = -1

    with open(dest, 'wb') as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
                if pct != last_pct:
                    last_pct = pct
                    report_fn('downloading', pct, f'{downloaded}/{total} bytes')

    report_fn('downloading', 100, 'Download complete')


def _resolve_weight_url(model_name: str) -> str | None:
    """Look up the checkpoint download URL from openmim's metafile data."""
    try:
        from mim.commands.search import get_model_info
        import pandas as pd
        df = get_model_info('mmdet', shown_fields=['config', 'weight'])
        for _, row in df.iterrows():
            config_path = row.get('config', '')
            if not config_path:
                continue
            name = os.path.basename(config_path).replace('.py', '')
            if name == model_name:
                weight = row.get('weight', '')
                if pd.notna(weight) and str(weight).startswith('http'):
                    return str(weight)
    except Exception as e:
        logger.warning('_resolve_weight_url failed for %s: %s', model_name, e)
    return None


# ── Backend status reporting ────────────────────────────────────────────────

def write_backend_status(cam_stream: str, model_bundle: Dict[str, Any], extra: Dict[str, Any] | None = None):
    """Write a JSON status file so the API / frontend can inspect the active backend.

    Written to /data/status/<camStream>.backend.json with keys:
      backend       – 'mmdet', 'tensorrt', or 'none'
      model         – model name
      precision     – 'fp16', 'fp32', or 'n/a'
      device        – 'cuda:0' or 'cpu'
      trt_cached    – bool (True if engine was loaded from cache)
      message       – human-readable summary
    """
    import torch
    status_dir = Path('/data/status')
    status_dir.mkdir(parents=True, exist_ok=True)

    backend = model_bundle.get('backend', 'unknown')
    model_name = model_bundle.get('model_name', '')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if backend == 'tensorrt':
        precision = 'fp16'
        trt_cached = bool(model_bundle.get('trt_cached'))
        message = f'TensorRT FP16 engine active for {model_name}'
        if trt_cached:
            message += ' (loaded from cache)'
    elif backend == 'mmdet':
        precision = 'fp32'
        trt_cached = False
        # Check if tensorrt was requested but fell back
        requested = DETECT_BACKEND.lower()
        if requested == 'tensorrt':
            message = f'PyTorch FP32 fallback (TensorRT build failed) for {model_name}'
        else:
            message = f'PyTorch FP32 (MMDetection) for {model_name}'
    else:
        precision = 'n/a'
        trt_cached = False
        message = 'No inference backend'

    status = {
        'backend': backend,
        'model': model_name,
        'precision': precision,
        'device': device,
        'trt_cached': trt_cached,
        'requested_backend': DETECT_BACKEND.lower(),
        'message': message,
    }
    if extra:
        status.update(extra)

    status_file = status_dir / f'{cam_stream}.backend.json'
    with open(status_file, 'w') as f:
        json.dump(status, f)
    logger.info('[status] %s: %s', cam_stream, message)


def getModel(model_name: str) -> Dict[str, Any]:
    backend = DETECT_BACKEND.lower()
    if backend == 'tensorrt':
        return get_tensorrt_model(model_name)
    if backend == 'mmdet':
        return get_mmdet_model(model_name)
    raise ValueError(f'Unsupported DETECT_BACKEND: {backend}. Supported: mmdet, tensorrt')

def get_mmdet_model(model_name: str) -> Dict[str, Any]:
    """Load any MMDetection model by name.

    For models listed in MMDET_MODEL_ZOO the checkpoint is pre-downloaded for
    offline/cached use.  For *any other* model name we resolve the config via
    openmim metafiles (which covers more models than DetInferencer's internal
    registry).
    """
    cache_root = Path('/data/mmdet')
    checkpoint_path = cache_root / 'checkpoints' / f'{model_name}.pth'

    # If we have a curated entry, pre-download the checkpoint for offline use
    if model_name in MMDET_MODEL_ZOO and not checkpoint_path.is_file():
        download_file(MMDET_MODEL_ZOO[model_name]['checkpoint'], str(checkpoint_path))

    try:
        from mmdet.apis import DetInferencer
    except Exception as exc:
        raise RuntimeError('MMDetection is not installed. Install mmdet, mmengine, and mmcv for DETECT_BACKEND=mmdet.') from exc

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Try loading by name first (works for models in DetInferencer's registry)
    # If that fails, resolve the config path from openmim metafiles
    inferencer = None
    if checkpoint_path.is_file():
        try:
            inferencer = DetInferencer(model=model_name, weights=str(checkpoint_path), device=device, show_progress=False)
        except ValueError:
            # Model name not in DetInferencer registry — resolve config from openmim
            config_path = _resolve_config_path(model_name)
            if config_path:
                logger.info('Loading %s via config path: %s', model_name, config_path)
                inferencer = DetInferencer(model=config_path, weights=str(checkpoint_path), device=device, show_progress=False)
            else:
                raise
    else:
        try:
            logger.info('No cached checkpoint for "%s", DetInferencer will auto-download...', model_name)
            inferencer = DetInferencer(model=model_name, device=device, show_progress=False)
        except ValueError:
            config_path = _resolve_config_path(model_name)
            if config_path:
                logger.info('Loading %s via config path (auto-download): %s', model_name, config_path)
                inferencer = DetInferencer(model=config_path, device=device, show_progress=False)
            else:
                raise

    native_wh = MMDET_MODEL_ZOO.get(model_name, {}).get('native_input_wh', (640, 640))
    return {
        'backend': 'mmdet',
        'inferencer': inferencer,
        'model_name': model_name,
        'native_input_wh': native_wh,
    }


def _resolve_config_path(model_name: str) -> str | None:
    """Resolve the absolute config file path for a model from openmim metafiles.

    Returns an absolute path like
    '/usr/local/lib/python3.8/dist-packages/mmdet/.mim/configs/yolox/yolox_tiny_8xb8-300e_coco.py'
    that DetInferencer can load directly as a Config file, or None.
    """
    try:
        import mmdet
        mmdet_root = os.path.dirname(mmdet.__file__)

        from mim.commands.search import get_model_info
        df = get_model_info('mmdet', shown_fields=['config'])
        for _, row in df.iterrows():
            config = row.get('config', '')
            if not config:
                continue
            name = os.path.basename(config).replace('.py', '')
            if name == model_name:
                # Resolve to absolute path inside the installed mmdet package
                for prefix in [
                    os.path.join(mmdet_root, '.mim'),   # pip-installed mmdet
                    mmdet_root,                          # editable / dev install
                ]:
                    abs_path = os.path.join(prefix, config)
                    if os.path.isfile(abs_path):
                        logger.info('Resolved config for %s: %s', model_name, abs_path)
                        return abs_path
                # Last resort: return relative path (will likely fail)
                logger.warning('Config file not found on disk for %s: %s', model_name, config)
                return config
    except Exception as e:
        logger.warning('_resolve_config_path failed for %s: %s', model_name, e)
    return None


def get_tensorrt_model(model_name: str) -> Dict[str, Any]:
    """Load a model as a TensorRT FP16 engine.

    If a cached engine exists it is loaded directly — no MMDet / ONNX overhead.
    Otherwise, the full pipeline runs:  MMDet → ONNX export → TRT engine build.
    Falls back to the mmdet PyTorch backend if TensorRT is unavailable or the
    build fails.
    """
    try:
        import trt_backend
    except ImportError:
        logger.warning('trt_backend module not found; falling back to mmdet')
        return get_mmdet_model(model_name)

    if not trt_backend.is_available():
        logger.warning('TensorRT or CUDA not available; falling back to mmdet backend')
        return get_mmdet_model(model_name)

    input_wh = MMDET_MODEL_ZOO.get(model_name, {}).get('native_input_wh', (640, 640))

    # ── Fast path: load from cached engine (no MMDet needed) ───────────
    if trt_backend.is_engine_cached(model_name):
        try:
            trt_inferencer = trt_backend.TRTInferencer(
                str(trt_backend.engine_path(model_name)), input_wh,
            )
            logger.info('TensorRT engine loaded from cache for %s (FP16)', model_name)
            return {
                'backend': 'tensorrt',
                'inferencer': trt_inferencer,
                'model_name': model_name,
                'native_input_wh': input_wh,
                'trt_cached': True,
            }
        except Exception as e:
            logger.warning('Cached TRT engine failed to load: %s — rebuilding', e)

    # ── Slow path: build engine from MMDet model ──────────────────────
    mmdet_bundle = get_mmdet_model(model_name)
    input_wh = mmdet_bundle['native_input_wh']
    num_classes = mmdet_bundle['inferencer'].model.bbox_head.num_classes

    try:
        trt_inferencer = trt_backend.build_trt_model(
            model_name, mmdet_bundle['inferencer'], input_wh, num_classes,
        )

        # Free the PyTorch model to reclaim GPU memory
        del mmdet_bundle['inferencer']
        torch.cuda.empty_cache()

        logger.info('TensorRT backend ready for %s (FP16)', model_name)
        return {
            'backend': 'tensorrt',
            'inferencer': trt_inferencer,
            'model_name': model_name,
            'native_input_wh': input_wh,
            'num_classes': num_classes,
            'trt_cached': False,
        }
    except Exception as e:
        logger.error('TensorRT build failed: %s — falling back to mmdet', e, exc_info=True)
        return mmdet_bundle


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
        logger.warning('Failed to count: %s', e)
    return count_dict


async def watchMaskFile(saved_masks, cam_stream='frontCam', poll_interval=1.0):
    """
    Poll the per-stream mask file for changes and reload when modified.
    This replaces the old stdin-based approach with a simple file-watch loop.
    """
    mask_path = f'/data/masks/{cam_stream}.json'
    legacy_path = '/data/mask.json'
    last_mtime = 0.0

    logger.info('Watching mask file: %s', mask_path)

    while True:
        try:
            path = mask_path if os.path.exists(mask_path) else legacy_path

            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(path, 'r') as f:
                        loaded_masks = json.load(f)
                    saved_masks[:] = []
                    saved_masks.extend(prepMasks(loaded_masks))
                    logger.info('Reloaded masks from %s', path)
        except Exception as e:
            logger.error('Error reading mask file: %s', e, exc_info=True)

        await sleep(poll_interval)


async def watchSettingsFile(settings_dict, cam_stream='frontCam', poll_interval=1.0):
    """
    Poll the per-stream settings file for changes and update settings_dict.
    The backend writes /data/settings/<camStream>.json when settings change.
    """
    settings_path = f'/data/settings/{cam_stream}.json'
    last_mtime = 0.0

    logger.info('[settings] Watching %s', settings_path)

    while True:
        try:
            if os.path.exists(settings_path):
                mtime = os.path.getmtime(settings_path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(settings_path, 'r') as f:
                        new_settings = json.load(f)
                    # Log which keys changed
                    changed = {}
                    for k, v in new_settings.items():
                        old_v = settings_dict.get(k)
                        if old_v != v:
                            changed[k] = {'from': old_v, 'to': v}
                    settings_dict.update(new_settings)
                    if changed:
                        for k, diff in changed.items():
                            logger.info('[settings] %s: %s: %s -> %s', cam_stream, k, diff['from'], diff['to'])
                    else:
                        logger.debug('[settings] %s: settings file touched (no changes)', cam_stream)
        except Exception as e:
            logger.error('[settings] Error reading settings file: %s', e, exc_info=True)

        await sleep(poll_interval)


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
    logger.info('[masks] Refreshed %d mask(s)', len(out_masks))
    return out_masks

def get_extreme_points(masks, frame_buffer=FRAME_BUFFER):
    if len(masks) == 0:
        return 0, 0, RESOLUTION_X, RESOLUTION_Y
    low_x = RESOLUTION_X - 1
    low_y = RESOLUTION_Y - 1
    high_x = -1
    high_y = -1
    for mask in masks:
        points = np.array(mask["points"])
        points.astype(int)
        for point in points:
            low_x = point[0] if point[0] < low_x else low_x
            low_y = point[1] if point[1] < low_y else low_y
            high_x = point[0] if point[0] > high_x else high_x
            high_y = point[1] if point[1] > high_y else high_y

    return max(0, low_x - frame_buffer), max(0, low_y - frame_buffer), min(RESOLUTION_X - 1, high_x + frame_buffer), min(RESOLUTION_Y - 1, high_y + frame_buffer)


def initSliceInferer(model_bundle: Dict[str, Any], settings_dict=None):
    native_w, native_h = model_bundle.get('native_input_wh', (640, 640))

    def inferSlice(image_slice: np.ndarray) -> Optional[sv.Detections]:
        conf = float(settings_dict.get('confidence', CONF)) if settings_dict else CONF
        detections = infer_frame(image_slice, model_bundle, confidence=conf)
        return detections if detections is not False else empty_detections()

    slicer = sv.InferenceSlicer(
        callback=inferSlice,
        slice_wh=(native_w, native_h),
        overlap_wh=(int(0.2 * native_w), int(0.2 * native_h)),
        iou_threshold=IOU,
        thread_workers=6
    )
    return slicer

def infer(frame, model_bundle, confidence=None):
    return infer_frame(frame, model_bundle, confidence=confidence)

def infer_frame(frame: np.ndarray, model_bundle: Dict[str, Any], confidence=None):
    backend = model_bundle.get('backend')
    if backend == 'tensorrt':
        return infer_tensorrt(frame, model_bundle, confidence=confidence)
    if backend == 'mmdet':
        return infer_mmdet(frame, model_bundle['inferencer'], confidence=confidence)
    raise ValueError(f'Unsupported backend in model bundle: {backend}')

def infer_mmdet(frame: np.ndarray, inferencer, confidence=None) -> Optional[sv.Detections]:
    try:
        results = inferencer(frame, return_vis=False, no_save_pred=True, print_result=False)
        predictions = results.get('predictions', [])
        if not predictions:
            return False
        pred = predictions[0]
        bboxes = np.array(pred.get('bboxes', []), dtype=np.float32)
        scores = np.array(pred.get('scores', []), dtype=np.float32)
        labels = np.array(pred.get('labels', []), dtype=np.int64)
        if bboxes.size == 0:
            return False

        conf_threshold = confidence if confidence is not None else CONF
        keep = scores >= conf_threshold
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
        logger.error('Failed to extract detections from MMDetection result: %s', e, exc_info=True)
        return False


def infer_tensorrt(frame: np.ndarray, model_bundle: Dict[str, Any], confidence=None) -> Optional[sv.Detections]:
    """Run inference through the TensorRT engine."""
    trt_inf = model_bundle['inferencer']
    conf = confidence if confidence is not None else CONF
    class_list = CLASS_LIST if CLASS_LIST else None
    try:
        result = trt_inf(frame, conf=conf, iou=IOU, class_list=class_list)
        return result if result is not None else False
    except Exception as e:
        logger.error('TensorRT inference failed: %s', e, exc_info=True)
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
        logger.warning('Error when smoothing detections or updating tracker: %s', e, exc_info=True)

    zoneCounts = []
    lineCounts = []

    # line_zone.trigger(detections)
    # frame = line_zone_annotator.annotate(frame, line_counter=line_zone)

    zoneMasks = [m for m in saved_masks if m['type'] == 'ZONE']
    lineMasks = [m for m in saved_masks if m['type'] == 'LINE']

    # Annotate all detections if no zones are defined
    if len(zoneMasks) == 0:
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
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
                logger.warning('Failed to get detections: %s', e)
                continue

            count = zone.current_count
            zone_label = str(count) + ' - ' + saved_mask['label']

            filtered_detections = detections[zone_mask]
            
            # labels = [f"#{tracker_id}" for tracker_id in filtered_detections.tracker_id]

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
            logger.warning('Failed to get line counts: %s', e, exc_info=True)
            # continue

        frame = line_annotator.annotate(frame, line_counter=lineZone)

    
    return frame, zoneCounts, lineCounts