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
# Auto-select TensorRT on CUDA-capable devices (Jetson), fall back to mmdet on CPU
_default_backend = 'tensorrt' if torch.cuda.is_available() else 'mmdet'
DETECT_BACKEND = os.environ.get('DETECT_BACKEND', _default_backend)
RESOLUTION_X = int(os.environ.get('RESOLUTION_X', 640))
RESOLUTION_Y = int(os.environ.get('RESOLUTION_Y', 480))
DEVICE_NAME = os.environ.get('DEVICE_NAME', 'UNKNOWN_DEVICE')
CONF = float(os.environ.get('CONF', '0.1'))
NMS_IOU = float(os.environ.get('NMS_IOU', '0.5'))
SAHI_IOU = float(os.environ.get('SAHI_IOU', '0.5'))
# Keep legacy IOU as alias for backward compatibility
IOU = NMS_IOU
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
mask_annotator = sv.MaskAnnotator(opacity=0.4)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_padding=4,
    text_color=sv.Color.BLACK,
)

tracker = sv.ByteTrack(minimum_consecutive_frames=1)
smoother = sv.DetectionsSmoother(length=5)

def _maybe_set_class_names_from_model_name(model_name: str) -> None:
    global CLASS_NAMES
    if CLASS_NAMES:
        return
    if 'coco' not in model_name.lower():
        return
    try:
        from model_catalog import COCO_CLASSES
        CLASS_NAMES = [str(name) for name in COCO_CLASSES]
    except Exception:
        return

def _maybe_set_class_names_from_inferencer(inferencer, model_name: str) -> None:
    global CLASS_NAMES
    if CLASS_NAMES:
        return
    classes = None
    try:
        classes = inferencer.model.dataset_meta.get('classes')
    except Exception:
        classes = None
    if not classes:
        try:
            classes = inferencer.dataset_meta.get('classes')
        except Exception:
            classes = None
    if not classes:
        _maybe_set_class_names_from_model_name(model_name)
        return
    CLASS_NAMES = [str(name) for name in classes]

def _class_id_to_name(class_id: int | None) -> str:
    if class_id is None:
        return ''
    try:
        class_id_int = int(class_id)
    except Exception:
        return str(class_id)
    if CLASS_NAMES and 0 <= class_id_int < len(CLASS_NAMES):
        name = CLASS_NAMES[class_id_int]
        if name is not None and str(name).strip() != '':
            return str(name)
    return str(class_id_int)

def _build_labels(detections: sv.Detections) -> list[str]:
    if detections.class_id is None:
        return []
    return [_class_id_to_name(class_id) for class_id in detections.class_id]

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

    _maybe_set_class_names_from_inferencer(inferencer, model_name)
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
            _maybe_set_class_names_from_model_name(model_name)
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


def get_youtube_video(url, height=None):
    """Resolve a YouTube URL into a direct playable stream URL.

    Uses the standalone yt-dlp binary (/usr/local/bin/yt-dlp) which bundles its
    own Python and stays up-to-date with YouTube's nsig extraction changes,
    independent of the system Python 3.10 constraint.

    Falls back to the yt_dlp Python library if the binary is not available.
    """
    import shutil
    import subprocess
    import json as _json

    cookie_file = '/data/cookies.txt'

    # Prefer muxed format so OpenCV can open it with a plain HTTP GET.
    if height and height > 0:
        format_str = f'best[height<={height}]/best'
    else:
        format_str = 'best'

    logger.info('[yt-dlp] Resolving: %s (format=%s)', url, format_str)

    yt_dlp_bin = shutil.which('yt-dlp')
    if yt_dlp_bin is None:
        logger.warning('[yt-dlp] Standalone binary not found — falling back to Python yt_dlp library')
        return _get_youtube_video_lib(url, height)

    # Build the command:
    #   -f <format>       : format selector
    #   -g                : print the resolved stream URL to stdout
    #   --print %(width)sx%(height)s : print WIDTHxHEIGHT on a second line
    #   --no-warnings     : keep stderr clean
    #   --no-playlist     : single video only
    cmd = [
        yt_dlp_bin,
        '-f', format_str,
        '--print', '%(url)s',                   # line 1: stream URL
        '--print', '%(width)sx%(height)s',      # line 2: WIDTHxHEIGHT
        '--no-warnings',
        '--no-playlist',
    ]

    if os.path.isfile(cookie_file):
        logger.info('[yt-dlp] Using cookie file: %s', cookie_file)
        cmd += ['--cookies', cookie_file]

    cmd.append(url)

    logger.info('[yt-dlp] Running: %s', ' '.join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.error('[yt-dlp] Timed out after 30 s')
        return {'url': url, 'width': 0, 'height': 0}

    if result.returncode != 0:
        logger.error('[yt-dlp] Failed (exit %d): %s', result.returncode, result.stderr.strip())
        # Fall back to the Python library in case the binary is outdated too
        return _get_youtube_video_lib(url, height)

    lines = result.stdout.strip().splitlines()
    stream_url = lines[0] if len(lines) >= 1 else ''
    resolution_str = lines[1] if len(lines) >= 2 else '0x0'

    width, height_val = 0, 0
    try:
        parts = resolution_str.split('x')
        width = int(parts[0]) if parts[0] not in ('NA', 'None', '') else 0
        height_val = int(parts[1]) if len(parts) > 1 and parts[1] not in ('NA', 'None', '') else 0
    except (ValueError, IndexError):
        pass

    if stream_url:
        logger.info('[yt-dlp] Resolved %dx%d — URL starts with: %s',
                     width, height_val, stream_url[:120])
    else:
        logger.error('[yt-dlp] No URL in stdout: %s', result.stdout[:200])

    return {
        'url': stream_url or url,
        'width': width,
        'height': height_val,
    }


def _get_youtube_video_lib(url, height=None):
    """Fallback: use the yt_dlp Python library (may have stale nsig extraction)."""
    import yt_dlp

    if height and height > 0:
        format_str = f'best[height<={height}]/best'
    else:
        format_str = 'best'

    cookie_file = '/data/cookies.txt'

    ydl_opts = {
        'format': format_str,
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['mweb', 'android'],
            },
        },
    }

    if os.path.isfile(cookie_file):
        ydl_opts['cookiefile'] = cookie_file

    logger.info('[yt_dlp-lib] Resolving: %s (format=%s)', url, format_str)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        stream_url = info.get('url', '')

        if not stream_url:
            for fmt in info.get('requested_formats', []):
                if fmt.get('vcodec', 'none') != 'none' and fmt.get('url'):
                    stream_url = fmt['url']
                    break

        if not stream_url:
            for fmt in reversed(info.get('formats', [])):
                if (fmt.get('vcodec', 'none') != 'none' and
                    fmt.get('acodec', 'none') != 'none' and
                    fmt.get('url')):
                    stream_url = fmt['url']
                    if fmt.get('width'):
                        info['width'] = fmt['width']
                    if fmt.get('height'):
                        info['height'] = fmt['height']
                    break

        if stream_url:
            logger.info('[yt_dlp-lib] Resolved — URL len=%d', len(stream_url))
        else:
            logger.error('[yt_dlp-lib] Could not find any playable URL')

        return {
            'url': stream_url or url,
            'width': info.get('width', 0),
            'height': info.get('height', 0),
        }

# Function to display frame rate and timestamp on the frame
def overlay_text(frame, text, position=(10, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_sahi_grid(
    frame: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    slice_wh: tuple[int, int],
    overlap_wh: tuple[int, int],
    line_color: tuple[int, int, int] = (160, 160, 160),
    overlap_color: tuple[int, int, int] = (120, 120, 120),
    line_thickness: int = 1,
    overlay_alpha: float = 0.25,
) -> np.ndarray:
    low_x, low_y, high_x, high_y = crop_rect
    if low_x >= high_x or low_y >= high_y:
        return frame

    slice_w, slice_h = slice_wh
    overlap_w, overlap_h = overlap_wh
    step_x = max(1, slice_w - overlap_w)
    step_y = max(1, slice_h - overlap_h)

    draw_high_x = max(low_x, high_x - 1)
    draw_high_y = max(low_y, high_y - 1)
    limit_x = draw_high_x + 1
    limit_y = draw_high_y + 1

    if overlay_alpha > 0:
        overlay = frame.copy()
        for y in range(low_y, limit_y, step_y):
            for x in range(low_x, limit_x, step_x):
                x2 = min(x + slice_w, limit_x)
                y2 = min(y + slice_h, limit_y)
                x2_draw = max(x, x2 - 1)
                y2_draw = max(y, y2 - 1)

                if overlap_w > 0 and x2 < limit_x:
                    ox1 = max(x2 - overlap_w, low_x)
                    ox2 = min(x2, limit_x) - 1
                    if ox2 >= ox1:
                        cv2.rectangle(overlay, (ox1, y), (ox2, y2_draw), overlap_color, -1)

                if overlap_h > 0 and y2 < limit_y:
                    oy1 = max(y2 - overlap_h, low_y)
                    oy2 = min(y2, limit_y) - 1
                    if oy2 >= oy1:
                        cv2.rectangle(overlay, (x, oy1), (x2_draw, oy2), overlap_color, -1)

                if overlap_w > 0 and overlap_h > 0 and x2 < limit_x and y2 < limit_y:
                    ox1 = max(x2 - overlap_w, low_x)
                    oy1 = max(y2 - overlap_h, low_y)
                    ox2 = min(x2, limit_x) - 1
                    oy2 = min(y2, limit_y) - 1
                    if ox2 >= ox1 and oy2 >= oy1:
                        cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), overlap_color, -1)

        frame = cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0)

    for y in range(low_y, limit_y, step_y):
        for x in range(low_x, limit_x, step_x):
            x2 = min(x + slice_w, limit_x)
            y2 = min(y + slice_h, limit_y)
            x2_draw = max(x, x2 - 1)
            y2_draw = max(y, y2 - 1)
            cv2.rectangle(frame, (x, y), (x2_draw, y2_draw), line_color, line_thickness)

    cv2.rectangle(frame, (low_x, low_y), (draw_high_x, draw_high_y), line_color, line_thickness)
    return frame

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
        points = np.array(mask["points"], dtype=np.int32)
        for point in points:
            low_x = point[0] if point[0] < low_x else low_x
            low_y = point[1] if point[1] < low_y else low_y
            high_x = point[0] if point[0] > high_x else high_x
            high_y = point[1] if point[1] > high_y else high_y

    low_x  = max(0, low_x - frame_buffer)
    low_y  = max(0, low_y - frame_buffer)
    high_x = min(RESOLUTION_X - 1, high_x + frame_buffer)
    high_y = min(RESOLUTION_Y - 1, high_y + frame_buffer)

    # Ensure valid bounds (low < high) — avoids zero-size crops
    if low_x >= high_x or low_y >= high_y:
        return 0, 0, RESOLUTION_X, RESOLUTION_Y

    return low_x, low_y, high_x, high_y


def initSliceInferer(model_bundle: Dict[str, Any], settings_dict=None):
    native_w, native_h = model_bundle.get('native_input_wh', (640, 640))
    slice_count = [0]  # mutable counter for logging

    def inferSlice(image_slice: np.ndarray) -> Optional[sv.Detections]:
        import time as _time
        slice_count[0] += 1
        t0 = _time.monotonic()
        conf = float(settings_dict.get('confidence', CONF)) if settings_dict else CONF
        detections = infer_frame(image_slice, model_bundle, confidence=conf)
        dt = _time.monotonic() - t0
        det_count = len(detections) if detections and detections is not False else 0
        logger.debug('[SAHI] slice #%d (%dx%d) inferred in %.1fms → %d detections',
                     slice_count[0], image_slice.shape[1], image_slice.shape[0], dt * 1000, det_count)
        return detections if detections is not False else empty_detections()

    sahi_iou = float(settings_dict.get('sahiIou', SAHI_IOU)) if settings_dict else SAHI_IOU
    overlap_ratio = float(settings_dict.get('overlapRatio', 0.2)) if settings_dict else 0.2
    overlap_w = int(overlap_ratio * native_w)
    overlap_h = int(overlap_ratio * native_h)
    slicer = sv.InferenceSlicer(
        callback=inferSlice,
        slice_wh=(native_w, native_h),
        overlap_ratio_wh=None,
        overlap_wh=(overlap_w, overlap_h),
        iou_threshold=sahi_iou,
        thread_workers=6
    )
    slicer._sahi_slice_count = slice_count  # expose for reset
    slicer._sahi_slice_wh = (native_w, native_h)
    slicer._sahi_overlap_wh = (overlap_w, overlap_h)
    slicer._sahi_iou = sahi_iou
    slicer._sahi_overlap_ratio = overlap_ratio
    logger.debug('[SAHI] InferenceSlicer created: slice=%dx%d, overlap=%dx%d (%.0f%%), sahiIou=%.2f',
                native_w, native_h, overlap_w, overlap_h, overlap_ratio * 100, sahi_iou)
    return slicer

def infer(frame, model_bundle, confidence=None, iou=None):
    return infer_frame(frame, model_bundle, confidence=confidence, iou=iou)

def infer_frame(frame: np.ndarray, model_bundle: Dict[str, Any], confidence=None, iou=None):
    backend = model_bundle.get('backend')
    if backend == 'tensorrt':
        return infer_tensorrt(frame, model_bundle, confidence=confidence, iou=iou)
    if backend == 'mmdet':
        return infer_mmdet(frame, model_bundle['inferencer'], confidence=confidence)
    raise ValueError(f'Unsupported backend in model bundle: {backend}')

def infer_mmdet(frame: np.ndarray, inferencer, confidence=None) -> Optional[sv.Detections]:
    try:
        # return_datasamples=True gives us the raw DetDataSample which
        # sv.Detections.from_mmdetection() can convert directly — including
        # instance masks (RLE or tensor) without manual decoding.
        results = inferencer(frame, return_vis=False, no_save_pred=True,
                             print_result=False, return_datasamples=True)
        predictions = results.get('predictions', [])
        if not predictions:
            return False

        detections = sv.Detections.from_mmdetection(predictions[0])
        if len(detections) == 0:
            return False

        # Apply confidence + class filtering
        conf_threshold = confidence if confidence is not None else CONF
        keep = detections.confidence >= conf_threshold
        if CLASS_LIST:
            keep = np.logical_and(keep, np.isin(detections.class_id, CLASS_LIST))

        detections = detections[keep]
        return detections if len(detections) > 0 else False

    except Exception as e:
        logger.error('Failed to extract detections from MMDetection result: %s', e, exc_info=True)
        return False


def infer_tensorrt(frame: np.ndarray, model_bundle: Dict[str, Any], confidence=None, iou=None) -> Optional[sv.Detections]:
    """Run inference through the TensorRT engine."""
    trt_inf = model_bundle['inferencer']
    conf = confidence if confidence is not None else CONF
    iou_threshold = iou if iou is not None else NMS_IOU
    class_list = CLASS_LIST if CLASS_LIST else None
    try:
        result = trt_inf(frame, conf=conf, iou=iou_threshold, class_list=class_list)
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

def processFrame(frame, detections, saved_masks, settings_dict=None):

    # Suppress harmless numpy warnings from supervision's ByteTrack / zone
    # triggers when detections have edge-case coordinates (zero-area boxes,
    # NaN from empty slices, etc.).  These are non-fatal floating-point noise.
    with np.errstate(all='ignore'):
        return _processFrameInner(frame, detections, saved_masks, settings_dict)

def _processFrameInner(frame, detections, saved_masks, settings_dict=None):

    try:
        has_masks = detections.mask is not None
        detections = tracker.update_with_detections(detections)
        use_smoothing = settings_dict.get('useSmoothing', SMOOTHING) if settings_dict else SMOOTHING
        # DetectionsSmoother is incompatible with segmentation masks —
        # it can drop or corrupt them.  Skip smoothing when masks are present.
        if use_smoothing and not has_masks:
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
        if detections.mask is not None:
            frame = mask_annotator.annotate(scene=frame, detections=detections)
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        labels = _build_labels(detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

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

            if filtered_detections.mask is not None:
                frame = mask_annotator.annotate(scene=frame, detections=filtered_detections)
            frame = bounding_box_annotator.annotate(scene=frame, detections=filtered_detections)
            labels = _build_labels(filtered_detections)
            frame = label_annotator.annotate(scene=frame, detections=filtered_detections, labels=labels)
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