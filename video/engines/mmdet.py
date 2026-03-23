"""MMDetection inference engine (with optional TensorRT acceleration)."""
from __future__ import annotations

import logging
import os
import pickle
import platform as _platform
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import supervision as sv
import torch
import time as _time_mod

from engines.base import InferenceEngine

logger = logging.getLogger('engines.mmdet')

# ── Platform detection ──────────────────────────────────────────────────────
_IS_AMD64 = _platform.machine() in ('x86_64', 'AMD64')

try:
    import mmdet as _mmdet  # noqa: F401
    _HAS_MMDET = True
except ImportError:
    _HAS_MMDET = False

# amd64 builds use a modern stack that only supports HuggingFace.
if _IS_AMD64:
    _HAS_MMDET = False

# Exceptions raised by torch.load() for corrupted checkpoint files
_CORRUPT_CHECKPOINT_ERRORS = (RuntimeError, pickle.UnpicklingError, zipfile.BadZipFile, EOFError, OSError)

# Throttle per-frame profile logging.
_PROFILE_LOG_INTERVAL = 5.0
_last_profile_log = 0.0


def _should_log_profile() -> bool:
    global _last_profile_log
    now = _time_mod.monotonic()
    if now - _last_profile_log >= _PROFILE_LOG_INTERVAL:
        _last_profile_log = now
        return True
    return False


# ── Jetson ReduceOp stub ────────────────────────────────────────────────────
if _HAS_MMDET and not hasattr(torch.distributed, 'ReduceOp'):
    class _ReduceOpStub:
        SUM = 0; PRODUCT = 1; MIN = 2; MAX = 3; BAND = 4; BOR = 5; BXOR = 6
    torch.distributed.ReduceOp = _ReduceOpStub

# ── Monkey-patch mmcv NMS ───────────────────────────────────────────────────
def _patch_mmcv_nms():
    if not _HAS_MMDET:
        return
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
        pass

_patch_mmcv_nms()


# ── Model zoo ───────────────────────────────────────────────────────────────

_MMDET_MODEL_ZOO_FULL: Dict[str, Any] = {
    "rtmdet_tiny_8xb32-300e_coco": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
        "native_input_wh": (640, 640),
        "expected_size": 57532893,
        "sha256": "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb",
    },
    "rtmdet_s_8xb32-300e_coco": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
        "native_input_wh": (640, 640),
        "expected_size": 91450098,
        "sha256": "387a891e157cf0ab57d76b3ffc17bf77247089d672532427930b3140f9e789d6",
    },
    "rtmdet_m_8xb32-300e_coco": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
        "native_input_wh": (640, 640),
        "expected_size": 224299609,
        "sha256": "229f527ca88498e8894a778a62a878a322b4a3ea2cae09ea537d34b7e907792b",
    },
}

MMDET_MODEL_ZOO: Dict[str, Any] = {} if _IS_AMD64 else _MMDET_MODEL_ZOO_FULL


# ── Checkpoint helpers ──────────────────────────────────────────────────────

def download_file(url: str, destination: str) -> None:
    """Download a file to *destination* atomically via a .part temp file."""
    import hashlib
    import shutil
    import urllib.request

    logger.info('Downloading %s...', url)
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')

    try:
        response = urllib.request.urlopen(url)
        total = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 256
        with open(part, 'wb') as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

        if total > 0 and downloaded != total:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f'Download size mismatch for {url}: expected {total} bytes, got {downloaded}'
            )

        shutil.move(str(part), str(dest))
        logger.info('Download complete (%d bytes)', downloaded)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def _validate_checkpoint(path: Path, expected_size: int | None = None,
                         sha256_hex: str | None = None) -> bool:
    """Return True if the checkpoint file at *path* passes integrity checks."""
    import hashlib
    if not path.is_file():
        return False

    actual_size = path.stat().st_size

    if expected_size is not None:
        if actual_size != expected_size:
            logger.warning('Checkpoint %s: size mismatch — expected %d, got %d',
                           path.name, expected_size, actual_size)
            return False

    if sha256_hex is not None:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != sha256_hex:
            logger.warning('Checkpoint %s: SHA-256 mismatch — expected %s, got %s',
                           path.name, sha256_hex, h.hexdigest())
            return False

    if expected_size is None and sha256_hex is None:
        if actual_size < 4096:
            logger.warning('Checkpoint %s: suspiciously small (%d bytes)', path.name, actual_size)
            return False

    return True


def _delete_corrupt_checkpoint(path: Path) -> None:
    """Remove a corrupt checkpoint file and log the action."""
    try:
        path.unlink(missing_ok=True)
        logger.warning('Deleted corrupt checkpoint: %s', path)
    except OSError as e:
        logger.error('Failed to delete corrupt checkpoint %s: %s', path, e)


def _download_checkpoint(model_name: str, checkpoint_path: Path) -> None:
    """Download the checkpoint for *model_name* to *checkpoint_path*."""
    zoo = MMDET_MODEL_ZOO.get(model_name)
    if zoo:
        download_file(zoo['checkpoint'], str(checkpoint_path))
        if not _validate_checkpoint(
            checkpoint_path,
            expected_size=zoo.get('expected_size'),
            sha256_hex=zoo.get('sha256'),
        ):
            _delete_corrupt_checkpoint(checkpoint_path)
            raise RuntimeError(f'Downloaded checkpoint for {model_name} failed integrity validation')
        return

    weight_url = _resolve_weight_url(model_name)
    if weight_url:
        download_file(weight_url, str(checkpoint_path))
        if not _validate_checkpoint(checkpoint_path):
            _delete_corrupt_checkpoint(checkpoint_path)
            raise RuntimeError(f'Downloaded checkpoint for {model_name} failed integrity validation')
        return

    logger.info('No download URL found for %s — relying on DetInferencer auto-download', model_name)


def _resolve_weight_url(model_name: str) -> str | None:
    """Look up the checkpoint download URL from openmim's metafile data."""
    if not _HAS_MMDET:
        return None
    try:
        from model_catalog import _patch_packaging_version
        _patch_packaging_version()
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


# ── Class name helpers ──────────────────────────────────────────────────────

def _maybe_set_class_names_from_model_name(model_name: str, config=None) -> None:
    """Set class names from COCO if model is COCO-based (only if not already set)."""
    class_names = config.class_names if config else []
    if class_names:
        return
    if 'coco' not in model_name.lower():
        return
    try:
        from model_catalog import COCO_CLASSES
        names = [str(name) for name in COCO_CLASSES]
        if config:
            config.class_names = names
    except Exception:
        return


def _maybe_set_class_names_from_inferencer(inferencer, model_name: str,
                                           config=None) -> None:
    """Extract class names from MMDet inferencer metadata."""
    class_names = config.class_names if config else []
    if class_names:
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
        _maybe_set_class_names_from_model_name(model_name, config)
        return
    if config:
        config.class_names = [str(name) for name in classes]


# ── Config injection helper ────────────────────────────────────────────────

def _inject_load_image(cfg):
    """Ensure the test pipeline starts with ``LoadImageFromFile``."""
    for key in ('test_pipeline', 'val_pipeline'):
        pipeline = cfg.get(key)
        if not pipeline:
            continue
        names = [t.get('type', '') if isinstance(t, dict) else '' for t in pipeline]
        if 'LoadImageFromFile' not in names:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
            logger.info('Injected LoadImageFromFile into cfg.%s', key)

    try:
        ds_pipeline = cfg.test_dataloader.dataset.pipeline
        names = [t.get('type', '') if isinstance(t, dict) else '' for t in ds_pipeline]
        if 'LoadImageFromFile' not in names:
            ds_pipeline.insert(0, dict(type='LoadImageFromFile'))
            logger.info('Injected LoadImageFromFile into cfg.test_dataloader.dataset.pipeline')
    except (AttributeError, KeyError):
        pass


def _resolve_config_path(model_name: str) -> str | None:
    """Resolve the absolute config file path for a model from openmim metafiles."""
    try:
        import mmdet
        mmdet_root = os.path.dirname(mmdet.__file__)

        from model_catalog import _patch_packaging_version
        _patch_packaging_version()
        from mim.commands.search import get_model_info
        df = get_model_info('mmdet', shown_fields=['config'])
        for _, row in df.iterrows():
            config = row.get('config', '')
            if not config:
                continue
            name = os.path.basename(config).replace('.py', '')
            if name == model_name:
                for prefix in [
                    os.path.join(mmdet_root, '.mim'),
                    mmdet_root,
                ]:
                    abs_path = os.path.join(prefix, config)
                    if os.path.isfile(abs_path):
                        logger.info('Resolved config for %s: %s', model_name, abs_path)
                        return abs_path
                logger.warning('Config file not found on disk for %s: %s', model_name, config)
                return config
    except Exception as e:
        logger.warning('_resolve_config_path failed for %s: %s', model_name, e)
    return None


# ── Engine implementation ───────────────────────────────────────────────────

class MMDetEngine(InferenceEngine):

    @property
    def name(self) -> str:
        return 'mmdet'

    def available(self) -> bool:
        return _HAS_MMDET

    def list_models(self) -> Dict[str, Any]:
        return MMDET_MODEL_ZOO

    def load_model(self, model_name: str, config=None) -> Dict[str, Any]:
        """Load an MMDet model, with optional TensorRT acceleration."""
        detect_backend = config.detect_backend if config else 'mmdet'

        # TensorRT acceleration: try cached engine first
        if detect_backend == 'tensorrt':
            trt_bundle = self._try_load_tensorrt(model_name, config)
            if trt_bundle is not None:
                return trt_bundle
            logger.info('No cached TRT engine for %s — falling back to MMDet PyTorch', model_name)

        return self._load_mmdet(model_name, config)

    def infer(self, frame: np.ndarray, model_bundle: Dict[str, Any],
              confidence: float | None = None, iou: float | None = None,
              config=None) -> Optional[sv.Detections]:
        if model_bundle.get('trt_accelerated'):
            return self._infer_tensorrt(frame, model_bundle, confidence, iou, config)
        return self._infer_mmdet(frame, model_bundle, confidence, config)

    # ── MMDet PyTorch inference ─────────────────────────────────────────────

    def _infer_mmdet(self, frame: np.ndarray, model_bundle: Dict[str, Any],
                     confidence: float | None = None,
                     config=None) -> Optional[sv.Detections]:
        inferencer = model_bundle['inferencer']
        conf_threshold = confidence if confidence is not None else (config.conf if config else 0.1)
        class_list = (config.class_list if config else []) or []
        try:
            _t0 = _time_mod.monotonic()
            results = inferencer(frame, return_vis=False, no_save_pred=True,
                                 print_result=False, return_datasamples=True)
            _t_forward = _time_mod.monotonic()
            predictions = results.get('predictions', [])
            if not predictions:
                return False

            detections = sv.Detections.from_mmdetection(predictions[0])
            if len(detections) == 0:
                return False

            keep = detections.confidence >= conf_threshold
            if class_list:
                keep = np.logical_and(keep, np.isin(detections.class_id, class_list))

            detections = detections[keep]
            _t_post = _time_mod.monotonic()
            if _should_log_profile():
                logger.info('[PROFILE-MMDET] forward=%.0fms  post=%.0fms  total=%.0fms  frame=%dx%d',
                            (_t_forward - _t0) * 1000, (_t_post - _t_forward) * 1000,
                            (_t_post - _t0) * 1000, frame.shape[1], frame.shape[0])
            return detections if len(detections) > 0 else False

        except Exception as e:
            logger.error('Failed to extract detections from MMDetection result: %s', e, exc_info=True)
            return False

    # ── TensorRT accelerated inference ──────────────────────────────────────

    def _infer_tensorrt(self, frame: np.ndarray, model_bundle: Dict[str, Any],
                        confidence: float | None = None, iou: float | None = None,
                        config=None) -> Optional[sv.Detections]:
        trt_inf = model_bundle['inferencer']
        conf = confidence if confidence is not None else (config.conf if config else 0.1)
        iou_threshold = iou if iou is not None else (config.nms_iou if config else 0.5)
        class_list = (config.class_list if config else []) or None
        try:
            _t0 = _time_mod.monotonic()
            result = trt_inf(frame, conf=conf, iou=iou_threshold, class_list=class_list)
            _dt = _time_mod.monotonic() - _t0
            if _should_log_profile():
                logger.info('[PROFILE-TRT] total=%.0fms  frame=%dx%d',
                            _dt * 1000, frame.shape[1], frame.shape[0])
            return result if result is not None else False
        except Exception as e:
            logger.error('TensorRT inference failed: %s', e, exc_info=True)
            return False

    # ── Model loading helpers ───────────────────────────────────────────────

    def _load_mmdet(self, model_name: str, config=None) -> Dict[str, Any]:
        """Load a pure MMDetection PyTorch model."""
        if not _HAS_MMDET:
            raise RuntimeError(
                'MMDetection is not installed. Install mmdet, mmengine, and mmcv.'
            )
        cache_root = Path('/data/mmdet')
        checkpoint_path = cache_root / 'checkpoints' / f'{model_name}.pth'
        zoo = MMDET_MODEL_ZOO.get(model_name, {})

        def _ensure_checkpoint():
            if checkpoint_path.is_file():
                if _validate_checkpoint(
                    checkpoint_path,
                    expected_size=zoo.get('expected_size'),
                    sha256_hex=zoo.get('sha256'),
                ):
                    return
                logger.warning('Corrupt checkpoint for %s — re-downloading', model_name)
                _delete_corrupt_checkpoint(checkpoint_path)
            _download_checkpoint(model_name, checkpoint_path)

        _ensure_checkpoint()

        try:
            from mmdet.apis import DetInferencer
        except Exception as exc:
            raise RuntimeError(
                'MMDetection is not installed. Install mmdet, mmengine, and mmcv.'
            ) from exc

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        def _try_load(model_or_cfg, weights=None):
            kwargs = dict(model=model_or_cfg, device=device, show_progress=False)
            if weights:
                kwargs['weights'] = weights
            try:
                return DetInferencer(**kwargs)
            except ValueError as exc:
                if 'LoadImageFromFile is not found' not in str(exc):
                    raise
                logger.info('Patching missing LoadImageFromFile for %s', model_or_cfg)
                from mmengine.config import Config
                cfg_path = model_or_cfg
                if not os.path.isfile(cfg_path):
                    cfg_path = _resolve_config_path(model_name)
                if not cfg_path:
                    raise
                cfg = Config.fromfile(cfg_path)
                _inject_load_image(cfg)
                kwargs['model'] = cfg
                return DetInferencer(**kwargs)

        def _load_inferencer():
            if checkpoint_path.is_file():
                try:
                    return _try_load(model_name, weights=str(checkpoint_path))
                except ValueError:
                    config_path = _resolve_config_path(model_name)
                    if config_path:
                        logger.info('Loading %s via config path: %s', model_name, config_path)
                        return _try_load(config_path, weights=str(checkpoint_path))
                    raise
            else:
                try:
                    logger.info('No cached checkpoint for "%s", DetInferencer will auto-download...', model_name)
                    return _try_load(model_name)
                except ValueError:
                    config_path = _resolve_config_path(model_name)
                    if config_path:
                        logger.info('Loading %s via config path (auto-download): %s', model_name, config_path)
                        return _try_load(config_path)
                    raise

        try:
            inferencer = _load_inferencer()
        except _CORRUPT_CHECKPOINT_ERRORS as exc:
            logger.warning(
                'Checkpoint for %s appears corrupt (%s: %s) — deleting and retrying',
                model_name, type(exc).__name__, exc,
            )
            _delete_corrupt_checkpoint(checkpoint_path)
            _ensure_checkpoint()
            inferencer = _load_inferencer()

        _maybe_set_class_names_from_inferencer(inferencer, model_name, config)
        native_wh = MMDET_MODEL_ZOO.get(model_name, {}).get('native_input_wh', (640, 640))
        return {
            'backend': 'mmdet',
            'inferencer': inferencer,
            'model_name': model_name,
            'native_input_wh': native_wh,
        }

    def _try_load_tensorrt(self, model_name: str, config=None) -> Dict[str, Any] | None:
        """Try to load a cached TensorRT engine; return ``None`` on failure."""
        try:
            import trt_backend
        except ImportError:
            return None

        if not trt_backend.is_available():
            return None

        input_wh = MMDET_MODEL_ZOO.get(model_name, {}).get('native_input_wh', (640, 640))

        if trt_backend.is_engine_cached(model_name):
            try:
                trt_inferencer = trt_backend.TRTInferencer(
                    str(trt_backend.engine_path(model_name)), input_wh,
                )
                logger.info('TensorRT engine loaded from cache for %s (FP16)', model_name)
                _maybe_set_class_names_from_model_name(model_name, config)
                return {
                    'backend': 'mmdet',
                    'trt_accelerated': True,
                    'inferencer': trt_inferencer,
                    'model_name': model_name,
                    'native_input_wh': input_wh,
                    'trt_cached': True,
                }
            except Exception as e:
                logger.warning('Cached TRT engine failed to load: %s — rebuilding', e)

        return None
