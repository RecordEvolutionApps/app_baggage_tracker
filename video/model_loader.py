"""Model loading — dispatches to TensorRT or MMDet backend."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch

from config import StreamConfig
from model_zoo import (
    MMDET_MODEL_ZOO,
    _CORRUPT_CHECKPOINT_ERRORS,
    _delete_corrupt_checkpoint,
    _download_checkpoint,
    _validate_checkpoint,
    download_file,
)

logger = logging.getLogger('model_loader')

# ── Jetson L4T PyTorch lacks full distributed support ───────────────────────
# Stub out ReduceOp so that mmengine doesn't crash on import.
if not hasattr(torch.distributed, 'ReduceOp'):
    class _ReduceOpStub:
        SUM = 0; PRODUCT = 1; MIN = 2; MAX = 3; BAND = 4; BOR = 5; BXOR = 6
    torch.distributed.ReduceOp = _ReduceOpStub

# ── Monkey-patch mmcv NMS ───────────────────────────────────────────────────
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


# ── Class name helpers ──────────────────────────────────────────────────────

def _maybe_set_class_names_from_model_name(model_name: str, config: StreamConfig | None = None) -> None:
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
                                           config: StreamConfig | None = None) -> None:
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


# ── Model loading dispatch ─────────────────────────────────────────────────

def getModel(model_name: str, config: StreamConfig | None = None) -> Dict[str, Any]:
    """Load a model by name, routing to the appropriate backend."""
    detect_backend = config.detect_backend if config else os.environ.get('DETECT_BACKEND', 'mmdet')
    backend = detect_backend.lower()
    if backend == 'tensorrt':
        return get_tensorrt_model(model_name, config)
    if backend == 'mmdet':
        return get_mmdet_model(model_name, config)
    raise ValueError(f'Unsupported DETECT_BACKEND: {backend}. Supported: mmdet, tensorrt')


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


# ── MMDet backend ──────────────────────────────────────────────────────────

def get_mmdet_model(model_name: str, config: StreamConfig | None = None) -> Dict[str, Any]:
    """Load any MMDetection model by name."""
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
            logger.warning('Corrupt checkpoint detected for %s — deleting and re-downloading', model_name)
            _delete_corrupt_checkpoint(checkpoint_path)
        _download_checkpoint(model_name, checkpoint_path)

    _ensure_checkpoint()

    try:
        from mmdet.apis import DetInferencer
    except Exception as exc:
        raise RuntimeError(
            'MMDetection is not installed. Install mmdet, mmengine, and mmcv for DETECT_BACKEND=mmdet.'
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
            'Checkpoint for %s appears corrupt (%s: %s) — deleting and retrying once',
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


# ── TensorRT backend ──────────────────────────────────────────────────────

def get_tensorrt_model(model_name: str, config: StreamConfig | None = None) -> Dict[str, Any]:
    """Load a model as a TensorRT FP16 engine."""
    try:
        import trt_backend
    except ImportError:
        logger.warning('trt_backend module not found; falling back to mmdet')
        return get_mmdet_model(model_name, config)

    if not trt_backend.is_available():
        logger.warning('TensorRT or CUDA not available; falling back to mmdet backend')
        return get_mmdet_model(model_name, config)

    input_wh = MMDET_MODEL_ZOO.get(model_name, {}).get('native_input_wh', (640, 640))

    if trt_backend.is_engine_cached(model_name):
        try:
            trt_inferencer = trt_backend.TRTInferencer(
                str(trt_backend.engine_path(model_name)), input_wh,
            )
            logger.info('TensorRT engine loaded from cache for %s (FP16)', model_name)
            _maybe_set_class_names_from_model_name(model_name, config)
            return {
                'backend': 'tensorrt',
                'inferencer': trt_inferencer,
                'model_name': model_name,
                'native_input_wh': input_wh,
                'trt_cached': True,
            }
        except Exception as e:
            logger.warning('Cached TRT engine failed to load: %s — rebuilding', e)

    logger.info('No cached TRT engine for %s — using mmdet (TRT build is manual)', model_name)
    return get_mmdet_model(model_name, config)
