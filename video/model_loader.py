"""Model loading — thin dispatcher that delegates to engine backends.

All backend-specific logic now lives in ``engines/huggingface.py`` and
``engines/mmdet.py``.  This module preserves the public ``getModel()`` API
so that callers (``videoStream.py``, ``model_zoo.py``) need no changes.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from config import StreamConfig
from engines import get_engine, get_engine_for_model

logger = logging.getLogger('model_loader')


def getModel(model_name: str, config: StreamConfig | None = None) -> Dict[str, Any]:
    """Load a model by name, routing to the appropriate engine backend."""
    detect_backend = config.detect_backend if config else os.environ.get('DETECT_BACKEND', 'mmdet')
    backend = detect_backend.lower()

    # TensorRT is an optimisation layer within the mmdet engine.
    if backend == 'tensorrt':
        backend = 'mmdet'

    try:
        engine = get_engine(backend)
    except ValueError:
        # Requested backend unavailable — try to find any engine that supports the model.
        engine = get_engine_for_model(model_name)
        logger.warning('Backend %s unavailable, using %s for %s', detect_backend, engine.name, model_name)

    if not engine.available():
        # Engine registered but not available on this platform — try cross-backend fallback.
        try:
            engine = get_engine_for_model(model_name)
        except ValueError:
            raise RuntimeError(
                f'Backend {backend!r} is not available and no fallback engine '
                f'supports model {model_name!r}.'
            )
        logger.warning('Backend %s not available on this platform, falling back to %s',
                       detect_backend, engine.name)

    # Cross-backend routing: if the engine doesn't know this model, try another.
    if not engine.supports_model(model_name):
        try:
            alt = get_engine_for_model(model_name)
            logger.info('Model %s not in %s zoo, routing to %s', model_name, engine.name, alt.name)
            engine = alt
        except ValueError:
            pass  # let the engine try anyway (e.g. auto-download)

    return engine.load_model(model_name, config)


# ── Backward-compatible aliases (used by model_zoo.build_trt_for_model) ────

def get_huggingface_model(model_name: str, config: StreamConfig | None = None) -> Dict[str, Any]:
    """Load via HuggingFace engine — backward-compat wrapper."""
    return get_engine('huggingface').load_model(model_name, config)


def get_mmdet_model(model_name: str, config: StreamConfig | None = None) -> Dict[str, Any]:
    """Load via MMDet engine — backward-compat wrapper."""
    return get_engine('mmdet').load_model(model_name, config)
