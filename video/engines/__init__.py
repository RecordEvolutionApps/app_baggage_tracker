"""Engine registry — discover, register, and look up inference backends."""
from __future__ import annotations

import logging
import os
import platform as _platform
from typing import Dict

import torch

from engines.base import InferenceEngine

logger = logging.getLogger('engines')

# ── Singleton registry ──────────────────────────────────────────────────────
_registry: Dict[str, InferenceEngine] = {}


def register_engine(engine: InferenceEngine) -> None:
    """Register an engine instance under its :attr:`name`."""
    _registry[engine.name] = engine


def get_engine(backend: str) -> InferenceEngine:
    """Look up a registered engine by name, raising ``ValueError`` if unknown."""
    engine = _registry.get(backend)
    if engine is None:
        raise ValueError(
            f'Unknown inference backend: {backend!r}. '
            f'Registered backends: {list(_registry)}'
        )
    return engine


def get_default_engine() -> InferenceEngine:
    """Return the best engine for the current platform."""
    _is_amd64 = _platform.machine() in ('x86_64', 'AMD64')
    if torch.cuda.is_available() and 'mmdet' in _registry and _registry['mmdet'].available():
        return _registry['mmdet']
    if _is_amd64 and 'huggingface' in _registry and _registry['huggingface'].available():
        return _registry['huggingface']
    if 'mmdet' in _registry and _registry['mmdet'].available():
        return _registry['mmdet']
    if 'huggingface' in _registry and _registry['huggingface'].available():
        return _registry['huggingface']
    raise RuntimeError('No inference engine is available on this platform.')


def get_engine_for_model(model_name: str) -> InferenceEngine:
    """Return the first registered engine whose zoo contains *model_name*."""
    for engine in _registry.values():
        if engine.supports_model(model_name):
            return engine
    raise ValueError(
        f'Model {model_name!r} not found in any registered engine zoo. '
        f'Registered: {list(_registry)}'
    )


def list_all_engines() -> Dict[str, InferenceEngine]:
    """Return a copy of the registry."""
    return dict(_registry)


# ── Auto-register built-in engines on import ────────────────────────────────
def _auto_register():
    from engines.huggingface import HuggingFaceEngine
    from engines.mmdet import MMDetEngine

    register_engine(HuggingFaceEngine())
    register_engine(MMDetEngine())


_auto_register()
