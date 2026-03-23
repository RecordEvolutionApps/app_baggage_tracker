"""Inference dispatch — routes frames to the appropriate engine backend.

This module is a thin dispatcher.  All backend-specific logic lives in
``engines/huggingface.py`` and ``engines/mmdet.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import supervision as sv

from config import StreamConfig
from engines import get_engine

logger = logging.getLogger('inference')


def empty_detections() -> sv.Detections:
    """Return an empty ``sv.Detections`` object."""
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int64),
    )


def infer(frame: np.ndarray, model_bundle: Dict[str, Any],
          confidence: float | None = None, iou: float | None = None,
          config: StreamConfig | None = None) -> Optional[sv.Detections]:
    """Top-level inference entry point — delegates to the correct engine."""
    return infer_frame(frame, model_bundle, confidence=confidence, iou=iou, config=config)


def infer_frame(frame: np.ndarray, model_bundle: Dict[str, Any],
                confidence: float | None = None, iou: float | None = None,
                config: StreamConfig | None = None) -> Optional[sv.Detections]:
    """Route inference to the correct engine backend."""
    backend = model_bundle.get('backend')
    engine = get_engine(backend)
    return engine.infer(frame, model_bundle, confidence=confidence, iou=iou, config=config)


def move_detections(detections: sv.Detections, offset_x: int, offset_y: int) -> sv.Detections:
    """Offset bounding boxes after SAHI crop-region inference."""
    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        box[0] += offset_x  # xmin
        box[1] += offset_y  # ymin
        box[2] += offset_x  # xmax
        box[3] += offset_y  # ymax
    return detections
