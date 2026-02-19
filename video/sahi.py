"""SAHI (Slicing Aided Hyper Inference) helpers — slicer creation and geometry."""
from __future__ import annotations

import logging
import time as _time_mod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import supervision as sv

from config import StreamConfig
from inference import infer_frame, empty_detections, move_detections

logger = logging.getLogger('sahi')


@dataclass
class SahiGridInfo:
    """Metadata needed to draw the SAHI grid overlay."""
    rect: tuple[int, int, int, int]
    slice_wh: tuple[int, int]
    overlap_wh: tuple[int, int]


def get_extreme_points(masks: list, frame_buffer: int = 64,
                       frame_wh: tuple[int, int] | None = None,
                       config: StreamConfig | None = None) -> tuple[int, int, int, int]:
    """Return (low_x, low_y, high_x, high_y) bounding all mask points + buffer.

    *frame_wh*: optional ``(width, height)`` of the actual video frame.
    Falls back to ``config.resolution_x/y`` when not provided.
    """
    if frame_wh:
        res_x, res_y = frame_wh
    elif config:
        res_x, res_y = config.resolution_x, config.resolution_y
    else:
        res_x, res_y = 640, 480

    if len(masks) == 0:
        return 0, 0, res_x, res_y
    low_x = res_x - 1
    low_y = res_y - 1
    high_x = -1
    high_y = -1
    for mask in masks:
        points = np.array(mask["points"], dtype=np.int32)
        for point in points:
            low_x = point[0] if point[0] < low_x else low_x
            low_y = point[1] if point[1] < low_y else low_y
            high_x = point[0] if point[0] > high_x else high_x
            high_y = point[1] if point[1] > high_y else high_y

    low_x = max(0, low_x - frame_buffer)
    low_y = max(0, low_y - frame_buffer)
    high_x = min(res_x - 1, high_x + frame_buffer)
    high_y = min(res_y - 1, high_y + frame_buffer)

    # Ensure valid bounds (low < high)
    if low_x >= high_x or low_y >= high_y:
        return 0, 0, res_x, res_y

    return low_x, low_y, high_x, high_y


def initSliceInferer(model_bundle: Dict[str, Any],
                     settings_dict: dict | None = None,
                     config: StreamConfig | None = None):
    """Create an ``sv.InferenceSlicer`` using the model's native input size."""
    native_w, native_h = model_bundle.get('native_input_wh', (640, 640))
    conf_default = config.conf if config else 0.1
    sahi_iou_default = config.sahi_iou if config else 0.5
    slice_count = [0]  # mutable counter for logging

    def inferSlice(image_slice: np.ndarray) -> Optional[sv.Detections]:
        import time as _time
        slice_count[0] += 1
        t0 = _time.monotonic()
        conf = float(settings_dict.get('confidence', conf_default)) if settings_dict else conf_default
        detections = infer_frame(image_slice, model_bundle, confidence=conf, config=config)
        dt = _time.monotonic() - t0
        det_count = len(detections) if detections and detections is not False else 0
        logger.debug('[SAHI] slice #%d (%dx%d) inferred in %.1fms → %d detections',
                     slice_count[0], image_slice.shape[1], image_slice.shape[0], dt * 1000, det_count)
        return detections if detections is not False else empty_detections()

    sahi_iou = float(settings_dict.get('sahiIou', sahi_iou_default)) if settings_dict else sahi_iou_default
    overlap_ratio = float(settings_dict.get('overlapRatio', 0.2)) if settings_dict else 0.2
    overlap_w = int(overlap_ratio * native_w)
    overlap_h = int(overlap_ratio * native_h)
    slicer = sv.InferenceSlicer(
        callback=inferSlice,
        slice_wh=(native_w, native_h),
        overlap_wh=(overlap_w, overlap_h),
        iou_threshold=sahi_iou,
        thread_workers=6
    )
    slicer._sahi_slice_count = slice_count
    slicer._sahi_slice_wh = (native_w, native_h)
    slicer._sahi_overlap_wh = (overlap_w, overlap_h)
    slicer._sahi_iou = sahi_iou
    slicer._sahi_overlap_ratio = overlap_ratio
    logger.debug('[SAHI] InferenceSlicer created: slice=%dx%d, overlap=%dx%d (%.0f%%), sahiIou=%.2f',
                native_w, native_h, overlap_w, overlap_h, overlap_ratio * 100, sahi_iou)
    return slicer


def run_sahi_inference(frame, slicer, model, saved_masks, stream_settings, config):
    """Manage slicer lifecycle and run SAHI tiled inference on *frame*.

    * Re-creates the slicer when SAHI IoU or overlap ratio change.
    * Computes the crop region from mask extreme points.
    * Runs the slicer on the cropped region and offsets detections back.

    Returns ``(detections, updated_slicer, SahiGridInfo | None)``.
    """
    sahi_iou = float(stream_settings.get('sahiIou', config.sahi_iou))
    overlap_ratio = float(stream_settings.get('overlapRatio', 0.2))

    # Re-create slicer when params drift
    if (slicer is None
            or getattr(slicer, '_sahi_iou', None) != sahi_iou
            or getattr(slicer, '_sahi_overlap_ratio', None) != overlap_ratio):
        slicer = initSliceInferer(model, stream_settings, config)

    # Crop region from mask bounds
    frame_buf = int(stream_settings.get('frameBuffer', config.frame_buffer))
    low_x, low_y, high_x, high_y = get_extreme_points(
        saved_masks, frame_buf,
        frame_wh=(config.resolution_x, config.resolution_y),
    )
    crop_w, crop_h = high_x - low_x, high_y - low_y
    logger.debug('[SAHI] crop=(%d,%d)-(%d,%d) size=%dx%d, masks=%d',
                 low_x, low_y, high_x, high_y, crop_w, crop_h, len(saved_masks))

    # Run slicer on the cropped region
    frame_ = frame[low_y:high_y, low_x:high_x]
    t0 = _time_mod.monotonic()
    if hasattr(slicer, '_sahi_slice_count'):
        slicer._sahi_slice_count[0] = 0
    detections = slicer(frame_)
    dt = _time_mod.monotonic() - t0
    det_count = len(detections) if detections else 0
    n_slices = slicer._sahi_slice_count[0] if hasattr(slicer, '_sahi_slice_count') else '?'
    logger.debug('[SAHI] %s slices in %.1fms, %d detections (pre-offset)',
                 n_slices, dt * 1000, det_count)

    # Offset detections back to full-frame coordinates
    detections = move_detections(detections, low_x, low_y)

    # Build grid info for the overlay
    grid_info = None
    slice_wh = getattr(slicer, '_sahi_slice_wh', None)
    overlap_wh = getattr(slicer, '_sahi_overlap_wh', None)
    if slice_wh and overlap_wh:
        grid_info = SahiGridInfo(
            rect=(low_x, low_y, high_x, high_y),
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
        )

    return detections, slicer, grid_info
