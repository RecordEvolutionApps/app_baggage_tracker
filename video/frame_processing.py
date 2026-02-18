"""Frame processing — tracking, smoothing, and annotation of detections."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import supervision as sv

from config import StreamConfig
from masks import count_detections, count_polygon_zone

logger = logging.getLogger('frame_processing')


# ── Supervision annotators & tracker (module-level singletons) ──────────────

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


# ── Label helpers ───────────────────────────────────────────────────────────

def _class_id_to_name(class_id: int | None, config: StreamConfig | None = None) -> str:
    """Resolve a class ID to a human-readable name."""
    if class_id is None:
        return ''
    try:
        class_id_int = int(class_id)
    except Exception:
        return str(class_id)
    class_names = config.class_names if config else []
    if class_names and 0 <= class_id_int < len(class_names):
        name = class_names[class_id_int]
        if name is not None and str(name).strip() != '':
            return str(name)
    return str(class_id_int)


def _build_labels(detections: sv.Detections, config: StreamConfig | None = None) -> list[str]:
    """Build label list for annotation."""
    if detections.class_id is None:
        return []
    return [_class_id_to_name(class_id, config) for class_id in detections.class_id]


# ── Main frame processing ──────────────────────────────────────────────────

def processFrame(frame, detections: sv.Detections, saved_masks: list,
                 config: StreamConfig | None = None,
                 settings_dict: dict | None = None):
    """Entry point — delegates to ``_processFrameInner`` with numpy warning suppression."""
    with np.errstate(all='ignore'):
        return _processFrameInner(frame, detections, saved_masks, config, settings_dict)


def _processFrameInner(frame, detections: sv.Detections, saved_masks: list,
                       config: StreamConfig | None = None,
                       settings_dict: dict | None = None):
    """Track, smooth, annotate zones/lines, return frame + zone/line counts."""
    smoothing_default = config.smoothing if config else True
    device_name = config.device_name if config else 'UNKNOWN_DEVICE'
    class_list = config.class_list if config else []

    try:
        has_masks = detections.mask is not None
        detections = tracker.update_with_detections(detections)
        use_smoothing = settings_dict.get('useSmoothing', smoothing_default) if settings_dict else smoothing_default
        if use_smoothing and not has_masks:
            detections = smoother.update_with_detections(detections)
    except Exception as e:
        logger.warning('Error when smoothing detections or updating tracker: %s', e, exc_info=True)

    zoneCounts: list[dict] = []
    lineCounts: list[dict] = []

    zoneMasks = [m for m in saved_masks if m['type'] == 'ZONE']
    lineMasks = [m for m in saved_masks if m['type'] == 'LINE']

    # Annotate all detections if no zones are defined
    if len(zoneMasks) == 0:
        if detections.mask is not None:
            frame = mask_annotator.annotate(scene=frame, detections=detections)
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        labels = _build_labels(detections, config)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        count_dict = count_detections(detections)
        zoneCounts.append({'label': device_name + "_" + "default", 'count': count_dict})
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

            count_dict = count_polygon_zone(zone, class_list)
            zoneCounts.append({'label': saved_mask['label'], 'count': count_dict})

            if filtered_detections.mask is not None:
                frame = mask_annotator.annotate(scene=frame, detections=filtered_detections)
            frame = bounding_box_annotator.annotate(scene=frame, detections=filtered_detections)
            labels = _build_labels(filtered_detections, config)
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

        frame = line_annotator.annotate(frame, line_counter=lineZone)

    return frame, zoneCounts, lineCounts
