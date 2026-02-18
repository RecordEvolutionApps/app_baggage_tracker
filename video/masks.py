"""Mask parsing, polygon/line zone preparation, and zone counting."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import supervision as sv

logger = logging.getLogger('masks')


def get_contrast_color(hex_color: str) -> sv.Color:
    """Return black or white for text contrast on a given background colour."""
    c = sv.Color.from_hex(hex_color)
    luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
    if luminance < 128:
        return sv.Color(255, 255, 255)  # White for dark colours
    else:
        return sv.Color(0, 0, 0)  # Black for light colours


def prepMasks(in_masks: dict, resolution_x: int, resolution_y: int) -> list[dict]:
    """Parse raw mask JSON into supervision PolygonZone/LineZone objects with annotators.

    Parameters
    ----------
    in_masks : dict
        The raw mask JSON (``{ "polygons": [...] }``).
    resolution_x, resolution_y : int
        The current video frame resolution — needed for ``PolygonZone``.
    """
    pre_masks = [
        {
            'label': mask['label'],
            'type': mask.get('type', 'ZONE'),
            'points': [(int(point['x']), int(point['y'])) for point in mask['points'][:-1]],
            'color': mask['lineColor'],
        }
        for mask in in_masks['polygons']
    ]
    out_masks: list[dict] = []

    for mask in pre_masks:
        polygon = np.array(mask['points'])
        polygon.astype(int)

        if mask['type'] == 'ZONE':
            polygon_zone = sv.PolygonZone(
                polygon=polygon,
                frame_resolution_wh=(resolution_x, resolution_y),
                triggering_position=sv.Position.CENTER,
            )
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
                color=sv.Color.from_hex(mask['color']),
            )
            mask['annotator'] = line_annotator

        out_masks.append(mask)

    logger.info('[masks] Refreshed %d mask(s)', len(out_masks))
    return out_masks


def count_polygon_zone(zone, class_list: list[int]) -> dict:
    """Return per-class counts within a polygon zone."""
    count_dict: dict[int, int] = {}
    for class_id in class_list:
        count = zone.class_in_current_count.get(class_id, 0)
        count_dict[class_id] = count
    return count_dict


def count_detections(detections: sv.Detections) -> dict:
    """Return per-class counts from raw detections."""
    count_dict: dict = {}
    try:
        for xyxy, mask, conf, class_id, tracker_id, data in detections:
            if class_id in count_dict:
                count_dict[class_id] += 1
            else:
                count_dict[class_id] = 1
    except Exception as e:
        logger.warning('Failed to count: %s', e)
    return count_dict
