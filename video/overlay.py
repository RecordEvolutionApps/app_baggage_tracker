"""Drawing utilities for video frame overlays."""
from __future__ import annotations

import cv2
import numpy as np


def overlay_text(frame, text, position=(10, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    """Draw text on a video frame using cv2.putText."""
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
    """Draw the SAHI slicing grid overlay on a frame."""
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
