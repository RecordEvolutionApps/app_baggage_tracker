"""Centralized configuration and mutable runtime state for a video stream.

All cross-module globals (RESOLUTION_X/Y, CLASS_LIST, CLASS_NAMES, thresholds)
are gathered into a single ``StreamConfig`` dataclass so that state flows
explicitly through function arguments rather than via module-level mutation.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger('config')


# ── StreamConfig ────────────────────────────────────────────────────────────

@dataclass
class StreamConfig:
    """Holds all mutable per-stream state that used to live in module globals."""

    # Video source / CLI
    device: str = ''
    cam_stream: str = 'frontCam'
    port: int = 0
    requested_width: int | None = None   # --width CLI override (None = not set)
    requested_height: int | None = None  # --height CLI override (None = not set)

    # Resolution (mutated when the source is opened)
    resolution_x: int = 640
    resolution_y: int = 480
    framerate: int = 30

    # Model / inference
    object_model: str | None = None
    current_model_name: str | None = None
    use_sahi: bool = True

    # Thresholds
    conf: float = 0.1
    nms_iou: float = 0.5
    sahi_iou: float = 0.5
    frame_buffer: int = 64
    smoothing: bool = True

    # Class filtering
    class_list: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)

    # Detection backend
    detect_backend: str = 'mmdet'
    device_name: str = 'UNKNOWN_DEVICE'

    # IronFlock identity
    device_key: str | None = None
    device_url: str | None = None

    # Runtime mutable state (populated by watchers / main loop)
    saved_masks: list = field(default_factory=list)
    stream_settings: Dict[str, Any] = field(default_factory=dict)

    # Live model bundle (set after getModel())
    model: Dict[str, Any] = field(default_factory=dict)

    def settings_fingerprint(self) -> str:
        """Build a hashable fingerprint of all settings that affect inference."""
        mask_path = f'/data/masks/{self.cam_stream}.json'
        mask_mtime = 0.0
        try:
            mask_mtime = os.path.getmtime(mask_path)
        except OSError:
            pass
        parts = (
            str(self.current_model_name or ''),
            str(self.stream_settings.get('model', '') or ''),
            str(self.stream_settings.get('confidence', '') or ''),
            str(self.stream_settings.get('useSahi', '') or ''),
            str(self.stream_settings.get('nmsIou', '') or ''),
            str(self.stream_settings.get('sahiIou', '') or ''),
            str(self.stream_settings.get('overlapRatio', '') or ''),
            str(self.stream_settings.get('frameBuffer', '') or ''),
            str(self.stream_settings.get('classList', []) or []),
            str(self.stream_settings.get('classNames', []) or []),
            str(mask_mtime),
            str(len(self.saved_masks)),
        )
        return '|'.join(parts)


# ── Factories ───────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the videoStream CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Start a Video Stream for the given Camera Device',
    )
    parser.add_argument('device', type=str, help='A device path like e.g. /dev/video0')
    parser.add_argument('camStream', type=str, help='Stream name (e.g. frontCam)')
    parser.add_argument('--port', type=int, required=True, help='RTP port assigned by mediasoup')
    parser.add_argument('--width', type=int, default=None, help='Requested capture width (overrides RESOLUTION_X)')
    parser.add_argument('--height', type=int, default=None, help='Requested capture height (overrides RESOLUTION_Y)')
    return parser.parse_args(argv)


def _parse_class_list(raw: str) -> list[int]:
    """Parse the CLASS_LIST env var into a list of ints."""
    parts = raw.split(',')
    try:
        result = [int(num.strip()) for num in parts]
    except Exception:
        logger.warning('Invalid Class list given: %s', raw)
        return []
    if len(result) <= 1:
        return []
    return result


def create_config(args: argparse.Namespace) -> StreamConfig:
    """Build a ``StreamConfig`` from environment variables + parsed CLI args."""
    import torch

    resolution_x = int(os.environ.get('RESOLUTION_X', 640))
    resolution_y = int(os.environ.get('RESOLUTION_Y', 480))

    # Per-stream resolution overrides from CLI
    if args.width:
        resolution_x = args.width
    if args.height:
        resolution_y = args.height

    object_model = os.environ.get('OBJECT_MODEL')

    # Auto-select backend: TensorRT on CUDA, huggingface on CPU (AMD64 has no mmdet)
    if torch.cuda.is_available():
        _default_backend = 'tensorrt'
    else:
        # On AMD64 without CUDA, prefer huggingface (mmdet may not be installed)
        try:
            import mmdet  # noqa: F401
            _default_backend = 'mmdet'
        except ImportError:
            _default_backend = 'huggingface'

    cfg = StreamConfig(
        device=args.device,
        cam_stream=args.camStream,
        port=args.port,
        requested_width=args.width,
        requested_height=args.height,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        framerate=int(os.environ.get('FRAMERATE', 30)),
        object_model=object_model,
        current_model_name=object_model,
        use_sahi=(os.environ.get('USE_SAHI', 'false') == 'true'),
        conf=float(os.environ.get('CONF', '0.1')),
        nms_iou=float(os.environ.get('NMS_IOU', '0.5')),
        sahi_iou=float(os.environ.get('SAHI_IOU', '0.5')),
        frame_buffer=int(os.environ.get('FRAME_BUFFER', 64)),
        smoothing=(os.environ.get('SMOOTHING', 'false') == 'true'),
        class_list=_parse_class_list(os.environ.get('CLASS_LIST', '')),
        class_names=[],
        detect_backend=os.environ.get('DETECT_BACKEND', _default_backend),
        device_name=os.environ.get('DEVICE_NAME', 'UNKNOWN_DEVICE'),
        device_key=os.environ.get('DEVICE_KEY'),
        device_url=os.environ.get('DEVICE_URL'),
        stream_settings={'model': object_model},
    )

    logger.info('Using CLASS_LIST: %s', cfg.class_list)
    return cfg
