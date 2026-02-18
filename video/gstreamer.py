"""GStreamer RTP pipeline construction and VideoWriter creation."""
from __future__ import annotations

import logging
import sys

import cv2
import torch

from config import StreamConfig

logger = logging.getLogger('gstreamer')


def build_rtp_pipeline(config: StreamConfig) -> str:
    """Build the GStreamer H.264 → RTP pipeline string (HW or SW encoding)."""
    if torch.cuda.is_available():
        # Hardware h264 encoding on Jetson
        # profile=0 (Baseline) matches mediasoup's declared profile-level-id 42e01f
        # idrinterval=1 makes every I-frame a true IDR (NAL type 5)
        # videoconvert (CPU) is required because nvvidconv cannot accept BGR directly
        base_bps = 8_000_000
        base_pixels = 1280 * 720
        actual_pixels = config.resolution_x * config.resolution_y
        hw_bitrate = max(4_000_000, int(base_bps * actual_pixels / base_pixels))
        logger.info('HW encoder bitrate: %d bps (%dx%d)', hw_bitrate,
                     config.resolution_x, config.resolution_y)
        output_format = (
            "queue ! videoconvert ! video/x-raw, format=I420"
            " ! nvvidconv"
            " ! video/x-raw(memory:NVMM), format=I420"
            " ! nvv4l2h264enc maxperf-enable=true preset-level=1 profile=0"
            "   insert-sps-pps=true insert-vui=true iframeinterval=10 idrinterval=1"
            f"   control-rate=1 bitrate={hw_bitrate}"
            " ! h264parse"
            " ! rtph264pay pt=96 ssrc=11111111 config-interval=-1"
        )
    else:
        # CPU software encoding (x264)
        base_bitrate = 8000  # kbps at 1280x720
        base_pixels = 1280 * 720
        actual_pixels = config.resolution_x * config.resolution_y
        scaled_bitrate = max(4000, int(base_bitrate * actual_pixels / base_pixels))
        logger.info('x264 bitrate: %d kbps (%dx%d = %d pixels)',
                    scaled_bitrate, config.resolution_x, config.resolution_y, actual_pixels)
        # NOTE: RTCP PLI from mediasoup cannot reach x264enc (udpsink is
        # send-only), so on-demand keyframes are not possible.
        # key-int-max=1: every frame is an IDR keyframe — at low actual
        # FPS (e.g. CPU inference at ~0.25 fps) the overhead is negligible
        # and the browser can start decoding instantly on the very first
        # frame after (re)connecting.
        output_format = (
            "videoconvert ! video/x-raw, format=I420"
            f" ! x264enc tune=zerolatency bitrate={scaled_bitrate}"
            "   key-int-max=1 bframes=0 speed-preset=ultrafast"
            " ! video/x-h264, profile=constrained-baseline"
            " ! h264parse"
            " ! rtph264pay pt=96 ssrc=11111111 config-interval=-1"
        )

    pipeline = (
        "appsrc ! " + output_format +
        f" ! udpsink host=127.0.0.1 port={config.port} sync=false async=false"
    )
    return pipeline


def open_video_writer(config: StreamConfig) -> cv2.VideoWriter:
    """Build the GStreamer pipeline and open a ``cv2.VideoWriter``."""
    logger.info('Streaming %s to port %d (127.0.0.1)', config.cam_stream, config.port)

    if config.port < 1024 or config.port > 65535:
        logger.error('Port %d out of range', config.port)
        sys.exit(1)

    pipeline = build_rtp_pipeline(config)
    logger.info('GStreamer pipeline: %s', pipeline)
    out = cv2.VideoWriter(
        pipeline, cv2.CAP_GSTREAMER, 0,
        config.framerate,
        (config.resolution_x, config.resolution_y),
        True,
    )
    if out.isOpened():
        logger.info('GStreamer VideoWriter opened successfully (hw encoder: %s)',
                     'yes' if torch.cuda.is_available() else 'no')
    else:
        logger.error('GStreamer VideoWriter FAILED to open — check GStreamer plugin availability')
    return out
