"""Video source management — opening USB cameras, RTSP, HTTP, images, and YouTube."""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from config import StreamConfig
from youtube import get_youtube_video

logger = logging.getLogger('video_source')


_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _is_image_url(url: str) -> bool:
    """Return True if *url* is an HTTP(S) URL pointing to an image file."""
    if not url.startswith(('http://', 'https://')):
        return False
    path_part = url.split('?')[0].split('#')[0]
    ext = os.path.splitext(path_part)[1].lower()
    return ext in _IMAGE_EXTENSIONS


def _is_image_file(path: str) -> bool:
    """Return True if *path* is a local image file."""
    if path.startswith(('http://', 'https://', 'rtsp://')):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in _IMAGE_EXTENSIONS and os.path.isfile(path)


def _download_image(url: str) -> str:
    """Download an image URL to /data/images/ and return the local path."""
    cache_dir = Path('/data/images')
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = os.path.splitext(url.split('?')[0].split('#')[0])[1].lower() or '.jpg'
    dest = cache_dir / f'{url_hash}{ext}'
    if dest.is_file():
        logger.info('Image already cached: %s', dest)
        return str(dest)
    logger.info('Downloading image: %s → %s', url, dest)
    urllib.request.urlretrieve(url, str(dest))
    return str(dest)


class ImageCapture:
    """cv2.VideoCapture-compatible wrapper that yields the same image forever.

    The existing main loop works unchanged: ``cap.read()`` returns
    ``(True, frame)``, ``cap.isOpened()`` returns True, etc.
    """

    def __init__(self, image_path: str, width: int, height: int):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'Cannot read image: {image_path}')
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        self._frame = img
        self._width = width
        self._height = height
        self._opened = True
        logger.info('ImageCapture opened: %s (%dx%d)', image_path, width, height)

    def set_frame(self, frame: np.ndarray):
        """Replace the current frame (for future triggered-snapshot use)."""
        self._frame = frame
        self._height, self._width = frame.shape[:2]

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        return True, self._frame.copy()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return 0.0
        return 0.0

    def set(self, prop_id, value):
        pass

    def release(self):
        self._opened = False


def setVideoSource(device: str, config: StreamConfig):
    """Open the appropriate ``VideoCapture`` for the given device string.

    Mutates ``config.resolution_x`` / ``config.resolution_y`` to match
    the actual source resolution.
    """
    if device.startswith('http'):
        # ── Image URL — download once, serve as static frame ──────────
        if _is_image_url(device):
            logger.info('Image URL detected: %s', device)
            local_path = _download_image(device)
            img = cv2.imread(local_path)
            if img is not None:
                config.resolution_x = img.shape[1]
                config.resolution_y = img.shape[0]
            cap = ImageCapture(local_path, config.resolution_x, config.resolution_y)
            logger.info('Image source ready: %dx%d', config.resolution_x, config.resolution_y)
            return cap

        if device.startswith('https://youtu') or device.startswith('https://www.youtube.com'):
            yt_height = config.requested_height
            try:
                video = get_youtube_video(device, yt_height)
                stream_url = video.get('url', device)
                if video.get('width'):
                    config.resolution_x = video['width']
                if video.get('height'):
                    config.resolution_y = video['height']
                logger.info('YouTube resolved to %dx%d, opening stream...',
                            config.resolution_x, config.resolution_y)
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.info('FFmpeg backend failed, trying auto backend')
                    cap = cv2.VideoCapture(stream_url)
                if cap.isOpened():
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info('YouTube opened: %dx%d', actual_w, actual_h)
                else:
                    logger.error('YouTube VideoCapture FAILED to open stream URL')
            except Exception as err:
                logger.warning('YouTube yt-dlp failed: %s — trying direct URL', err)
                cap = cv2.VideoCapture(device)
        else:
            # Direct HTTPS/HTTP streaming
            logger.info('Opening HTTP(S) video: %s', device)
            cap = None
            if torch.cuda.is_available():
                gst_pipe = (
                    f"souphttpsrc location={device} ! queue"
                    " ! h264parse ! nvv4l2decoder"
                    " ! nvvidconv ! video/x-raw, format=BGRx"
                    " ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
                )
                logger.info('Trying GStreamer HW decode for HTTP: %s', gst_pipe)
                cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
                if not cap.isOpened():
                    logger.warning('GStreamer HTTP HW decode failed, falling back')
                    cap = None

            if cap is None:
                cap = cv2.VideoCapture(device, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.warning('FFmpeg backend failed, trying auto backend: %s', device)
                    cap = cv2.VideoCapture(device, cv2.CAP_ANY)

            if cap.isOpened():
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width > 0 and actual_height > 0:
                    config.resolution_x = actual_width
                    config.resolution_y = actual_height
                    logger.info('HTTP video opened: %dx%d',
                                config.resolution_x, config.resolution_y)
            else:
                logger.error('Failed to open HTTP video: %s', device)

    elif device.startswith('rtsp:'):
        cap = None
        if torch.cuda.is_available():
            gst_pipe = (
                f"rtspsrc location={device} latency=200 ! queue"
                " ! rtph264depay ! h264parse ! nvv4l2decoder"
                " ! nvvidconv ! video/x-raw, format=BGRx"
                " ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
            )
            logger.info('RTSP via GStreamer HW decode: %s', gst_pipe)
            cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                logger.warning('GStreamer RTSP pipeline failed, falling back to FFmpeg')
                cap = None

        if cap is None:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;10000000'
            logger.info('Opening RTSP via FFmpeg: %s', device)
            cap = cv2.VideoCapture(device, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.warning('FFmpeg RTSP failed, trying auto backend: %s', device)
                cap = cv2.VideoCapture(device, cv2.CAP_ANY)

        if cap.isOpened():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w > 0 and actual_h > 0:
                config.resolution_x = actual_w
                config.resolution_y = actual_h
                logger.info('RTSP stream opened: %dx%d',
                            config.resolution_x, config.resolution_y)
        else:
            logger.error('Failed to open RTSP stream: %s', device)

    elif device.startswith('demoVideo'):
        cap = cv2.VideoCapture('/app/video/luggagebelt.m4v', cv2.CAP_FFMPEG)
        time.sleep(0.3)
        config.resolution_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        config.resolution_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info('Demo video opened: %dx%d (FFmpeg backend)',
                     config.resolution_x, config.resolution_y)

    elif _is_image_file(device):
        img = cv2.imread(device)
        if img is not None:
            config.resolution_x = img.shape[1]
            config.resolution_y = img.shape[0]
        cap = ImageCapture(device, config.resolution_x, config.resolution_y)
        logger.info('Image file source: %s (%dx%d)', device,
                     config.resolution_x, config.resolution_y)
    else:
        # USB camera
        if device.startswith('/dev/video'):
            dev_index = int(device.replace('/dev/video', ''))
        elif device[-1].isdigit():
            dev_index = int(device[-1])
        else:
            logger.error('Unrecognised device string: %s', device)
            sys.exit(1)
        cap = cv2.VideoCapture(dev_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution_x)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution_y)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w > 0 and actual_h > 0:
            config.resolution_x = actual_w
            config.resolution_y = actual_h
            logger.info('USB camera actual resolution: %dx%d',
                         config.resolution_x, config.resolution_y)
    return cap


def _is_loopable_source(device: str) -> bool:
    """Return True for finite-length sources that should loop (demo videos, YouTube)."""
    if device.startswith('demoVideo'):
        return True
    if device.startswith('https://youtu') or device.startswith('https://www.youtube.com'):
        return True
    return False


def reopen_source(cap, config: StreamConfig) -> tuple:
    """Release *cap* and reopen the video source.

    Determines internally whether the source is loopable (demo video,
    YouTube) or a live feed that genuinely failed.  Returns
    ``(new_cap, looped)`` where *looped* is ``True`` when the source
    simply reached the end and was silently reopened.
    """
    device = config.device
    looped = _is_loopable_source(device)
    cap.release()
    if device.startswith('demoVideo'):
        new_cap = cv2.VideoCapture('/app/video/luggagebelt.m4v', cv2.CAP_FFMPEG)
        logger.info('Demo video looped to start')
        return new_cap, True
    new_cap = setVideoSource(device, config)
    if looped:
        logger.info('Loopable source reopened: %s', device)
    return new_cap, looped


def open_with_retry(device: str, config: StreamConfig, max_retries: int = 12):
    """Open a video source, retrying up to *max_retries* times on failure."""
    cap = setVideoSource(device, config)
    retries = 0
    while not cap.isOpened() and retries < max_retries:
        retries += 1
        logger.warning('Source not open, retrying in 5s (%d/%d)...', retries, max_retries)
        time.sleep(5)
        cap = setVideoSource(device, config)

    if not cap.isOpened():
        logger.error('Failed to open video source after retries: %s', device)

    logger.info('Resolution: %dx%d, source: %s',
                config.resolution_x, config.resolution_y, device)
    return cap
