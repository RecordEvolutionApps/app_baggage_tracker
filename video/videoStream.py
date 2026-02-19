import sys
import os

# ── Jetson TLS diagnostics & fix (arm64 only) ──────────────────────────────
# On Jetson (L4T), the static TLS block is limited. PyTorch + CUDA can exhaust
# it, preventing GStreamer NVIDIA plugins from loading libGLdispatch.so.0.
# We force-load these .so files before any heavy imports (torch, cv2).
import platform as _platform
if _platform.machine() == 'aarch64':
    import ctypes
    _tls_libs = [
        '/lib/aarch64-linux-gnu/libGLdispatch.so.0',
        '/usr/lib/aarch64-linux-gnu/libGLESv2.so.2',
        '/usr/lib/aarch64-linux-gnu/libEGL.so.1',
        '/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvidconv.so',
        '/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so',
    ]
    for _lib in _tls_libs:
        if os.path.exists(_lib):
            try:
                ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
                print(f'[TLS] Pre-loaded: {_lib}', flush=True)
            except OSError as e:
                print(f'[TLS] FAILED to pre-load {_lib}: {e}', flush=True)
    LD_PRELOAD = os.environ.get('LD_PRELOAD', '<not set>')
    print(f'[TLS] LD_PRELOAD={LD_PRELOAD}', flush=True)
    del _lib, _tls_libs, LD_PRELOAD
del _platform
# ────────────────────────────────────────────────────────────────────────────

import collections
import json
import time
import logging
from asyncio import get_event_loop, sleep
from datetime import datetime

import cv2
import supervision as sv
import torch

from config import parse_args, create_config
from frame_processing import processFrame
from gstreamer import open_video_writer
from inference import infer, empty_detections
from model_loader import getModel
from model_zoo import write_backend_status
from overlay import overlay_text, draw_sahi_grid
from publisher import Publisher, StubIronFlock
from sahi import run_sahi_inference
from video_source import setVideoSource, reopen_source, ImageCapture
from watchers import watchMaskFile, watchSettingsFile

# Configure logging for the whole video service
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('videoStream')


# ── Profiling helper ───────────────────────────────────────────────────────

class _Profiler:
    """Lightweight per-phase wall-clock profiler for the main video loop.

    Usage:
        prof = _Profiler(interval=30.0)
        prof.begin()          # start full-loop timer
        prof.mark('read')     # mark end of 'read' phase
        prof.mark('infer')    # mark end of 'infer' phase
        ...
        prof.end_loop()       # finish this iteration
        # every *interval* seconds a summary is logged
    """

    def __init__(self, interval: float = 30.0):
        self._interval = interval
        self._accum: dict[str, float] = collections.defaultdict(float)
        self._counts: dict[str, int] = collections.defaultdict(int)
        self._total_loop = 0.0
        self._loop_count = 0
        self._last_report = time.monotonic()
        self._t0 = 0.0
        self._cursor = 0.0

    def begin(self):
        """Call at the very start of each loop iteration."""
        self._t0 = time.monotonic()
        self._cursor = self._t0

    def mark(self, phase: str):
        """Record elapsed time since last mark (or begin) under *phase*."""
        now = time.monotonic()
        dt = now - self._cursor
        self._accum[phase] += dt
        self._counts[phase] += 1
        self._cursor = now

    def end_loop(self):
        """Finish iteration; log if interval has elapsed."""
        now = time.monotonic()
        self._total_loop += now - self._t0
        self._loop_count += 1
        if now - self._last_report >= self._interval:
            self._report()
            self._last_report = now

    def _report(self):
        n = self._loop_count or 1
        parts = []
        for phase in self._accum:
            avg_ms = self._accum[phase] / n * 1000
            pct = self._accum[phase] / self._total_loop * 100 if self._total_loop else 0
            parts.append(f'{phase}={avg_ms:.1f}ms({pct:.0f}%)')
        total_ms = self._total_loop / n * 1000
        logger.info('[PROFILE] %d frames, %.1f ms/frame avg | %s',
                    self._loop_count, total_ms, '  '.join(parts))
        # reset accumulators
        self._accum.clear()
        self._counts.clear()
        self._total_loop = 0.0
        self._loop_count = 0

# ── Bootstrap ──────────────────────────────────────────────────────────────

args = parse_args()
cfg = create_config(args)

device = cfg.device
logger.info('Camera: %s', device)

# Check OpenCV video backends
_bi = cv2.getBuildInformation()
import re as _re
_gst = bool(_re.search(r'GStreamer:\s+YES', _bi))
_ffmpeg = bool(_re.search(r'FFMPEG:\s+YES', _bi))
logger.info('OpenCV backends — FFmpeg: %s, GStreamer: %s',
    'YES' if _ffmpeg else 'NO',
    'YES' if _gst else 'NO')

# ── Open video source ──────────────────────────────────────────────────────

cap = setVideoSource(device, cfg)

# If the source failed to open, retry in a loop before giving up
_open_retries = 0
while not cap.isOpened() and _open_retries < 12:
    _open_retries += 1
    logger.warning('Source not open, retrying in 5s (%d/12)...', _open_retries)
    time.sleep(5)
    cap = setVideoSource(device, cfg)

if not cap.isOpened():
    logger.error('Failed to open video source after retries: %s', device)

logger.info('Resolution: %dx%d, source: %s', cfg.resolution_x, cfg.resolution_y, device)
cap.set(3, cfg.resolution_x)
cap.set(4, cfg.resolution_y)

# Write the actual resolved resolution so the web backend / frontend can read it
_status_dir = os.path.join('/data', 'status')
os.makedirs(_status_dir, exist_ok=True)
_resolution_file = os.path.join(_status_dir, f'{cfg.cam_stream}.resolution.json')
with open(_resolution_file, 'w') as _rf:
    json.dump({'width': cfg.resolution_x, 'height': cfg.resolution_y}, _rf)
logger.info('Wrote resolution status: %dx%d → %s', cfg.resolution_x, cfg.resolution_y, _resolution_file)

# ── Load persisted settings (written by web backend) ───────────────────────
# Read the settings file *before* loading the model so that the user's
# last-applied model choice takes precedence over the OBJECT_MODEL env var.
_settings_path = f'/data/settings/{cfg.cam_stream}.json'
if os.path.exists(_settings_path):
    try:
        with open(_settings_path, 'r') as _sf:
            _saved = json.load(_sf)
        cfg.stream_settings.update(_saved)
        _saved_model = _saved.get('model')
        if _saved_model and _saved_model != 'none':
            cfg.object_model = _saved_model
            cfg.current_model_name = _saved_model
            logger.info('Loaded model from settings file: %s', cfg.object_model)
        else:
            logger.info('Settings file has no model or model=none')
    except Exception as _e:
        logger.warning('Could not read settings file %s: %s', _settings_path, _e)
else:
    logger.info('No settings file found at %s, using env/defaults', _settings_path)

# ── Load initial model ─────────────────────────────────────────────────────

model = getModel(cfg.object_model, cfg)
cfg.model = model
cfg.current_model_name = cfg.object_model
logger.info('Model native input: %s', model.get('native_input_wh', 'unknown'))
logger.info('[PROFILE] Backend=%s, model=%s, SAHI=%s, device=%s, res=%dx%d',
            model.get('backend', '?'), cfg.object_model, cfg.use_sahi,
            'CUDA' if torch.cuda.is_available() else 'CPU',
            cfg.resolution_x, cfg.resolution_y)
write_backend_status(cfg.cam_stream, model, detect_backend=cfg.detect_backend)


# ── Inference loop ─────────────────────────────────────────────────────────

async def main(config):
    global cap, model  # noqa: PLW0603

    publisher = config._publisher
    stream_settings = config.stream_settings
    saved_masks = config.saved_masks

    try:
        fps_monitor = sv.FPSMonitor()
        start_time = time.time()
        start_time1 = time.time()

        out = open_video_writer(config)

        # SAHI slicer (created lazily)
        slicer = None  # created lazily by run_sahi_inference()

        # ── Image source: cached-inference state ────────────────────────
        _is_image_source = isinstance(cap, ImageCapture)
        _img_cached_frame = None
        _img_inference_fps = 0.0
        _img_inference_ts = ''
        logger.info('Starting main video loop... (image_source=%s)', _is_image_source)
        prof = _Profiler(interval=30.0)
        success = True
        frame = None
        loop_start = time.monotonic()
        frame_interval = 1.0 / config.framerate
        next_frame_time = time.monotonic()
        aggCounts = []
        zoneCounts = []
        lineCounts = []

        while cap.isOpened():
            prof.begin()
            await sleep(0)  # Give other tasks time to run
            elapsed_time = time.time() - start_time
            elapsed_time1 = time.time() - start_time1

            sahi_grid = None

            skip_inference = stream_settings.get('model', '') == 'none'
            prof.mark('setup')

            # ── Hot-reload model when the user picks a different one ──
            new_model_name = stream_settings.get('model', config.current_model_name)
            if new_model_name and new_model_name != 'none' and new_model_name != config.current_model_name:
                logger.info('Model changed: %s -> %s — reloading...',
                            config.current_model_name, new_model_name)
                try:
                    model = getModel(new_model_name, config)
                    config.current_model_name = new_model_name
                    config.model = model
                    slicer = None  # force SAHI re-init with new model
                    logger.info('Model reloaded: %s, native input: %s',
                                config.current_model_name, model.get('native_input_wh', 'unknown'))
                    write_backend_status(config.cam_stream, model, detect_backend=config.detect_backend)
                except Exception as e:
                    logger.error('Failed to reload model %s: %s', new_model_name, e, exc_info=True)

            # Dynamically update class filter from stream settings
            settings_class_list = stream_settings.get('classList', None)
            if settings_class_list is not None and len(settings_class_list) > 0:
                new_list = [int(c) for c in settings_class_list]
                if new_list != config.class_list:
                    config.class_list = new_list
                    logger.info('Updated CLASS_LIST from settings: %s', new_list)
            elif settings_class_list is not None and len(settings_class_list) == 0:
                if config.class_list:
                    config.class_list = []
                    logger.info('Reset CLASS_LIST to all classes (no filter)')

            # Dynamically update open-vocab class names from stream settings
            settings_class_names = stream_settings.get('classNames', None)
            if settings_class_names is not None and isinstance(settings_class_names, list):
                if len(settings_class_names) > 0:
                    if settings_class_names != config.class_names:
                        config.class_names = settings_class_names
                        logger.info('Updated CLASS_NAMES from settings: %s', settings_class_names)

            # ── Image source: fast path — use cached annotated frame ────
            if _is_image_source and _img_cached_frame is not None:
                fp = config.settings_fingerprint()
                if fp == _img_fingerprint:
                    out_frame = _img_cached_frame.copy()
                    overlay_text(out_frame, f'Timestamp: {_img_inference_ts}', position=(10, 30))
                    overlay_text(out_frame, f'FPS: {_img_inference_fps:.2f}', position=(10, 60))

                    wait_time = next_frame_time - time.monotonic()
                    if wait_time > 0:
                        await sleep(wait_time)
                    next_frame_time += frame_interval
                    if next_frame_time < time.monotonic() - 0.5:
                        next_frame_time = time.monotonic() + frame_interval

                    if elapsed_time1 >= 2.0:
                        publisher.publish_image(out_frame)
                        start_time1 = time.time()
                    if elapsed_time > 10.0:
                        publisher.publish_cameras()
                        start_time = time.time()

                    out.write(out_frame)
                    continue
                else:
                    logger.info('[image] Settings changed, re-running inference')
                    _img_cached_frame = None
                    _img_fingerprint = None

            # --- Fixed-timestep pacing ---
            wait_time = next_frame_time - time.monotonic()
            if wait_time > 0:
                await sleep(wait_time)
            prof.mark('pacing')

            # Read one frame
            success, frame = cap.read()

            # When inference is active and we fell behind, drop frames to catch up
            if success and not skip_inference:
                video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if video_ms > 0:
                    real_ms = (time.monotonic() - loop_start) * 1000
                    while real_ms >= video_ms and video_ms > 0:
                        success, frame = cap.read()
                        if not success:
                            break
                        real_ms = (time.monotonic() - loop_start) * 1000
                        video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Schedule next frame relative to previous target (absorbs jitter)
            next_frame_time += frame_interval
            if next_frame_time < time.monotonic() - 0.5:
                next_frame_time = time.monotonic() + frame_interval

            if not success:
                cap, looped = reopen_source(cap, config)
                loop_start = time.monotonic()
                next_frame_time = time.monotonic()
                if not looped:
                    logger.warning('Restarting video source: %s', device)
                    await sleep(1)
                    if elapsed_time >= 2.0:
                        logger.error('Video frame could not be read from source: %s', device)
                        start_time = time.time()
                continue

            fps_monitor.tick()
            prof.mark('read')

            # ── Run inference on this frame ──────────────────────────────
            if skip_inference:
                detections = empty_detections()
            else:
                use_sahi = stream_settings.get('useSahi', config.use_sahi)
                _t_inference_start = time.monotonic()
                try:
                    if use_sahi:
                        detections, slicer, sahi_grid = run_sahi_inference(
                            frame, slicer, model, saved_masks, stream_settings, config)
                        _dt_inf = time.monotonic() - _t_inference_start
                        n_slices = getattr(slicer, '_sahi_slice_count', [None])[0]
                        logger.info('[PROFILE-INF] SAHI: %.0fms total, %s slices, %.0fms/slice, backend=%s, frame=%dx%d',
                                    _dt_inf * 1000, n_slices,
                                    (_dt_inf * 1000 / n_slices) if n_slices else 0,
                                    model.get('backend', '?'),
                                    frame.shape[1], frame.shape[0])
                    else:
                        slicer = None
                        conf = float(stream_settings.get('confidence', config.conf))
                        nms_iou = float(stream_settings.get('nmsIou', config.nms_iou))
                        detections = infer(frame, model, confidence=conf, iou=nms_iou, config=config)
                        _dt_inf = time.monotonic() - _t_inference_start
                        logger.info('[PROFILE-INF] Direct: %.0fms, backend=%s, frame=%dx%d',
                                    _dt_inf * 1000, model.get('backend', '?'),
                                    frame.shape[1], frame.shape[0])

                    if not detections:
                        detections = empty_detections()
                except Exception as e:
                    logger.error('Inference error: %s', e, exc_info=True)
                    detections = empty_detections()

                if _is_image_source:
                    _dt_inference = time.monotonic() - _t_inference_start
                    _img_inference_fps = 1.0 / _dt_inference if _dt_inference > 0 else 0.0
                    _img_inference_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            prof.mark('inference')

            # ── Overlay detections & annotate ───────────────────────────

            frame, zoneCounts, lineCounts = processFrame(
                frame, detections, saved_masks,
                config=config, settings_dict=stream_settings,
            )

            prof.mark('processFrame')

            if sahi_grid:
                frame = draw_sahi_grid(frame, sahi_grid.rect, sahi_grid.slice_wh, sahi_grid.overlap_wh)

            # Draw overlays
            if _is_image_source:
                overlay_text(frame, f'Timestamp: {_img_inference_ts}', position=(10, 30))
                overlay_text(frame, f'FPS: {_img_inference_fps:.2f}', position=(10, 60))
            else:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                overlay_text(frame, f'Timestamp: {current_time}', position=(10, 30))
                overlay_text(frame, f'FPS: {fps_monitor.fps:.2f}', position=(10, 60))

            prof.mark('overlay')

            # Aggregate line counts for later publish
            for item in lineCounts:
                agg = next((a for a in aggCounts if a["label"] == item["label"]), None)
                if not agg:
                    agg = {"label": item["label"]}
                    aggCounts.append(agg)
                agg["num_in"] = agg.get("num_in", 0) + item["num_in"]
                agg["num_out"] = agg.get("num_out", 0) + item["num_out"]

            # Publish data
            if elapsed_time1 >= 2.0:
                publisher.publish_image(frame)
                for item in zoneCounts:
                    publisher.publish_class_count(item["label"], item["count"])
                for item in aggCounts:
                    publisher.publish_line_count(item["label"], item["num_in"], item["num_out"])
                aggCounts = []
                start_time1 = time.time()

            if elapsed_time > 10.0:
                publisher.publish_cameras()
                start_time = time.time()
            prof.mark('publish')

            # ── Image source: cache the annotated frame for reuse ───────
            if _is_image_source:
                _img_cached_frame = frame.copy()
                _img_fingerprint = config.settings_fingerprint()
                logger.info('[image] Inference complete, frame cached (fingerprint=%s…)',
                            _img_fingerprint[:40])

            out.write(frame)
            prof.mark('gst_write')
            prof.end_loop()

        logger.warning('Video source is not open: %s', device)
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            logger.debug('cv2.destroyAllWindows() not supported in headless mode')
    except Exception as e:
        logger.critical('Fatal error in video loop: %s', e, exc_info=True)
        sys.exit(1)


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ENV = os.environ.get('ENV', '')

    if ENV == 'DEV':
        ironflock = StubIronFlock()
    else:
        from ironflock import IronFlock
        ironflock = IronFlock()

    cfg._publisher = Publisher(ironflock, cfg)

    loop = get_event_loop()
    loop.create_task(watchMaskFile(cfg))
    loop.create_task(watchSettingsFile(cfg))
    loop.create_task(main(cfg))

    if ENV == 'DEV':
        loop.run_forever()
    else:
        ironflock.run()
