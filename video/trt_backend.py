"""TensorRT FP16 inference backend for RTMDet-family models.

Exports MMDetection models to ONNX, builds optimised TensorRT engines,
and runs inference with hardware-accelerated FP16 precision on NVIDIA GPUs.

The first call builds and caches the engine (~1-3 min on Jetson AGX).
Subsequent calls load from cache in seconds.

Environment variables
---------------------
DETECT_BACKEND=tensorrt    Enable TensorRT inference
TRT_CACHE_DIR=/data/trt    Engine / ONNX cache directory  (default: /data/tensorrt)
TRT_WORKSPACE_MB=1024      TensorRT builder workspace in MB
"""
from __future__ import annotations

import os
import logging
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, Dict

import supervision as sv

logger = logging.getLogger('trt_backend')

# --------------------------------------------------------------------------- #
# TensorRT availability
# --------------------------------------------------------------------------- #
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    logger.info('TensorRT %s available', trt.__version__)
except ImportError:
    TRT_AVAILABLE = False
    trt = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TRT_CACHE = Path(os.environ.get('TRT_CACHE_DIR', '/data/tensorrt'))
TRT_WORKSPACE_MB = int(os.environ.get('TRT_WORKSPACE_MB', '8192'))

# RTMDet preprocessing constants (ImageNet BGR mean / std used by mmdet)
_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)   # BGR
_STD  = np.array([57.375, 57.12,  58.395],  dtype=np.float32)   # BGR

# Same constants as GPU tensors (allocated once, reused every frame)
_MEAN_GPU = torch.tensor([103.53, 116.28, 123.675], dtype=torch.float32, device='cuda').view(1, 1, 3) if torch.cuda.is_available() else None
_STD_GPU  = torch.tensor([57.375, 57.12,  58.395],  dtype=torch.float32, device='cuda').view(1, 1, 3) if torch.cuda.is_available() else None

# RTMDet FPN strides (same for tiny / s / m / l / x)
_STRIDES = (8, 16, 32)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def is_available() -> bool:
    """True when TensorRT runtime *and* a CUDA GPU are present."""
    return TRT_AVAILABLE and torch.cuda.is_available()


def engine_path(model_name: str) -> Path:
    return TRT_CACHE / f'{model_name}_fp16.engine'


def onnx_path(model_name: str) -> Path:
    return TRT_CACHE / f'{model_name}.onnx'


def is_engine_cached(model_name: str) -> bool:
    return engine_path(model_name).is_file()


# ═══════════════════════════════════════════════════════════════════════════ #
# ONNX Export                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class _ExportWrapper(torch.nn.Module):
    """Thin wrapper around an MMDet model for clean ONNX export.

    Runs  backbone → neck → bbox_head.forward()  and returns the raw
    (pre-NMS) class scores and bbox distance predictions concatenated
    across all FPN levels.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone  = model.backbone
        self.neck      = model.neck
        self.bbox_head = model.bbox_head

    def forward(self, images: torch.Tensor):
        feats = self.backbone(images)
        feats = self.neck(feats)

        # head.forward() → (cls_scores_list, bbox_preds_list)
        outputs = self.bbox_head(feats)
        cls_list, box_list = outputs[0], outputs[1]

        flat_cls: list[torch.Tensor] = []
        flat_box: list[torch.Tensor] = []
        for cls, box in zip(cls_list, box_list):
            B, C, H, W = cls.shape
            flat_cls.append(cls.permute(0, 2, 3, 1).reshape(B, H * W, C))
            flat_box.append(box.permute(0, 2, 3, 1).reshape(B, H * W, 4))

        # (B, total_anchors, num_classes),  (B, total_anchors, 4)
        return torch.cat(flat_cls, dim=1), torch.cat(flat_box, dim=1)


def export_to_onnx(
    model_name: str,
    inferencer,
    input_wh: Tuple[int, int],
) -> str:
    """Export an MMDetection model to ONNX.  Returns path to the .onnx file."""
    dst = onnx_path(model_name)
    if dst.is_file():
        logger.info('ONNX cached: %s', dst)
        return str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)

    model   = inferencer.model
    wrapper = _ExportWrapper(model).eval()
    device  = next(model.parameters()).device
    w, h    = input_wh
    dummy   = torch.randn(1, 3, h, w, device=device)

    logger.info('Exporting %s → ONNX (input %dx%d) …', model_name, w, h)
    t0 = time.monotonic()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(dst),
            input_names=['images'],
            output_names=['scores', 'boxes'],
            opset_version=13,
            do_constant_folding=True,
        )

    logger.info(
        'ONNX export done in %.1fs → %s (%.1f MB)',
        time.monotonic() - t0, dst, dst.stat().st_size / 1e6,
    )
    return str(dst)


# ═══════════════════════════════════════════════════════════════════════════ #
# TensorRT Engine Build                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

def build_engine_from_onnx(model_name: str, onnx_file: str) -> str:
    """Build a TensorRT FP16 engine from an ONNX file.  Returns engine path."""
    if not TRT_AVAILABLE:
        raise RuntimeError('TensorRT Python package is not installed')

    dst = engine_path(model_name)
    if dst.is_file():
        logger.info('TRT engine cached: %s', dst)
        return str(dst)

    logger.info(
        'Building TRT FP16 engine from %s (workspace %d MB) …',
        onnx_file, TRT_WORKSPACE_MB,
    )
    t0 = time.monotonic()

    trt_log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_log)
    flags   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser  = trt.OnnxParser(network, trt_log)

    with open(onnx_file, 'rb') as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError('ONNX parse failed:\n' + '\n'.join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, TRT_WORKSPACE_MB << 20)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info('FP16 mode enabled (platform supports fast FP16)')
    else:
        logger.warning('FP16 not natively supported on this GPU — building FP32 engine')

    logger.info('TensorRT engine build started — this may take 5-20 minutes on Jetson. Please wait...')

    # Log a heartbeat so the user knows the process is alive
    import threading
    _build_done = threading.Event()
    def _heartbeat():
        elapsed = 0
        while not _build_done.is_set():
            _build_done.wait(60)
            if not _build_done.is_set():
                elapsed += 1
                logger.info('TensorRT engine build still running... (%d min elapsed)', elapsed)
    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    serialised = builder.build_serialized_network(network, config)
    _build_done.set()

    if serialised is None:
        raise RuntimeError('TensorRT engine serialisation failed')

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(str(dst), 'wb') as f:
        f.write(serialised)

    logger.info(
        'TRT engine built in %.0fs → %s (%.1f MB)',
        time.monotonic() - t0, dst, dst.stat().st_size / 1e6,
    )
    return str(dst)


# ═══════════════════════════════════════════════════════════════════════════ #
# TensorRT Inference                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class TRTInferencer:
    """High-performance TensorRT inference for RTMDet-family models.

    Uses **torch CUDA tensors** for zero-copy GPU memory management (no
    pycuda dependency).  Pre-computes anchor grids for O(1) bbox decoding.
    """

    # ── Initialisation ─────────────────────────────────────────────────
    def __init__(
        self,
        engine_file: str,
        input_wh: Tuple[int, int],
        num_classes: int = 80,
    ):
        if not TRT_AVAILABLE:
            raise RuntimeError('TensorRT is not installed')

        self.input_w, self.input_h = input_wh
        self.num_classes = num_classes

        # Load serialised engine
        trt_log = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_log)
        with open(engine_file, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to load TRT engine: {engine_file}')

        self.context = self.engine.create_execution_context()
        self._use_v3 = hasattr(self.context, 'execute_async_v3')

        # Allocate I/O buffers as torch CUDA tensors
        self._io: Dict[str, torch.Tensor] = {}
        self._io_names: list[str] = []
        # Map TensorRT dtypes directly to torch dtypes (avoids trt.nptype()
        # which uses removed np.bool on older TRT + newer NumPy).
        _trt_to_torch = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32:   torch.int32,
            trt.int8:    torch.int8,
        }

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            trt_dt = self.engine.get_tensor_dtype(name)
            t_dt   = _trt_to_torch.get(trt_dt, torch.float32)
            buf = torch.empty(shape, dtype=t_dt, device='cuda')
            self._io[name] = buf
            self._io_names.append(name)
            if self._use_v3:
                self.context.set_tensor_address(name, buf.data_ptr())

        self._stream = torch.cuda.Stream()

        # Pre-compute anchor points for bbox decoding (CPU for numpy fallback)
        self._anchors = self._make_anchors()   # (N, 2)
        # GPU anchor grid for CUDA-accelerated postprocessing
        self._anchors_gpu = torch.from_numpy(self._anchors).cuda()  # (N, 2)

        # Persistent GPU pad buffer — avoids re-allocation every frame
        self._pad_buf = torch.full(
            (self.input_h, self.input_w, 3), 114,
            dtype=torch.uint8, device='cuda',
        )

        logger.info(
            'TRT engine ready: %s  input=%dx%d  anchors=%d  classes=%d',
            engine_file, self.input_w, self.input_h,
            len(self._anchors), self.num_classes,
        )

    # ── Anchor grid ────────────────────────────────────────────────────
    def _make_anchors(self) -> np.ndarray:
        """Grid centre points for every FPN level, shape (N, 2)."""
        pts: list[np.ndarray] = []
        for s in _STRIDES:
            gh, gw = self.input_h // s, self.input_w // s
            yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
            level = np.stack([
                (xx.ravel() + 0.5) * s,
                (yy.ravel() + 0.5) * s,
            ], axis=1).astype(np.float32)
            pts.append(level)
        return np.concatenate(pts, axis=0)

    # ── Preprocessing ──────────────────────────────────────────────────
    def preprocess(
        self, frame: np.ndarray,
    ) -> Tuple[torch.Tensor, float, int, int]:
        """Letterbox resize + normalise on GPU.

        Uploads the raw frame to CUDA first, then does resize, pad,
        float conversion, mean/std normalisation, and HWC→NCHW transpose
        entirely on the GPU — avoiding ~5 heavy CPU operations per frame.

        Returns (blob_nchw_gpu, scale, pad_x, pad_y).
        """
        oh, ow = frame.shape[:2]
        if oh == 0 or ow == 0:
            # Empty frame — return a zero-filled blob so inference produces no detections
            blob = torch.zeros(1, 3, self.input_h, self.input_w, dtype=torch.float32, device='cuda')
            return blob, 1.0, 0, 0
        scale = min(self.input_w / ow, self.input_h / oh)
        nw, nh = int(ow * scale), int(oh * scale)

        # Upload raw uint8 frame to GPU (single CPU→GPU copy)
        frame_gpu = torch.from_numpy(frame).cuda()           # (H, W, 3) uint8

        # GPU resize via torch (bilinear, same quality as cv2.INTER_LINEAR)
        # torch interpolate expects (N, C, H, W) float
        frame_chw = frame_gpu.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
        resized = torch.nn.functional.interpolate(
            frame_chw, size=(nh, nw), mode='bilinear', align_corners=False,
        )  # (1, 3, nh, nw)
        resized_hwc = resized.squeeze(0).permute(1, 2, 0).to(torch.uint8)  # (nh, nw, 3)

        # Pad on GPU (re-use persistent buffer)
        px = (self.input_w - nw) // 2
        py = (self.input_h - nh) // 2
        self._pad_buf.fill_(114)
        self._pad_buf[py:py + nh, px:px + nw] = resized_hwc

        # Normalise: float conversion + (pixel - mean) / std, then HWC → NCHW
        blob = (self._pad_buf.float() - _MEAN_GPU) / _STD_GPU  # (H, W, 3)
        blob = blob.permute(2, 0, 1).unsqueeze(0)              # (1, 3, H, W)
        blob = blob.contiguous()
        return blob, scale, px, py

    # ── Postprocessing ─────────────────────────────────────────────────
    def postprocess(
        self,
        scores_gpu: torch.Tensor,
        bboxes_gpu: torch.Tensor,
        scale: float,
        pad_x: int,
        pad_y: int,
        conf_threshold: float,
        iou_threshold: float,
        class_list: list | None,
    ) -> sv.Detections | None:
        """Decode bbox distances, undo letterbox, run NMS — all on GPU.

        Only the final kept detections (~50-200) are downloaded to CPU,
        instead of all ~8400 anchors.
        """
        # Sigmoid on GPU (fused, vectorised)
        scores = torch.sigmoid(scores_gpu)                        # (N, C)

        # Best class per anchor
        max_scores, class_ids = scores.max(dim=1)                 # (N,), (N,)

        # Confidence pre-filter on GPU (eliminates ~90%+ of anchors early)
        keep = max_scores >= conf_threshold
        if class_list:
            class_mask = torch.zeros(self.num_classes, dtype=torch.bool, device='cuda')
            class_mask[class_list] = True
            keep &= class_mask[class_ids]

        max_scores = max_scores[keep]
        class_ids  = class_ids[keep]
        bboxes_k   = bboxes_gpu[keep]                             # (K, 4)
        anchors_k  = self._anchors_gpu[keep]                      # (K, 2)

        if max_scores.numel() == 0:
            return None

        # Decode: anchor ± distance → xyxy (on GPU)
        x1 = anchors_k[:, 0] - bboxes_k[:, 0]
        y1 = anchors_k[:, 1] - bboxes_k[:, 1]
        x2 = anchors_k[:, 0] + bboxes_k[:, 2]
        y2 = anchors_k[:, 1] + bboxes_k[:, 3]
        bboxes = torch.stack([x1, y1, x2, y2], dim=1)            # (K, 4)

        # Undo letterbox padding → original image coordinates (on GPU)
        bboxes[:, 0] = (bboxes[:, 0] - pad_x) / scale
        bboxes[:, 2] = (bboxes[:, 2] - pad_x) / scale
        bboxes[:, 1] = (bboxes[:, 1] - pad_y) / scale
        bboxes[:, 3] = (bboxes[:, 3] - pad_y) / scale

        # NMS on GPU via torchvision
        try:
            import torchvision
            nms_idx = torchvision.ops.nms(bboxes, max_scores, iou_threshold)
        except Exception:
            # Fallback: download to CPU for numpy NMS
            nms_idx_np = _nms_numpy(
                bboxes.cpu().numpy(),
                max_scores.cpu().numpy(),
                iou_threshold,
            )
            nms_idx = torch.from_numpy(nms_idx_np).cuda()

        bboxes     = bboxes[nms_idx]
        max_scores = max_scores[nms_idx]
        class_ids  = class_ids[nms_idx]

        if bboxes.numel() == 0:
            return None

        # Only now download the final ~50-200 kept detections to CPU
        return sv.Detections(
            xyxy=bboxes.cpu().numpy().astype(np.float32),
            confidence=max_scores.cpu().numpy().astype(np.float32),
            class_id=class_ids.cpu().numpy().astype(np.int64),
        )

    # ── Inference ──────────────────────────────────────────────────────
    def __call__(
        self,
        frame: np.ndarray,
        conf: float = 0.1,
        iou: float = 0.5,
        class_list: list | None = None,
    ) -> sv.Detections | None:
        """Run end-to-end inference on a BGR uint8 frame."""
        blob_gpu, scale, px, py = self.preprocess(frame)

        # Copy preprocessed GPU tensor directly into the engine input buffer
        self._io['images'].copy_(blob_gpu)

        # Execute the engine
        with torch.cuda.stream(self._stream):
            if self._use_v3:
                self.context.execute_async_v3(self._stream.cuda_stream)
            else:
                bindings = [int(self._io[n].data_ptr()) for n in self._io_names]
                self.context.execute_async_v2(
                    bindings=bindings,
                    stream_handle=self._stream.cuda_stream,
                )
        self._stream.synchronize()

        # Keep outputs on GPU as float32 tensors for GPU postprocessing
        scores_gpu = self._io['scores'].float()[0]   # (N, C)
        boxes_gpu  = self._io['boxes'].float()[0]    # (N, 4)

        # Sanity: swap if ONNX output order differs from expected
        if scores_gpu.shape[-1] == 4 and boxes_gpu.shape[-1] != 4:
            scores_gpu, boxes_gpu = boxes_gpu, scores_gpu

        return self.postprocess(
            scores_gpu, boxes_gpu, scale, px, py,
            conf, iou, class_list,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# High-level API  (used by model_utils.get_tensorrt_model)                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def build_trt_model(
    model_name: str,
    mmdet_inferencer,
    input_wh: Tuple[int, int],
    num_classes: int = 80,
) -> TRTInferencer:
    """Full pipeline:  ONNX export → TRT engine build → load inferencer."""
    onnx_file   = export_to_onnx(model_name, mmdet_inferencer, input_wh)
    engine_file = build_engine_from_onnx(model_name, onnx_file)
    return TRTInferencer(engine_file, input_wh, num_classes=num_classes)


# ═══════════════════════════════════════════════════════════════════════════ #
# Utilities                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def _nms_numpy(
    bboxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure-numpy NMS fallback (no torchvision dependency)."""
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)
