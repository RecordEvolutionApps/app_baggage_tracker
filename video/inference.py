"""Inference dispatch — routes frames to the appropriate backend (HuggingFace, MMDet, or TensorRT)."""
from __future__ import annotations

import logging
import time as _time_mod
from typing import Any, Dict, Optional

import numpy as np
import supervision as sv
import torch
from PIL import Image

from config import StreamConfig

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
    """Top-level inference entry point — delegates to the correct backend."""
    return infer_frame(frame, model_bundle, confidence=confidence, iou=iou, config=config)


def infer_frame(frame: np.ndarray, model_bundle: Dict[str, Any],
                confidence: float | None = None, iou: float | None = None,
                config: StreamConfig | None = None) -> Optional[sv.Detections]:
    """Route inference to the correct backend."""
    backend = model_bundle.get('backend')
    if backend == 'tensorrt':
        return infer_tensorrt(frame, model_bundle, confidence=confidence, iou=iou, config=config)
    if backend == 'huggingface':
        return infer_huggingface(frame, model_bundle, confidence=confidence, iou=iou, config=config)
    if backend == 'mmdet':
        return infer_mmdet(frame, model_bundle['inferencer'], confidence=confidence, config=config)
    raise ValueError(f'Unsupported backend in model bundle: {backend}')


def infer_mmdet(frame: np.ndarray, inferencer, confidence: float | None = None,
                config: StreamConfig | None = None) -> Optional[sv.Detections]:
    """Run inference through the MMDetection backend."""
    conf_threshold = confidence if confidence is not None else (config.conf if config else 0.1)
    class_list = (config.class_list if config else []) or []
    try:
        _t0 = _time_mod.monotonic()
        results = inferencer(frame, return_vis=False, no_save_pred=True,
                             print_result=False, return_datasamples=True)
        _t_forward = _time_mod.monotonic()
        predictions = results.get('predictions', [])
        if not predictions:
            return False

        detections = sv.Detections.from_mmdetection(predictions[0])
        if len(detections) == 0:
            return False

        keep = detections.confidence >= conf_threshold
        if class_list:
            keep = np.logical_and(keep, np.isin(detections.class_id, class_list))

        detections = detections[keep]
        _t_post = _time_mod.monotonic()
        logger.info('[PROFILE-MMDET] forward=%.0fms  post=%.0fms  total=%.0fms  frame=%dx%d',
                    (_t_forward - _t0) * 1000, (_t_post - _t_forward) * 1000,
                    (_t_post - _t0) * 1000, frame.shape[1], frame.shape[0])
        return detections if len(detections) > 0 else False

    except Exception as e:
        logger.error('Failed to extract detections from MMDetection result: %s', e, exc_info=True)
        return False


def infer_tensorrt(frame: np.ndarray, model_bundle: Dict[str, Any],
                   confidence: float | None = None, iou: float | None = None,
                   config: StreamConfig | None = None) -> Optional[sv.Detections]:
    """Run inference through the TensorRT engine."""
    trt_inf = model_bundle['inferencer']
    conf = confidence if confidence is not None else (config.conf if config else 0.1)
    iou_threshold = iou if iou is not None else (config.nms_iou if config else 0.5)
    class_list = (config.class_list if config else []) or None
    try:
        _t0 = _time_mod.monotonic()
        result = trt_inf(frame, conf=conf, iou=iou_threshold, class_list=class_list)
        _dt = _time_mod.monotonic() - _t0
        logger.info('[PROFILE-TRT] total=%.0fms  frame=%dx%d',
                    _dt * 1000, frame.shape[1], frame.shape[0])
        return result if result is not None else False
    except Exception as e:
        logger.error('TensorRT inference failed: %s', e, exc_info=True)
        return False


def _infer_hf_segmentation(frame: np.ndarray, model_bundle: Dict[str, Any],
                           conf_threshold: float,
                           class_list: list) -> Optional[sv.Detections]:
    """Run Mask2Former / universal-segmentation inference and return sv.Detections.

    Uses the model's raw per-instance binary masks (which can overlap) instead of
    the overlap-resolved segmentation map produced by post_process_instance_segmentation.
    This avoids the mismatch where the tracker holds a large Kalman bbox from a previous
    frame while the current-frame mask was clipped to a tiny fragment by overlap resolution.
    """
    import torch.nn.functional as _F

    model = model_bundle['model']
    processor = model_bundle['processor']

    pil_image = Image.fromarray(frame[:, :, ::-1]) if frame.shape[2] == 3 else Image.fromarray(frame)
    inputs = processor(images=pil_image, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        outputs = model(**inputs)

    h, w = frame.shape[:2]

    # class_queries_logits: (1, num_queries, num_labels + 1)
    # Last label slot is the no-object class — exclude it.
    class_logits = outputs.class_queries_logits[0]          # (Q, C+1)
    pred_probs = class_logits.softmax(-1)[:, :-1]           # (Q, C)  no-object excluded
    pred_scores, pred_classes = pred_probs.max(-1)          # (Q,), (Q,)

    # Filter by confidence
    keep_idx = torch.where(pred_scores > conf_threshold)[0]
    if len(keep_idx) == 0:
        return False

    # masks_queries_logits: (1, num_queries, H', W') — typically 1/4 of model input size
    mask_logits = outputs.masks_queries_logits[0][keep_idx]  # (M, H', W')

    # Resize to full frame resolution
    mask_logits_full = _F.interpolate(
        mask_logits.unsqueeze(0).float(),
        size=(h, w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)  # (M, H, W)

    # sigmoid > 0.5  ⟺  logit > 0  — standard DETR-family mask threshold
    binary_masks = (mask_logits_full > 0).cpu().numpy()   # (M, H, W) bool

    scores_np  = pred_scores[keep_idx].cpu().numpy().astype(np.float32)
    classes_np = pred_classes[keep_idx].cpu().numpy().astype(np.int64)

    # Optional class filter
    if class_list:
        keep_cls = np.isin(classes_np, class_list)
        binary_masks = binary_masks[keep_cls]
        scores_np    = scores_np[keep_cls]
        classes_np   = classes_np[keep_cls]
        if len(scores_np) == 0:
            return False

    # Derive bbox from each instance's own mask (always aligned — no overlap resolution)
    boxes, valid = [], []
    for i, mask in enumerate(binary_masks):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        valid.append(i)

    if not boxes:
        return False

    valid = np.array(valid)
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=scores_np[valid],
        class_id=classes_np[valid],
        mask=binary_masks[valid],
    )


def infer_huggingface(frame: np.ndarray, model_bundle: Dict[str, Any],
                      confidence: float | None = None, iou: float | None = None,
                      config: StreamConfig | None = None) -> Optional[sv.Detections]:
    """Run inference through the HuggingFace Transformers backend."""
    model = model_bundle['model']
    processor = model_bundle['processor']
    conf_threshold = confidence if confidence is not None else (config.conf if config else 0.1)
    class_list = (config.class_list if config else []) or []

    try:
        _t0 = _time_mod.monotonic()

        if model_bundle.get('is_segmentation'):
            result = _infer_hf_segmentation(frame, model_bundle, conf_threshold, class_list)
            _t1 = _time_mod.monotonic()
            logger.info('[PROFILE-HF-SEG] total=%.0fms  frame=%dx%d',
                        (_t1 - _t0) * 1000, frame.shape[1], frame.shape[0])
            return result

        # Convert BGR (OpenCV) to RGB PIL image
        pil_image = Image.fromarray(frame[:, :, ::-1]) if frame.shape[2] == 3 else Image.fromarray(frame)

        # Preprocess
        inputs = processor(images=pil_image, return_tensors='pt')
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        _t_forward = _time_mod.monotonic()

        # Post-process: convert to target image size
        target_sizes = torch.tensor([frame.shape[:2]], device=device)  # (H, W)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )

        if not results:
            return False

        result = results[0]  # batch size 1
        if len(result.get('scores', [])) == 0:
            return False

        # Use supervision's built-in converter — handles both detection
        # and segmentation (masks populated automatically when present)
        id2label = getattr(model_bundle.get('model'), 'config', None)
        id2label = getattr(id2label, 'id2label', None) if id2label else None
        detections = sv.Detections.from_transformers(
            transformers_results=result,
            id2label=id2label,
        )

        # Apply class filter
        if class_list:
            keep = np.isin(detections.class_id, class_list)
            detections = detections[keep]

        _t_post = _time_mod.monotonic()
        logger.info('[PROFILE-HF] forward=%.0fms  post=%.0fms  total=%.0fms  frame=%dx%d',
                    (_t_forward - _t0) * 1000, (_t_post - _t_forward) * 1000,
                    (_t_post - _t0) * 1000, frame.shape[1], frame.shape[0])

        return detections if len(detections) > 0 else False

    except Exception as e:
        logger.error('HuggingFace inference failed: %s', e, exc_info=True)
        return False


def move_detections(detections: sv.Detections, offset_x: int, offset_y: int) -> sv.Detections:
    """Offset bounding boxes after SAHI crop-region inference."""
    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        box[0] += offset_x  # xmin
        box[1] += offset_y  # ymin
        box[2] += offset_x  # xmax
        box[3] += offset_y  # ymax
    return detections
