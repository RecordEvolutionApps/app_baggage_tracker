"""HuggingFace Transformers inference engine."""
from __future__ import annotations

import logging
import time as _time_mod
from typing import Any, Dict, Optional

import numpy as np
import supervision as sv
import torch
from PIL import Image

from engines.base import InferenceEngine

logger = logging.getLogger('engines.huggingface')

# ── Platform detection ──────────────────────────────────────────────────────
try:
    import transformers as _transformers  # noqa: F401
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# Throttle per-frame profile logging.
_PROFILE_LOG_INTERVAL = 5.0
_last_profile_log = 0.0


def _should_log_profile() -> bool:
    global _last_profile_log
    now = _time_mod.monotonic()
    if now - _last_profile_log >= _PROFILE_LOG_INTERVAL:
        _last_profile_log = now
        return True
    return False


# ── Model zoo ───────────────────────────────────────────────────────────────

HF_MODEL_ZOO: Dict[str, Any] = {
    "rt-detr-l": {
        "repo_id": "PekingU/rtdetr_r50vd_coco_o365",
        "label": "RT-DETR-L (ResNet-50)",
        "arch": "rt-detr",
        "dataset": "coco+objects365",
        "native_input_wh": (640, 640),
        "description": "Real-Time Detection Transformer with ResNet-50 backbone. "
                       "End-to-end transformer detector, no NMS needed.",
        "license": "Apache-2.0",
    },
    "rt-detr-x": {
        "repo_id": "PekingU/rtdetr_r101vd_coco_o365",
        "label": "RT-DETR-X (ResNet-101)",
        "arch": "rt-detr",
        "dataset": "coco+objects365",
        "native_input_wh": (640, 640),
        "description": "Real-Time Detection Transformer with ResNet-101 backbone. "
                       "Higher accuracy variant.",
        "license": "Apache-2.0",
    },
    "detr-resnet-50": {
        "repo_id": "facebook/detr-resnet-50",
        "label": "DETR (ResNet-50)",
        "arch": "detr",
        "dataset": "coco_hf",
        "native_input_wh": (800, 800),
        "description": "Original DEtection TRansformer with ResNet-50 backbone. "
                       "End-to-end set prediction, no NMS needed.",
        "license": "Apache-2.0",
    },
    "detr-resnet-101": {
        "repo_id": "facebook/detr-resnet-101",
        "label": "DETR (ResNet-101)",
        "arch": "detr",
        "dataset": "coco_hf",
        "native_input_wh": (800, 800),
        "description": "DETR with ResNet-101 backbone. Higher accuracy, heavier.",
        "license": "Apache-2.0",
    },
    "conditional-detr-resnet-50": {
        "repo_id": "microsoft/conditional-detr-resnet-50",
        "label": "Conditional DETR (ResNet-50)",
        "arch": "conditional-detr",
        "dataset": "coco_hf",
        "native_input_wh": (800, 800),
        "description": "Conditional DETR with faster convergence than vanilla DETR.",
        "license": "Apache-2.0",
    },
    "deformable-detr": {
        "repo_id": "SenseTime/deformable-detr",
        "label": "Deformable DETR",
        "arch": "deformable-detr",
        "dataset": "coco_hf",
        "native_input_wh": (800, 800),
        "description": "Deformable attention for efficient multi-scale detection.",
        "license": "Apache-2.0",
    },
    "yolos-tiny": {
        "repo_id": "hustvl/yolos-tiny",
        "label": "YOLOS Tiny",
        "arch": "yolos",
        "dataset": "coco_hf",
        "native_input_wh": (512, 512),
        "description": "YOLO-style detection with a Vision Transformer backbone. "
                       "Lightweight and fast.",
        "license": "Apache-2.0",
    },
    "yolos-small": {
        "repo_id": "hustvl/yolos-small",
        "label": "YOLOS Small",
        "arch": "yolos",
        "dataset": "coco_hf",
        "native_input_wh": (512, 512),
        "description": "YOLOS Small — balanced speed and accuracy.",
        "license": "Apache-2.0",
    },
    "yolos-base": {
        "repo_id": "hustvl/yolos-base",
        "label": "YOLOS Base",
        "arch": "yolos",
        "dataset": "coco_hf",
        "native_input_wh": (800, 800),
        "description": "YOLOS Base — higher accuracy than YOLOS Small, still ViT-based.",
        "license": "Apache-2.0",
    },
    "detr-resnet-50-dc5": {
        "repo_id": "facebook/detr-resnet-50-dc5",
        "label": "DETR DC5 (ResNet-50)",
        "arch": "detr",
        "dataset": "coco_hf",
        "native_input_wh": (800, 800),
        "description": "DETR with dilated C5 feature map — better small-object detection than vanilla DETR.",
        "license": "Apache-2.0",
    },
    "detr-resnet-101-dc5": {
        "repo_id": "facebook/detr-resnet-101-dc5",
        "label": "DETR DC5 (ResNet-101)",
        "arch": "detr",
        "dataset": "coco_hf",
        "native_input_wh": (800, 800),
        "description": "DETR DC5 with ResNet-101 backbone — highest accuracy pure-DETR variant.",
        "license": "Apache-2.0",
    },
    "rt-detr-nano": {
        "repo_id": "PekingU/rtdetr_r18vd",
        "label": "RT-DETR Nano (ResNet-18)",
        "arch": "rt-detr",
        "dataset": "coco",
        "native_input_wh": (640, 640),
        "description": "Lightest RT-DETR variant with ResNet-18 backbone. "
                       "Best choice for CPU / memory-constrained devices.",
        "license": "Apache-2.0",
    },
    # ── Instance segmentation ─────────────────────────────────────────────────
    "mask2former-swin-tiny-coco-instance": {
        "repo_id": "facebook/mask2former-swin-tiny-coco-instance",
        "label": "Mask2Former Swin-T (Instance)",
        "arch": "mask2former",
        "dataset": "coco",
        "native_input_wh": (800, 800),
        "hf_task": "instance_segmentation",
        "description": "Mask2Former with Swin-Tiny backbone for COCO instance segmentation. "
                       "Produces per-instance binary masks alongside bounding boxes.",
        "license": "Apache-2.0",
    },
    "mask2former-swin-small-coco-instance": {
        "repo_id": "facebook/mask2former-swin-small-coco-instance",
        "label": "Mask2Former Swin-S (Instance)",
        "arch": "mask2former",
        "dataset": "coco",
        "native_input_wh": (800, 800),
        "hf_task": "instance_segmentation",
        "description": "Mask2Former with Swin-Small backbone for COCO instance segmentation. "
                       "Higher accuracy than Swin-Tiny, ~2× slower on CPU.",
        "license": "Apache-2.0",
    },
}


# ── Engine implementation ───────────────────────────────────────────────────

class HuggingFaceEngine(InferenceEngine):

    @property
    def name(self) -> str:
        return 'huggingface'

    def available(self) -> bool:
        return _HAS_TRANSFORMERS

    def list_models(self) -> Dict[str, Any]:
        return HF_MODEL_ZOO

    def load_model(self, model_name: str, config=None) -> Dict[str, Any]:
        if not _HAS_TRANSFORMERS:
            raise RuntimeError(
                'HF Transformers is not installed. '
                'Install transformers for DETECT_BACKEND=huggingface.'
            )

        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        zoo_entry = HF_MODEL_ZOO.get(model_name, {})
        repo_id = zoo_entry.get('repo_id', model_name)
        native_wh = zoo_entry.get('native_input_wh', (640, 640))
        hf_task = zoo_entry.get('hf_task', 'object_detection')
        is_segmentation = hf_task == 'instance_segmentation'

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cache_dir = '/data/huggingface'

        logger.info('Loading HuggingFace model %s (repo: %s, task: %s) on %s...',
                    model_name, repo_id, hf_task, device)

        try:
            processor = AutoImageProcessor.from_pretrained(repo_id, cache_dir=cache_dir)
            if is_segmentation:
                from transformers import AutoModelForUniversalSegmentation
                model = AutoModelForUniversalSegmentation.from_pretrained(
                    repo_id, cache_dir=cache_dir)
            else:
                model = AutoModelForObjectDetection.from_pretrained(
                    repo_id, cache_dir=cache_dir)
            model = model.to(device)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                f'Failed to load HuggingFace model "{repo_id}": {exc}'
            ) from exc

        # Extract class names from model config.
        id2label = getattr(model.config, 'id2label', {})
        if id2label and config is not None and not config.class_names:
            config.class_names = [id2label.get(i, f'class_{i}') for i in sorted(id2label.keys())]

        logger.info('HuggingFace model %s loaded (%d classes)', model_name, len(id2label))

        return {
            'backend': 'huggingface',
            'model': model,
            'processor': processor,
            'model_name': model_name,
            'native_input_wh': native_wh,
            'is_segmentation': is_segmentation,
        }

    def infer(self, frame: np.ndarray, model_bundle: Dict[str, Any],
              confidence: float | None = None, iou: float | None = None,
              config=None) -> Optional[sv.Detections]:
        model = model_bundle['model']
        processor = model_bundle['processor']
        conf_threshold = confidence if confidence is not None else (config.conf if config else 0.1)
        class_list = (config.class_list if config else []) or []

        try:
            _t0 = _time_mod.monotonic()

            if model_bundle.get('is_segmentation'):
                result = self._infer_segmentation(frame, model_bundle, conf_threshold, class_list)
                _t1 = _time_mod.monotonic()
                if _should_log_profile():
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
            target_sizes = torch.tensor([frame.shape[:2]], device=device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=conf_threshold
            )

            if not results:
                return False

            result = results[0]  # batch size 1
            if len(result.get('scores', [])) == 0:
                return False

            id2label = getattr(model_bundle.get('model'), 'config', None)
            id2label = getattr(id2label, 'id2label', None) if id2label else None
            detections = sv.Detections.from_transformers(
                transformers_results=result,
                id2label=id2label,
            )

            if class_list:
                keep = np.isin(detections.class_id, class_list)
                detections = detections[keep]

            _t_post = _time_mod.monotonic()
            if _should_log_profile():
                logger.info('[PROFILE-HF] forward=%.0fms  post=%.0fms  total=%.0fms  frame=%dx%d',
                            (_t_forward - _t0) * 1000, (_t_post - _t_forward) * 1000,
                            (_t_post - _t0) * 1000, frame.shape[1], frame.shape[0])

            return detections if len(detections) > 0 else False

        except Exception as e:
            logger.error('HuggingFace inference failed: %s', e, exc_info=True)
            return False

    # ── Segmentation helper ─────────────────────────────────────────────────

    @staticmethod
    def _infer_segmentation(frame: np.ndarray, model_bundle: Dict[str, Any],
                            conf_threshold: float,
                            class_list: list) -> Optional[sv.Detections]:
        """Run Mask2Former / universal-segmentation inference.

        Uses per-instance binary masks (which can overlap) instead of
        the overlap-resolved segmentation map.
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

        class_logits = outputs.class_queries_logits[0]
        pred_probs = class_logits.softmax(-1)[:, :-1]
        pred_scores, pred_classes = pred_probs.max(-1)

        keep_idx = torch.where(pred_scores > conf_threshold)[0]
        if len(keep_idx) == 0:
            return False

        mask_logits = outputs.masks_queries_logits[0][keep_idx]
        mask_logits_full = _F.interpolate(
            mask_logits.unsqueeze(0).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        binary_masks = (mask_logits_full > 0).cpu().numpy()
        scores_np = pred_scores[keep_idx].cpu().numpy().astype(np.float32)
        classes_np = pred_classes[keep_idx].cpu().numpy().astype(np.int64)

        if class_list:
            keep_cls = np.isin(classes_np, class_list)
            binary_masks = binary_masks[keep_cls]
            scores_np = scores_np[keep_cls]
            classes_np = classes_np[keep_cls]
            if len(scores_np) == 0:
                return False

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
