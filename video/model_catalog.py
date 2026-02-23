"""
Model Catalog — architecture descriptions, config notation decoder,
model discovery from MMDetection metafiles, and dataset class lists.

Pure data & logic module with no FastAPI dependency.
"""
from __future__ import annotations

import logging
import os
import re
import threading

logger = logging.getLogger('model_catalog')

# ── Platform detection ──────────────────────────────────────────────────────
import platform as _platform
_IS_AMD64 = _platform.machine() in ('x86_64', 'AMD64')

# ── Detect available ML frameworks ──────────────────────────────────────────
try:
    import mmdet as _mmdet  # noqa: F401
    HAS_MMDET = True
except ImportError:
    HAS_MMDET = False

# amd64 builds use a modern stack that only supports HuggingFace, not MMDetection.
if _IS_AMD64:
    HAS_MMDET = False

try:
    import transformers as _transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ── Fix packaging.version on Python 3.8 ─────────────────────────────────────
# openmim 0.3.x → pkg_resources → packaging.version.Version can receive a
# non-string version object from certain setuptools builds, causing:
#   TypeError: expected string or bytes-like object
# Coercing to str() before the regex search fixes it everywhere.

def _patch_packaging_version():
    try:
        import packaging.version
        _orig_init = packaging.version.Version.__init__
        if getattr(_orig_init, '_patched', False):
            return  # already applied

        def _safe_init(self, version):
            return _orig_init(self, str(version) if version is not None else '0')
        _safe_init._patched = True
        packaging.version.Version.__init__ = _safe_init
    except Exception:
        pass

_patch_packaging_version()


# ── Dataset class lists ─────────────────────────────────────────────────────
# Standard COCO-2017 80-class list (used by virtually all COCO-pretrained models)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

# Map dataset keywords → class lists so we never need to load the model
DATASET_CLASSES: dict[str, list[str]] = {
    'coco': COCO_CLASSES,
}


# ── Helper utilities ────────────────────────────────────────────────────────

def classes_to_dicts(names: list[str]) -> list[dict]:
    return [{"id": idx, "name": name} for idx, name in enumerate(names)]


def extract_model_classes(inferencer) -> list[dict]:
    classes = None
    try:
        classes = inferencer.model.dataset_meta.get('classes')
    except Exception:
        classes = None
    if not classes:
        try:
            classes = inferencer.dataset_meta.get('classes')
        except Exception:
            classes = None
    if not classes:
        raise RuntimeError('Model classes not found in dataset metadata')

    return [{"id": idx, "name": str(name)} for idx, name in enumerate(classes)]


def head_content_length(url: str) -> int | None:
    """Return Content-Length from a HEAD request, or None on failure."""
    import urllib.request
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=5) as resp:
            cl = resp.headers.get('Content-Length')
            return int(cl) if cl else None
    except Exception:
        return None


# ── Architecture knowledge base ─────────────────────────────────────────────
# Maps architecture family keys (lowercase first segment of config name) to
# a human-readable explanation of what the architecture is, what it's good at,
# and typical use-cases.  Shown in the model-picker UI so users can make an
# educated choice.
ARCH_INFO: dict[str, dict[str, str]] = {
    'rtmdet': {
        'name': 'RTMDet (Real-Time Detection Transformer)',
        'description': (
            'A high-speed single-stage anchor-free detector optimised for real-time '
            'inference. Achieves a strong accuracy/speed trade-off by using a CSPNeXt '
            'backbone with large-kernel depth-wise convolutions and a task-aligned head.'
        ),
        'good_for': 'Real-time video surveillance, edge deployment, and latency-sensitive applications.',
        'tradeoff': 'Best balance of speed and accuracy among single-stage detectors.',
    },
    'faster': {
        'name': 'Faster R-CNN',
        'description': (
            'A classic two-stage detector: a Region Proposal Network (RPN) proposes '
            'candidate bounding boxes and a second stage classifies and refines them. '
            'One of the most studied architectures in object detection.'
        ),
        'good_for': 'General-purpose detection where accuracy matters more than raw speed.',
        'tradeoff': 'Higher accuracy than single-stage detectors but slower due to two-stage pipeline.',
    },
    'cascade': {
        'name': 'Cascade R-CNN',
        'description': (
            'Extends Faster R-CNN with multiple detection heads applied sequentially at '
            'increasing IoU thresholds. Each stage refines the proposals from the previous one, '
            'producing higher-quality bounding boxes.'
        ),
        'good_for': 'High-precision detection, overlapping objects, and applications needing tight bounding boxes.',
        'tradeoff': 'Highest box quality among R-CNN variants; heavier and slower than single-stage methods.',
    },
    'yolox': {
        'name': 'YOLOX',
        'description': (
            'An anchor-free version of the YOLO family with a decoupled head and '
            'strong data augmentation (Mosaic + MixUp). Balances speed and accuracy '
            'with a clean single-stage design.'
        ),
        'good_for': 'Real-time detection on GPUs; industrial inspection, robotics.',
        'tradeoff': 'Faster than two-stage detectors with competitive accuracy; less precise than cascade methods.',
    },
    'yolov3': {
        'name': 'YOLOv3',
        'description': (
            'Third generation of "You Only Look Once" — a single-stage detector that '
            'predicts bounding boxes and class probabilities in a single forward pass '
            'using feature pyramids at three scales.'
        ),
        'good_for': 'Legacy deployments and resource-constrained environments; well-understood baseline.',
        'tradeoff': 'Fast but less accurate than modern detectors; widely supported in deployment frameworks.',
    },
    'yolov5': {
        'name': 'YOLOv5',
        'description': (
            'A popular community iteration of YOLO with CSP backbone, PANet neck, and '
            'extensive augmentation. Known for ease of training and deployment.'
        ),
        'good_for': 'Quick prototyping, edge deployment, and production systems needing ONNX/TensorRT export.',
        'tradeoff': 'Excellent speed; slightly less accurate than larger models on complex scenes.',
    },
    'yolov6': {
        'name': 'YOLOv6',
        'description': (
            'A hardware-friendly redesign of YOLO with an EfficientRep backbone and '
            'Rep-PAN neck optimised for deployment on GPUs and edge hardware.'
        ),
        'good_for': 'Industrial deployment on GPUs (TensorRT) and edge accelerators.',
        'tradeoff': 'Designed for deployment efficiency; competitive accuracy with very fast inference.',
    },
    'yolov7': {
        'name': 'YOLOv7',
        'description': (
            'Extends YOLO with trainable bag-of-freebies techniques like planned '
            're-parameterised convolution and coarse-to-fine label assignment.'
        ),
        'good_for': 'High-speed detection on GPUs; competitive accuracy without pre-training on extra data.',
        'tradeoff': 'State-of-the-art speed/accuracy; requires careful tuning for custom datasets.',
    },
    'yolov8': {
        'name': 'YOLOv8 (Ultralytics)',
        'description': (
            'The latest Ultralytics YOLO generation — anchor-free with a C2f backbone block, '
            'decoupled head, and distribution focal loss. Supports detection, segmentation, '
            'pose estimation, and tracking out of the box.'
        ),
        'good_for': 'Versatile real-time detection, multi-task models, and rapid deployment.',
        'tradeoff': 'Very fast with strong accuracy; ecosystem lock-in to Ultralytics tooling.',
    },
    'detr': {
        'name': 'DETR (DEtection TRansformer)',
        'description': (
            'The first end-to-end transformer-based detector. Uses a set-based global loss '
            'with bipartite matching — eliminates the need for anchor boxes and NMS post-processing.'
        ),
        'good_for': 'Clean end-to-end detection without hand-crafted components; research and applications needing NMS-free pipelines.',
        'tradeoff': 'Simpler pipeline but slower convergence; lower speed than CNN-based detectors.',
    },
    'dino': {
        'name': 'DINO (DETR with Improved deNoising anchOr boxes)',
        'description': (
            'Improves DETR with denoising anchor boxes, a contrastive query selection strategy, '
            'and a mixed query approach. Achieves state-of-the-art detection accuracy.'
        ),
        'good_for': 'High-accuracy detection benchmarks; complex scenes with many overlapping objects.',
        'tradeoff': 'Top accuracy among transformer detectors; computationally expensive.',
    },
    'retinanet': {
        'name': 'RetinaNet',
        'description': (
            'A single-stage detector that introduced Focal Loss to address the class imbalance '
            'problem in dense detection. Uses a Feature Pyramid Network (FPN) for multi-scale features.'
        ),
        'good_for': 'Dense detection scenes, small objects, and a strong single-stage baseline.',
        'tradeoff': 'Good accuracy for single-stage; slower than YOLO variants but more robust to class imbalance.',
    },
    'fcos': {
        'name': 'FCOS (Fully Convolutional One-Stage)',
        'description': (
            'An anchor-free, per-pixel detector that predicts bounding boxes at every spatial '
            'location of the feature map. Uses center-ness scoring to suppress low-quality boxes.'
        ),
        'good_for': 'Anchor-free detection, dense scenes, and simplifying the detection pipeline.',
        'tradeoff': 'Clean design without anchor tuning; competitive accuracy with moderate speed.',
    },
    'centernet': {
        'name': 'CenterNet',
        'description': (
            'Detects objects as center keypoints and regresses width/height from the center. '
            'Eliminates anchors and NMS by design, treating detection as a keypoint estimation problem.'
        ),
        'good_for': 'Multi-object detection, pose estimation, and 3D detection.',
        'tradeoff': 'Simple and elegant; may struggle with very small or heavily overlapping objects.',
    },
    'ssd': {
        'name': 'SSD (Single Shot MultiBox Detector)',
        'description': (
            'A classic single-stage detector that predicts bounding boxes and scores at multiple '
            'feature map scales in a single pass. One of the earliest real-time detectors.'
        ),
        'good_for': 'Lightweight deployment, mobile/embedded systems, and legacy pipelines.',
        'tradeoff': 'Very fast; lower accuracy than modern detectors, especially on small objects.',
    },
    'atss': {
        'name': 'ATSS (Adaptive Training Sample Selection)',
        'description': (
            'Uses statistical properties of object features to automatically determine positive/negative '
            'training samples without hand-crafted rules. Built on top of a RetinaNet-like architecture.'
        ),
        'good_for': 'Improved detection baselines; research on better training sample strategies.',
        'tradeoff': 'Better accuracy than RetinaNet with same cost; good baseline for ablation studies.',
    },
    'gfl': {
        'name': 'GFL (Generalized Focal Loss)',
        'description': (
            'Extends focal loss to a continuous quality-estimation form, jointly learning '
            'localisation quality and classification in a unified representation.'
        ),
        'good_for': 'Dense detection with better localisation quality scoring.',
        'tradeoff': 'Improved quality over RetinaNet/ATSS; same inference speed.',
    },
    'sparse': {
        'name': 'Sparse R-CNN',
        'description': (
            'A fully sparse detector that uses a fixed set of learnable proposal boxes and '
            'features, avoiding dense candidate generation entirely. End-to-end trainable.'
        ),
        'good_for': 'Efficient detection without dense anchors or proposals; clean architecture.',
        'tradeoff': 'Elegant design; competitive accuracy with fewer hand-crafted components.',
    },
    'mask': {
        'name': 'Mask R-CNN',
        'description': (
            'Extends Faster R-CNN with a parallel branch that predicts instance segmentation masks '
            'alongside bounding boxes. The foundational model for instance segmentation.'
        ),
        'good_for': 'Instance segmentation, pixel-level object boundaries, and panoptic segmentation.',
        'tradeoff': 'Rich output (boxes + masks); slower than detection-only models.',
    },
    'grounding': {
        'name': 'Grounding DINO',
        'description': (
            'An open-vocabulary detector that combines DINO with grounded pre-training. Accepts '
            'free-text class descriptions and can detect arbitrary object categories without retraining.'
        ),
        'good_for': 'Zero-shot / open-vocabulary detection; detecting novel classes not in the training set.',
        'tradeoff': 'Extremely flexible; heavier model, slower inference than fixed-vocabulary detectors.',
    },
    'glip': {
        'name': 'GLIP (Grounded Language-Image Pre-training)',
        'description': (
            'Unifies phrase grounding and object detection via language-aware deep fusion. '
            'Learns region-word alignment and can detect objects from text prompts.'
        ),
        'good_for': 'Language-guided detection; detecting objects by description without fine-tuning.',
        'tradeoff': 'Powerful open-vocabulary capability; large model with higher compute requirements.',
    },
    'detic': {
        'name': 'Detic',
        'description': (
            'Detects 21,000+ concepts by training the classifier on image-level labels (ImageNet-21k) '
            'and the detector on base categories. Expands vocabulary massively with minimal annotation.'
        ),
        'good_for': 'Large-vocabulary detection; recognising rare or unusual object categories.',
        'tradeoff': 'Broad vocabulary coverage; accuracy on rare classes depends on image-level data quality.',
    },
    'yolo-world': {
        'name': 'YOLO-World',
        'description': (
            'An open-vocabulary version of YOLO that fuses vision and language features for real-time '
            'open-set detection. Combines YOLO speed with text-prompted class selection.'
        ),
        'good_for': 'Real-time open-vocabulary detection; fast deployment with flexible class prompts.',
        'tradeoff': 'Fast open-vocab inference; lower accuracy than larger grounding models.',
    },
    'convnext': {
        'name': 'ConvNeXt (Backbone)',
        'description': (
            'A modernised pure-ConvNet backbone inspired by Vision Transformer design choices. '
            'Often used as a drop-in backbone replacement for stronger feature extraction.'
        ),
        'good_for': 'Boosting detector accuracy with a stronger backbone; good ImageNet scaling.',
        'tradeoff': 'Better features than classic ResNet; slightly heavier than lightweight backbones.',
    },
    'swin': {
        'name': 'Swin Transformer (Backbone)',
        'description': (
            'A hierarchical Vision Transformer that computes self-attention within shifted windows. '
            'Widely used as a high-performance backbone for detection and segmentation.'
        ),
        'good_for': 'State-of-the-art detection accuracy when paired with strong detection heads.',
        'tradeoff': 'Excellent accuracy; higher memory and compute cost than CNN backbones.',
    },
    'co-detr': {
        'name': 'Co-DETR (Collaborative DETR)',
        'description': (
            'Enhances DETR training by enabling collaborative hybrid assignments from '
            'multiple parallel auxiliary heads, improving encoder feature learning.'
        ),
        'good_for': 'Top-tier detection accuracy on COCO and LVIS benchmarks.',
        'tradeoff': 'Very high accuracy; one of the most compute-intensive detectors.',
    },
    'ddq': {
        'name': 'DDQ (Dense Distinct Query)',
        'description': (
            'Proposes dense distinct queries for DETR-like detectors, initialising queries '
            'from dense priors and using distinct query selection to improve performance.'
        ),
        'good_for': 'Improving DETR convergence and accuracy with dense initialisation.',
        'tradeoff': 'Improved accuracy over base DETR; moderate additional cost.',
    },
    'tood': {
        'name': 'TOOD (Task-aligned One-stage Object Detection)',
        'description': (
            'Aligns classification and localisation tasks through a task-aligned head and '
            'task alignment learning. Improves single-stage detection by reducing task conflicts.'
        ),
        'good_for': 'High-quality single-stage detection with better task alignment.',
        'tradeoff': 'Better accuracy than FCOS/ATSS with similar speed.',
    },
    'vfnet': {
        'name': 'VFNet (VarifocalNet)',
        'description': (
            'Learns an IoU-aware classification score (varifocal representation) that jointly '
            'captures object presence confidence and localisation accuracy.'
        ),
        'good_for': 'Dense detection with improved scoring; better AP than GFL variants.',
        'tradeoff': 'Improved quality-aware scoring; same inference cost as standard dense detectors.',
    },
    'efficientdet': {
        'name': 'EfficientDet',
        'description': (
            'Uses compound scaling (jointly scaling backbone, feature network, and resolution) '
            'with a BiFPN feature pyramid for efficient multi-scale feature fusion.'
        ),
        'good_for': 'Scalable detection from mobile to cloud; efficient use of compute budget.',
        'tradeoff': 'Good accuracy-per-FLOP ratio; outperformed by newer architectures at same scale.',
    },
    'cornernet': {
        'name': 'CornerNet',
        'description': (
            'Detects objects as pairs of top-left and bottom-right corner keypoints, '
            'eliminating anchor boxes entirely. Pioneered keypoint-based detection.'
        ),
        'good_for': 'Anchor-free detection research; pairing with other keypoint methods.',
        'tradeoff': 'Novel approach; slower inference and higher memory than anchor-based detectors.',
    },
    'foveabox': {
        'name': 'FoveaBox',
        'description': (
            'An anchor-free detector that directly predicts category-sensitive semantic maps '
            'and category-agnostic bounding boxes for each location.'
        ),
        'good_for': 'Simple anchor-free detection pipeline.',
        'tradeoff': 'Clean anchor-free design; similar performance to FCOS.',
    },
    'libra': {
        'name': 'Libra R-CNN',
        'description': (
            'Addresses sample, feature, and objective imbalance in detection by introducing '
            'balanced sampling, balanced feature pyramid, and balanced L1 loss.'
        ),
        'good_for': 'Improving detection by addressing training imbalances.',
        'tradeoff': 'Better accuracy over Faster R-CNN baseline with minimal overhead.',
    },
    'paa': {
        'name': 'PAA (Probabilistic Anchor Assignment)',
        'description': (
            'Uses a probabilistic model to adaptively separate positive and negative anchors '
            'during training based on the model\'s own predictions.'
        ),
        'good_for': 'Improved anchor assignment in dense detectors; better training dynamics.',
        'tradeoff': 'Better AP than ATSS with same architecture; negligible inference overhead.',
    },
    'sabl': {
        'name': 'SABL (Side-Aware Boundary Localisation)',
        'description': (
            'Reformulates bounding box regression as classification of bucketed offsets '
            'for each side of the box, improving localisation accuracy.'
        ),
        'good_for': 'Applications requiring precise bounding box boundaries.',
        'tradeoff': 'Improved localisation accuracy; slightly more complex head design.',
    },
    'ghm': {
        'name': 'GHM (Gradient Harmonizing Mechanism)',
        'description': (
            'Harmonises gradient contributions from easy and hard examples during training, '
            'reducing the dominance of very easy negatives without hard mining.'
        ),
        'good_for': 'Stabilising training dynamics in dense detectors.',
        'tradeoff': 'Improved training stability; drop-in replacement for standard loss functions.',
    },
    'deformable': {
        'name': 'Deformable DETR',
        'description': (
            'Replaces DETR\'s global attention with deformable attention that only attends '
            'to a few key sampling points, dramatically improving training convergence and enabling '
            'efficient multi-scale feature processing.'
        ),
        'good_for': 'Faster DETR convergence and better small-object detection via multi-scale features.',
        'tradeoff': 'Much faster training than vanilla DETR; requires deformable attention CUDA ops.',
    },
    'conditional': {
        'name': 'Conditional DETR',
        'description': (
            'Reduces DETR training epochs by learning a conditional spatial query from the '
            'decoder embedding, providing better spatial priors for cross-attention.'
        ),
        'good_for': 'Faster DETR training convergence (reducing from 500 to 50 epochs).',
        'tradeoff': 'Same accuracy as DETR with 6-10x faster convergence.',
    },
    'dab': {
        'name': 'DAB-DETR (Dynamic Anchor Boxes)',
        'description': (
            'Uses dynamic anchor boxes as queries in DETR, providing better positional priors '
            'and enabling the decoder to iteratively refine box predictions.'
        ),
        'good_for': 'Better DETR convergence with interpretable anchor-based queries.',
        'tradeoff': 'Improved convergence and accuracy over vanilla DETR.',
    },
}


# ── Tag system ───────────────────────────────────────────────────────────────
# Tags are organized by dimension so users can filter models without knowing
# architecture internals.  Each model gets tags auto-assigned based on its
# architecture family, config name, and dataset.

TAG_DIMENSIONS = {
    'task':        'What the model does',
    'output':      'What kind of results it produces',
    'speed':       'Performance profile (speed vs quality)',
    'capability':  'Special abilities',
    'domain':      'What it was trained on',
}

# Architecture → tags mapping.  Keys are the arch family (first segment of
# config name).  Values list the tags that apply.
ARCH_TAGS: dict[str, list[str]] = {
    # ── Single-stage detectors (fast) ──
    'rtmdet':      ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:real-time'],
    'yolox':       ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:real-time'],
    'yolov3':      ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:real-time'],
    'yolov5':      ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:real-time'],
    'yolov6':      ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:real-time'],
    'yolov7':      ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:real-time'],
    'yolov8':      ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:real-time'],
    'ssd':         ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:real-time'],
    'retinanet':   ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'fcos':        ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'centernet':   ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'atss':        ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'gfl':         ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'tood':        ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'vfnet':       ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'paa':         ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'efficientdet': ['task:object-detection', 'output:bounding-box', 'speed:balanced'],

    # ── Two-stage / high-accuracy detectors ──
    'faster':      ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
    'cascade':     ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
    'libra':       ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
    'sparse':      ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
    'sabl':        ['task:object-detection', 'output:bounding-box', 'speed:accurate'],

    # ── Transformer-based detectors ──
    'detr':        ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
    'dino':        ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
    'deformable':  ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'conditional': ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'dab':         ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'co-detr':     ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
    'ddq':         ['task:object-detection', 'output:bounding-box', 'speed:accurate'],

    # ── Instance segmentation ──
    'mask':        ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:accurate'],
    'mask2former': ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:accurate'],
    'maskformer':  ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:accurate'],
    'queryinst':   ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:accurate'],
    'condinst':    ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:balanced'],
    'solov2':      ['task:instance-segmentation', 'output:mask', 'speed:balanced'],
    'solo':        ['task:instance-segmentation', 'output:mask', 'speed:balanced'],
    'pointrend':   ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:accurate'],
    'htc':         ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:accurate'],
    'scnet':       ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:accurate'],
    'panoptic':    ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:accurate'],
    'rtmdet-ins':  ['task:instance-segmentation', 'task:object-detection', 'output:mask', 'output:bounding-box', 'speed:fast', 'capability:real-time'],

    # ── Open-vocabulary / grounding ──
    'grounding':   ['task:object-detection', 'output:bounding-box', 'speed:accurate', 'capability:open-vocabulary'],
    'glip':        ['task:object-detection', 'output:bounding-box', 'speed:accurate', 'capability:open-vocabulary'],
    'detic':       ['task:object-detection', 'output:bounding-box', 'speed:balanced', 'capability:open-vocabulary'],
    'yolo-world':  ['task:object-detection', 'output:bounding-box', 'speed:fast', 'capability:open-vocabulary', 'capability:real-time'],

    # ── Keypoint / anchor-free research ──
    'cornernet':   ['task:object-detection', 'output:bounding-box', 'output:keypoints', 'speed:accurate'],
    'foveabox':    ['task:object-detection', 'output:bounding-box', 'speed:balanced'],
    'ghm':         ['task:object-detection', 'output:bounding-box', 'speed:balanced'],

    # ── Backbones (these are typically paired with a detection head) ──
    'convnext':    ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
    'swin':        ['task:object-detection', 'output:bounding-box', 'speed:accurate'],
}

# Dataset → domain tags
DATASET_TAGS: dict[str, str] = {
    'coco':        'domain:general-purpose',
    'voc':         'domain:general-purpose',
    'lvis':        'domain:general-purpose',
    'objects365':  'domain:general-purpose',
    'crowdhuman':  'domain:people',
    'wider':       'domain:faces',
    'cityscapes':  'domain:traffic',
    'openimages':  'domain:general-purpose',
}

# Size-variant → speed override (smaller variants are faster)
SIZE_SPEED_OVERRIDE: dict[str, str] = {
    'nano': 'speed:fast',
    'tiny': 'speed:fast',
    'small': 'speed:fast',
    's': 'speed:fast',
    'medium': 'speed:balanced',
    'm': 'speed:balanced',
    'large': 'speed:accurate',
    'l': 'speed:accurate',
    'xlarge': 'speed:accurate',
    'x': 'speed:accurate',
}


def compute_model_tags(arch: str, config_name: str, dataset: str, is_open_vocab: bool) -> list[str]:
    """Compute the full tag list for a model based on its metadata."""
    tags: list[str] = []

    # 1. Architecture-based tags
    arch_lower = arch.lower()
    base_tags = ARCH_TAGS.get(arch_lower, ['task:object-detection', 'output:bounding-box'])
    tags.extend(base_tags)

    # 2. Override speed tag based on model size variant
    parts = re.split(r'[_\-]', config_name.lower())
    for p in parts:
        if p in SIZE_SPEED_OVERRIDE:
            # Remove any existing speed tag and replace
            override = SIZE_SPEED_OVERRIDE[p]
            tags = [t for t in tags if not t.startswith('speed:')]
            tags.append(override)
            break

    # 3. Dataset / domain tags
    dataset_lower = dataset.lower() if dataset else ''
    for kw, tag in DATASET_TAGS.items():
        if kw in dataset_lower:
            tags.append(tag)
            break
    else:
        if dataset_lower:
            tags.append('domain:general-purpose')

    # 4. Instance segmentation detection from config name and architecture
    config_lower = config_name.lower()
    seg_keywords = ('mask', 'seg', 'segm', 'panoptic', 'ins_', '-ins-', '-ins_', '_ins-', '_ins_')
    seg_arch_keywords = ('mask', 'solo', 'htc', 'scnet', 'pointrend', 'queryinst', 'condinst', 'panoptic')
    has_seg = (
        any(kw in config_lower for kw in seg_keywords)
        or any(kw in arch_lower for kw in seg_arch_keywords)
        # RTMDet instance segmentation variants: rtmdet-ins*
        or (arch_lower.startswith('rtmdet') and 'ins' in config_lower)
    )
    if has_seg:
        if 'task:instance-segmentation' not in tags:
            tags.append('task:instance-segmentation')
        if 'output:mask' not in tags:
            tags.append('output:mask')

    # 5. Open vocabulary
    if is_open_vocab and 'capability:open-vocabulary' not in tags:
        tags.append('capability:open-vocabulary')

    # 6. Multi-class capability (COCO and larger datasets)
    if any(kw in dataset_lower for kw in ('coco', 'lvis', 'objects365', 'openimages')):
        tags.append('capability:multi-class')

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


def get_all_tags(models: list[dict]) -> dict[str, list[str]]:
    """Extract all unique tags from a model list, grouped by dimension."""
    by_dim: dict[str, set[str]] = {dim: set() for dim in TAG_DIMENSIONS}
    for m in models:
        for tag in m.get('tags', []):
            dim, _, value = tag.partition(':')
            if dim in by_dim:
                by_dim[dim].add(tag)
    return {dim: sorted(tags) for dim, tags in by_dim.items() if tags}


# ── Config notation decoder ─────────────────────────────────────────────────

def decode_config_notation(config_name: str) -> str:
    """Decode the cryptic notation in MMDetection config names into plain English.

    Examples:
      rtmdet_tiny_8xb32-300e_coco  →
        "tiny: very small/fast model variant | 8xb32: trained with 8 GPUs and
         batch-size 32 per GPU | 300e: trained for 300 epochs | coco: trained
         on the COCO dataset"
    """
    explanations: list[str] = []

    # ── Size variants ──
    size_map = {
        'nano': 'nano — smallest and fastest variant, minimal accuracy',
        'tiny': 'tiny — very small and fast model variant',
        'small': 'small — compact model, good speed/accuracy balance',
        's': None,  # handled specially below
        'medium': 'medium — mid-size model, balanced speed and accuracy',
        'm': None,
        'large': 'large — bigger model with higher accuracy, slower inference',
        'l': None,
        'xlarge': 'xlarge — extra-large, maximum accuracy at the cost of speed',
        'x': None,
    }
    # Check for explicit size words in config name segments
    parts = re.split(r'[_\-]', config_name.lower())
    for p in parts:
        if p in ('nano', 'tiny', 'small', 'medium', 'large', 'xlarge'):
            explanations.append(size_map[p])
            break
        # Single-letter sizes only if the whole segment is one letter
        if p in ('s',) and len(p) == 1:
            explanations.append('s — small model variant')
            break
        if p in ('m',) and len(p) == 1:
            explanations.append('m — medium model variant')
            break
        if p in ('l',) and len(p) == 1:
            explanations.append('l — large model variant')
            break
        if p in ('x',) and len(p) == 1:
            explanations.append('x — extra-large model variant')
            break

    # ── Training schedule: NxbM (GPUs × batch-size) ──
    m = re.search(r'(\d+)xb(\d+)', config_name)
    if m:
        gpus, bs = m.group(1), m.group(2)
        explanations.append(f'{gpus}xb{bs} — trained with {gpus} GPUs × batch-size {bs} per GPU')

    # ── Epochs: <N>e ──
    m = re.search(r'(?:^|[\-_])(\d+)e(?:[\-_]|$)', config_name)
    if m:
        epochs = m.group(1)
        explanations.append(f'{epochs}e — trained for {epochs} epochs')

    # ── LR schedule shorthand: 1x, 2x, 3x (mmdet convention: 1x=12 epochs) ──
    m = re.search(r'(?:^|[\-_])([123])x(?:[\-_]|$)', config_name)
    if m:
        factor = int(m.group(1))
        total = factor * 12
        explanations.append(f'{factor}x — {factor}× training schedule ({total} epochs)')

    # ── Multi-scale training ──
    if re.search(r'[\-_]ms[\-_]|[\-_]ms$', config_name):
        explanations.append('ms — multi-scale training (random image resizing for robustness)')

    # ── Backbone variants ──
    backbone_patterns = [
        (r'r50', 'r50 — ResNet-50 backbone (50 layers)'),
        (r'r101', 'r101 — ResNet-101 backbone (101 layers)'),
        (r'x101', 'x101 — ResNeXt-101 backbone'),
        (r'r2[\-_]101', 'r2-101 — Res2Net-101 backbone'),
        (r'swin[\-_]t', 'swin-t — Swin-Tiny transformer backbone'),
        (r'swin[\-_]s', 'swin-s — Swin-Small transformer backbone'),
        (r'swin[\-_]b', 'swin-b — Swin-Base transformer backbone'),
        (r'swin[\-_]l', 'swin-l — Swin-Large transformer backbone'),
        (r'convnext[\-_]t', 'convnext-t — ConvNeXt-Tiny backbone'),
        (r'convnext[\-_]s', 'convnext-s — ConvNeXt-Small backbone'),
        (r'convnext[\-_]b', 'convnext-b — ConvNeXt-Base backbone'),
        (r'convnext[\-_]l', 'convnext-l — ConvNeXt-Large backbone'),
        (r'pvt[\-_]v2[\-_]b\d', 'PVTv2 — Pyramid Vision Transformer v2 backbone'),
        (r'efficientnet', 'EfficientNet backbone'),
        (r'mobilenet', 'MobileNet backbone (lightweight, mobile-friendly)'),
        (r'regnet', 'RegNet backbone (systematically designed)'),
        (r'res2net', 'Res2Net backbone (multi-scale feature extraction)'),
        (r'hrnet', 'HRNet backbone (high-resolution representations)'),
    ]
    for pat, desc in backbone_patterns:
        if re.search(pat, config_name, re.IGNORECASE):
            explanations.append(desc)
            break

    # ── Neck variants ──
    if 'fpn' in config_name.lower():
        explanations.append('fpn — Feature Pyramid Network for multi-scale feature fusion')
    if 'pafpn' in config_name.lower() or 'nasfpn' in config_name.lower():
        explanations.append('pafpn/nasfpn — enhanced feature pyramid with path aggregation')
    if 'bifpn' in config_name.lower():
        explanations.append('bifpn — Bi-directional Feature Pyramid Network')

    # ── Other common tokens ──
    if 'dcn' in config_name.lower():
        explanations.append('dcn — Deformable Convolutions (learnable receptive field shapes)')
    if 'syncbn' in config_name.lower():
        explanations.append('syncbn — Synchronised Batch Normalisation across GPUs')
    if 'caffe' in config_name.lower():
        explanations.append('caffe — Caffe-style image preprocessing (BGR, different normalisation)')
    if 'softmax' in config_name.lower():
        explanations.append('softmax — softmax classification head')
    if re.search(r'[\-_]gc[\-_]', config_name):
        explanations.append('gc — Global Context module for enhanced feature aggregation')
    if 'mdpool' in config_name.lower():
        explanations.append('mdpool — multi-dilated pooling')

    # ── Dataset ──
    dataset_map = {
        'coco': 'coco — trained on COCO dataset (80 common object classes)',
        'voc': 'voc — trained on PASCAL VOC dataset (20 classes)',
        'lvis': 'lvis — trained on LVIS dataset (1,200+ fine-grained categories)',
        'objects365': 'objects365 — trained on Objects365 dataset (365 categories)',
        'crowdhuman': 'crowdhuman — trained on CrowdHuman dataset (pedestrian detection)',
        'wider': 'wider — trained on WIDER Face dataset (face detection)',
        'openimages': 'openimages — trained on Open Images dataset',
        'cityscapes': 'cityscapes — trained on Cityscapes dataset (urban driving scenes)',
    }
    for key, desc in dataset_map.items():
        if key in config_name.lower():
            explanations.append(desc)
            break

    return ' | '.join(explanations) if explanations else ''


def build_model_description(arch_key: str, config_name: str, architecture_field: str) -> str:
    """Combine architecture info and decoded config notation into a rich description."""
    sections: list[str] = []

    # Look up architecture info — try the exact key first, then the first
    # segment of the config name
    info = ARCH_INFO.get(arch_key.lower())
    if not info:
        # Try matching by prefix of common multi-word arch keys
        for key in ARCH_INFO:
            if arch_key.lower().startswith(key) or key.startswith(arch_key.lower()):
                info = ARCH_INFO[key]
                break

    if info:
        sections.append(f"Architecture: {info['name']}")
        sections.append(info['description'])
        sections.append(f"Best for: {info['good_for']}")
        sections.append(f"Trade-off: {info['tradeoff']}")
    elif architecture_field:
        sections.append(f"Architecture: {architecture_field}")

    notation = decode_config_notation(config_name)
    if notation:
        sections.append(f"Config notation: {notation}")

    return '\n'.join(sections)


# ── File size fetcher ───────────────────────────────────────────────────────

def populate_file_sizes(models: list[dict], url_map: dict[str, str] | None = None) -> None:
    """Fetch checkpoint file sizes for all models using concurrent HEAD requests."""
    from concurrent.futures import ThreadPoolExecutor

    if url_map is None:
        url_map = {}
        for m in models:
            model_id = m.get('id', '')
            url = m.get('_weight_url', '')
            if model_id and url and url.startswith('http'):
                url_map[model_id] = url

    if not url_map:
        return

    results: dict[str, int | None] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(head_content_length, url): model_id for model_id, url in url_map.items()}
        for future in futures:
            model_id = futures[future]
            results[model_id] = future.result()

    size_by_id = {m.get('id', ''): m for m in models}
    for model_id, size_bytes in results.items():
        if size_bytes and size_bytes > 0:
            # Match pretty-bytes (decimal) units used in download progress.
            size_mb = round(size_bytes / 1_000_000, 1)
            if model_id in size_by_id:
                size_by_id[model_id]['fileSize'] = size_mb


# ── Model discovery ─────────────────────────────────────────────────────────

def discover_mmdet_models() -> list[dict]:
    """
    Discover COCO-pretrained detection models from MMDetection's metafile
    registry via openmim. Results are cached after first call.
    Returns an empty list if MMDetection is not installed.
    """
    if hasattr(discover_mmdet_models, '_cache'):
        return discover_mmdet_models._cache

    if not HAS_MMDET:
        discover_mmdet_models._cache = []
        return []

    models: list[dict] = []

    try:
        from mim.commands.search import get_model_info
        # Field names are lowercase in openmim 0.3.x:
        #   model, weight, config, training_data, architecture, paper, readme
        df = get_model_info(
            'mmdet',
            shown_fields=[
                'model',
                'weight',
                'config',
                'training_data',
                'architecture',
                'paper',
                'readme',
            ],
        )
        seen: set[str] = set()
        for _, row in df.iterrows():
            import pandas as pd

            config_path = row.get('config', '')
            weight = row.get('weight', '')

            # Handle NaN values from pandas properly
            training_data_raw = row.get('training_data', '')
            training_data = str(training_data_raw).lower() if pd.notna(training_data_raw) else ''

            model_label_raw = row.get('model', '')
            model_label = str(model_label_raw) if pd.notna(model_label_raw) else ''

            architecture_raw = row.get('architecture', '')
            architecture = str(architecture_raw) if pd.notna(architecture_raw) else ''

            readme_raw = row.get('readme', '')
            readme = str(readme_raw) if pd.notna(readme_raw) else ''

            paper_raw = row.get('paper', '')
            # Extract real paper URL (skip NaN and placeholder 'URL,Title')
            paper = ''
            if pd.notna(paper_raw):
                paper_str = str(paper_raw).strip()
                if paper_str and paper_str != 'URL,Title' and ('http://' in paper_str or 'https://' in paper_str):
                    # If there's a comma, take the first part (URL)
                    paper = paper_str.split(',')[0].strip()

            # Fallback: generate GitHub link to model README or config
            if not paper and readme:
                paper = f"https://github.com/open-mmlab/mmdetection/tree/main/{readme}"
            elif not paper and config_path:
                paper = f"https://github.com/open-mmlab/mmdetection/tree/main/{config_path}"

            # Use model label and architecture as summary/description
            if model_label and architecture and training_data:
                # Replace commas with comma-space for readability
                arch_display = architecture.replace(',', ', ')
                summary = f"{model_label} is a {arch_display} model trained on {training_data.upper()}."
            else:
                summary = model_label

            # Only include models that have a checkpoint
            if not config_path or not weight:
                continue

            # The config basename (without .py) is the model ID that
            # DetInferencer accepts, e.g. "rtmdet_tiny_8xb32-300e_coco"
            name = os.path.basename(config_path).replace('.py', '')
            if not name or name in seen:
                continue
            seen.add(name)

            # Build human-readable label from config name
            parts = name.replace('_', ' ').replace('-', ' ').split()
            label = ' '.join(parts).title()
            # Architecture family = first segment before '_'
            arch = name.split('_')[0] if '_' in name else name.split('-')[0]

            # Detect open-vocabulary models by architecture/config name
            ov_keywords = ('grounding-dino', 'grounding_dino', 'glip', 'detic', 'yolo-world', 'yolo_world')
            is_open_vocab = any(kw in name.lower() for kw in ov_keywords) or any(kw in architecture.lower() for kw in ov_keywords)

            if is_open_vocab:
                label = f'{label} (Open Vocab)'

            description = build_model_description(arch, name, architecture)

            tags = compute_model_tags(arch, name, training_data, is_open_vocab)

            models.append({
                'id': name,
                'label': label,
                'arch': arch,
                'dataset': training_data,
                'architecture': architecture,
                'task': 'object_detection',
                'paper': paper,
                'summary': summary,
                'description': description,
                'openVocab': is_open_vocab,
                'tags': tags,
                '_weight_url': weight if pd.notna(weight) else '',
            })

        # Fetch checkpoint file sizes in background (don't block the response)
        try:
            from model_zoo import MMDET_MODEL_ZOO
        except Exception:
            MMDET_MODEL_ZOO = {}

        size_url_map = {}
        for m in models:
            model_id = m.get('id', '')
            curated = MMDET_MODEL_ZOO.get(model_id, {}).get('checkpoint')
            url = curated or m.get('_weight_url', '')
            if model_id and url and url.startswith('http'):
                size_url_map[model_id] = url

        def _bg_populate():
            try:
                populate_file_sizes(models, size_url_map)
            except Exception:
                pass
        threading.Thread(target=_bg_populate, daemon=True).start()

        # Sort by architecture, then by name
        models.sort(key=lambda m: (m['arch'], m['id']))
        logger.info('Discovered %d MMDetection COCO models from metafiles', len(models))

    except Exception as e:
        logger.error('mim-based model discovery failed: %s, falling back to curated list', e, exc_info=True)
        # Fallback: return the curated list from model_zoo
        try:
            from model_zoo import MMDET_MODEL_ZOO
            models = [
                {
                    'id': k,
                    'label': k.replace('_', ' ').replace('-', ' ').title(),
                    'arch': k.split('_')[0],
                    'dataset': 'coco',
                    'architecture': 'object detection',
                    'task': 'object_detection',
                    'paper': '',
                    'summary': f"{k.replace('_', ' ').title()} trained on COCO dataset",
                    'description': build_model_description(k.split('_')[0], k, 'object detection'),
                    'tags': compute_model_tags(k.split('_')[0], k, 'coco', False),
                }
                for k in MMDET_MODEL_ZOO
            ]
        except ImportError:
            models = [
                {
                    'id': 'rtmdet_tiny_8xb32-300e_coco',
                    'label': 'RTMDet Tiny',
                    'arch': 'rtmdet',
                    'dataset': 'coco',
                    'architecture': 'real-time detection',
                    'task': 'object_detection',
                    'paper': '',
                    'summary': 'RTMDet Tiny is a real-time detection model trained on COCO.',
                    'description': build_model_description('rtmdet', 'rtmdet_tiny_8xb32-300e_coco', 'real-time detection'),
                    'tags': compute_model_tags('rtmdet', 'rtmdet_tiny_8xb32-300e_coco', 'coco', False),
                },
                {
                    'id': 'rtmdet_s_8xb32-300e_coco',
                    'label': 'RTMDet Small',
                    'arch': 'rtmdet',
                    'dataset': 'coco',
                    'architecture': 'real-time detection',
                    'task': 'object_detection',
                    'paper': '',
                    'summary': 'RTMDet Small is a real-time detection model trained on COCO.',
                    'description': build_model_description('rtmdet', 'rtmdet_s_8xb32-300e_coco', 'real-time detection'),
                    'tags': compute_model_tags('rtmdet', 'rtmdet_s_8xb32-300e_coco', 'coco', False),
                },
                {
                    'id': 'rtmdet_m_8xb32-300e_coco',
                    'label': 'RTMDet Medium',
                    'arch': 'rtmdet',
                    'dataset': 'coco',
                    'architecture': 'real-time detection',
                    'task': 'object_detection',
                    'paper': '',
                    'summary': 'RTMDet Medium is a real-time detection model trained on COCO.',
                    'description': build_model_description('rtmdet', 'rtmdet_m_8xb32-300e_coco', 'real-time detection'),
                    'tags': compute_model_tags('rtmdet', 'rtmdet_m_8xb32-300e_coco', 'coco', False),
                },
            ]

    # Remove internal _weight_url key before caching
    for m in models:
        m.pop('_weight_url', None)

    discover_mmdet_models._cache = models
    return models


def discover_hf_models() -> list[dict]:
    """Return the curated HuggingFace model list for the UI/API.

    Each entry mirrors the shape of discover_mmdet_models entries so the
    frontend can treat them uniformly.
    """
    if hasattr(discover_hf_models, '_cache'):
        return discover_hf_models._cache

    try:
        from model_zoo import HF_MODEL_ZOO
    except ImportError:
        discover_hf_models._cache = []
        return []

    models: list[dict] = []
    for model_id, entry in HF_MODEL_ZOO.items():
        arch = entry.get('arch', model_id.split('-')[0])
        dataset = entry.get('dataset', 'coco')
        label = entry.get('label', model_id.replace('-', ' ').title())
        description = entry.get('description', '')

        # Build tags the same way MMDet models do
        tags = compute_model_tags(arch, model_id, dataset, False)
        # Add backend tag so the UI can distinguish
        tags.append('backend:huggingface')

        models.append({
            'id': model_id,
            'label': label,
            'arch': arch,
            'dataset': dataset,
            'architecture': f'{arch} (HuggingFace)',
            'task': 'object_detection',
            'paper': f'https://huggingface.co/{entry.get("repo_id", model_id)}',
            'summary': description,
            'description': description,
            'openVocab': False,
            'tags': tags,
            'backend': 'huggingface',
            'license': entry.get('license', ''),
        })

    models.sort(key=lambda m: (m['arch'], m['id']))
    logger.info('Discovered %d HuggingFace detection models', len(models))
    discover_hf_models._cache = models
    return models


def discover_all_models() -> list[dict]:
    """Return the union of MMDet + HuggingFace models for the API."""
    mmdet = discover_mmdet_models()
    hf = discover_hf_models()
    # Add backend tag to mmdet entries if missing
    for m in mmdet:
        if 'backend' not in m:
            m['backend'] = 'mmdet'
    return mmdet + hf


def get_model_classes(model_id: str, model_classes_cache: dict[str, list[dict]]) -> list[dict]:
    """Return detection class list for a model, using cache and dataset heuristics."""
    if model_id in ('none', ''):
        return []
    if model_id in model_classes_cache:
        return model_classes_cache[model_id]

    # Fast path: if the model name contains a known dataset keyword, return
    # the class list directly.  This covers the vast majority of models
    # and avoids loading the entire model.
    model_id_lower = model_id.lower()
    for keyword, class_list in DATASET_CLASSES.items():
        if keyword in model_id_lower:
            result = classes_to_dicts(class_list)
            model_classes_cache[model_id] = result
            return result

    # Fast path #2: look up dataset directly from HF_MODEL_ZOO metadata.
    # This avoids downloading the full model just to read id2label when the
    # dataset is already known (e.g. detr-resnet-50 → dataset='coco').
    try:
        from model_zoo import HF_MODEL_ZOO
        zoo_entry = HF_MODEL_ZOO.get(model_id, {})
        dataset_str = zoo_entry.get('dataset', '').lower()
        for keyword, class_list in DATASET_CLASSES.items():
            if keyword in dataset_str:
                result = classes_to_dicts(class_list)
                model_classes_cache[model_id] = result
                return result
    except Exception:
        pass

    # Secondary fast path: check discovered model metadata (if already cached)
    all_models = []
    if hasattr(discover_mmdet_models, '_cache'):
        all_models.extend(discover_mmdet_models._cache)
    if hasattr(discover_hf_models, '_cache'):
        all_models.extend(discover_hf_models._cache)
    for m in all_models:
        if m['id'] == model_id:
            dataset = m.get('dataset', '').lower()
            for keyword, class_list in DATASET_CLASSES.items():
                if keyword in dataset:
                    result = classes_to_dicts(class_list)
                    model_classes_cache[model_id] = result
                    return result
            break

    # HuggingFace model: load and inspect id2label from config only (no weights)
    try:
        from model_zoo import HF_MODEL_ZOO
        if model_id in HF_MODEL_ZOO or '/' in model_id:
            if HAS_TRANSFORMERS:
                from transformers import AutoConfig
                zoo_entry = HF_MODEL_ZOO.get(model_id, {})
                repo_id = zoo_entry.get('repo_id', model_id)
                cfg = AutoConfig.from_pretrained(repo_id, cache_dir='/data/huggingface')
                id2label = getattr(cfg, 'id2label', {})
                if id2label:
                    classes = [{'id': i, 'name': id2label[i]} for i in sorted(id2label.keys())]
                    model_classes_cache[model_id] = classes
                    return classes
    except Exception as e:
        logger.warning('Failed to get classes from HF config %s: %s', model_id, e)

    # Slow path: actually load via MMDetection (non-standard dataset)
    if HAS_MMDET:
        from model_loader import get_mmdet_model
        bundle = get_mmdet_model(model_id)
        inferencer = bundle.get('inferencer')
        if inferencer is None:
            raise RuntimeError('MMDetection inferencer unavailable')
        classes = extract_model_classes(inferencer)
        model_classes_cache[model_id] = classes
        return classes

    # Default: return COCO classes
    result = classes_to_dicts(COCO_CLASSES)
    model_classes_cache[model_id] = result
    return result
