"""Model zoo data, checkpoint download/cache management, and backend status reporting.

The canonical model zoo dicts now live in their respective engine modules:
  - ``engines.huggingface.HF_MODEL_ZOO``
  - ``engines.mmdet.MMDET_MODEL_ZOO``

This module re-exports them (and related helpers) so that existing callers
such as ``routes/models.py`` and ``model_catalog.py`` continue to work.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger('model_zoo')

# ── Platform detection ──────────────────────────────────────────────────────
IS_AMD64 = platform.machine() in ('x86_64', 'AMD64')

try:
    import mmdet as _mmdet  # noqa: F401
    HAS_MMDET = True
except ImportError:
    HAS_MMDET = False

if IS_AMD64:
    HAS_MMDET = False

try:
    import transformers as _transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ── Re-export zoo dicts and utilities from engine modules ───────────────────
from engines.huggingface import HF_MODEL_ZOO  # noqa: F401
from engines.mmdet import (  # noqa: F401
    MMDET_MODEL_ZOO,
    download_file,
    _validate_checkpoint,
    _delete_corrupt_checkpoint,
    _download_checkpoint,
    _resolve_weight_url,
)


# ── Cache management ───────────────────────────────────────────────────────

def is_model_cached(model_name: str) -> bool:
    """Check whether a model's checkpoint is already available locally."""
    if model_name in ('none', ''):
        return True

    # Check MMDet checkpoint cache
    cache_root = Path('/data/mmdet')
    checkpoint_path = cache_root / 'checkpoints' / f'{model_name}.pth'
    if checkpoint_path.is_file():
        zoo_entry = MMDET_MODEL_ZOO.get(model_name, {})
        if not _validate_checkpoint(
            checkpoint_path,
            expected_size=zoo_entry.get('expected_size'),
            sha256_hex=zoo_entry.get('sha256'),
        ):
            _delete_corrupt_checkpoint(checkpoint_path)
            return False
        return True

    # Check HuggingFace cache
    hf_entry = HF_MODEL_ZOO.get(model_name, {})
    if hf_entry:
        repo_id = hf_entry.get('repo_id', model_name)
        hf_cache = Path('/data/huggingface')
        if hf_cache.is_dir():
            # HF Hub stores models in models--<org>--<name> dirs
            safe_name = repo_id.replace('/', '--')
            model_dir = hf_cache / f'models--{safe_name}'
            if model_dir.is_dir():
                # Check for at least one snapshot
                snapshots = model_dir / 'snapshots'
                if snapshots.is_dir() and any(snapshots.iterdir()):
                    return True

    # Check mmengine / torch hub default cache locations
    for cache_dir in [
        Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints',
        Path.home() / '.cache' / 'mim',
    ]:
        if cache_dir.is_dir():
            for f in cache_dir.iterdir():
                if model_name.replace('-', '_') in f.name.replace('-', '_'):
                    return True
    return False


def get_cached_models() -> list[dict]:
    """Return a list of all locally cached models with their on-disk sizes."""
    from model_catalog import discover_mmdet_models, discover_hf_models

    known_models: dict[str, str] = {}
    try:
        for m in discover_mmdet_models():
            mid = m.get('id', '')
            if mid and mid != 'none':
                known_models[mid.replace('-', '_').lower()] = mid
    except Exception:
        pass
    try:
        for m in discover_hf_models():
            mid = m.get('id', '')
            if mid and mid != 'none':
                known_models[mid.replace('-', '_').lower()] = mid
    except Exception:
        pass
    for mid in MMDET_MODEL_ZOO:
        known_models[mid.replace('-', '_').lower()] = mid

    results: dict[str, dict] = {}

    def _add(model_id: str, path: Path):
        if model_id not in results:
            results[model_id] = {'model': model_id, 'size_bytes': 0, 'locations': []}
        try:
            results[model_id]['size_bytes'] += path.stat().st_size
        except OSError:
            pass
        results[model_id]['locations'].append(str(path))

    # 1) /data/mmdet/checkpoints/*.pth
    ckpt_dir = Path('/data/mmdet/checkpoints')
    if ckpt_dir.is_dir():
        for f in ckpt_dir.iterdir():
            if f.is_file() and f.suffix == '.pth':
                model_id = f.stem
                _add(model_id, f)

    # 2) /data/tensorrt/ — .engine and .onnx
    trt_dir = Path(os.environ.get('TRT_CACHE_DIR', '/data/tensorrt'))
    if trt_dir.is_dir():
        for f in trt_dir.iterdir():
            if f.is_file() and f.suffix in ('.engine', '.onnx'):
                model_id = f.stem.replace('_fp16', '').replace('_fp32', '').replace('_int8', '')
                _add(model_id, f)

    # 3) ~/.cache/torch/hub/checkpoints/ and ~/.cache/mim/
    for cache_dir in [
        Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints',
        Path.home() / '.cache' / 'mim',
    ]:
        if not cache_dir.is_dir():
            continue
        for f in cache_dir.iterdir():
            if not f.is_file():
                continue
            fname_norm = f.name.replace('-', '_').lower()
            for key, model_id in known_models.items():
                if key in fname_norm:
                    _add(model_id, f)
                    break

    # 4) /data/huggingface/ — HuggingFace Hub cache
    hf_cache = Path('/data/huggingface')
    if hf_cache.is_dir():
        # Map repo_id → model_id for reverse lookup
        repo_to_id: dict[str, str] = {}
        for mid, entry in HF_MODEL_ZOO.items():
            safe = entry.get('repo_id', mid).replace('/', '--')
            repo_to_id[safe] = mid
        for d in hf_cache.iterdir():
            if d.is_dir() and d.name.startswith('models--'):
                safe_name = d.name[len('models--'):]
                model_id = repo_to_id.get(safe_name, safe_name.replace('--', '/'))
                # Sum up snapshot file sizes
                snapshots = d / 'snapshots'
                if snapshots.is_dir():
                    for snap in snapshots.iterdir():
                        if snap.is_dir():
                            for f in snap.rglob('*'):
                                if f.is_file():
                                    _add(model_id, f)

    return list(results.values())


def delete_cached_model(model_name: str) -> dict:
    """Delete all cached files for a specific model."""
    if model_name in ('none', ''):
        return {'deleted': []}

    deleted: list[str] = []
    norm = model_name.replace('-', '_').lower()

    ckpt = Path('/data/mmdet/checkpoints') / f'{model_name}.pth'
    if ckpt.is_file():
        ckpt.unlink()
        deleted.append(str(ckpt))

    trt_dir = Path(os.environ.get('TRT_CACHE_DIR', '/data/tensorrt'))
    if trt_dir.is_dir():
        for suffix in ('_fp16.engine', '_fp32.engine', '_int8.engine', '.engine', '.onnx'):
            p = trt_dir / f'{model_name}{suffix}'
            if p.is_file():
                p.unlink()
                deleted.append(str(p))

    for cache_dir in [
        Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints',
        Path.home() / '.cache' / 'mim',
    ]:
        if not cache_dir.is_dir():
            continue
        for f in list(cache_dir.iterdir()):
            if f.is_file() and norm in f.name.replace('-', '_').lower():
                f.unlink()
                deleted.append(str(f))

    logger.info('Deleted cached files for %s: %s', model_name, deleted)
    return {'deleted': deleted}


def clear_all_cache() -> dict:
    """Remove all cached model files."""
    deleted_count = 0
    freed_bytes = 0

    dirs_to_clean = [
        Path('/data/mmdet/checkpoints'),
        Path(os.environ.get('TRT_CACHE_DIR', '/data/tensorrt')),
        Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints',
        Path.home() / '.cache' / 'mim',
    ]

    for d in dirs_to_clean:
        if not d.is_dir():
            continue
        for f in list(d.iterdir()):
            if f.is_file():
                try:
                    freed_bytes += f.stat().st_size
                    f.unlink()
                    deleted_count += 1
                except OSError as e:
                    logger.warning('Failed to delete %s: %s', f, e)

    logger.info('Cleared all caches: %d files, %d bytes freed', deleted_count, freed_bytes)
    return {'deleted_count': deleted_count, 'freed_bytes': freed_bytes}


# ── Model preparation (download with progress) ─────────────────────────────

def prepare_model(model_name: str, progress_callback=None):
    """Download a model checkpoint if not cached. Returns when ready."""
    def _report(status, progress=0, message=''):
        if progress_callback:
            progress_callback(status, progress, message)

    _report('checking', 0, f'Checking cache for {model_name}...')

    if model_name in ('none', ''):
        _report('ready', 100, 'No model needed')
        return

    if is_model_cached(model_name):
        _report('ready', 100, f'{model_name} already cached')
        return

    _report('downloading', 0, f'Downloading {model_name}...')

    cache_root = Path('/data/mmdet')
    checkpoint_path = cache_root / 'checkpoints' / f'{model_name}.pth'

    if model_name in MMDET_MODEL_ZOO:
        zoo = MMDET_MODEL_ZOO[model_name]
        url = zoo['checkpoint']
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _download_with_progress(url, str(checkpoint_path), _report)
        if not _validate_checkpoint(
            checkpoint_path,
            expected_size=zoo.get('expected_size'),
            sha256_hex=zoo.get('sha256'),
        ):
            _delete_corrupt_checkpoint(checkpoint_path)
            _report('error', 0, f'{model_name}: download failed integrity check')
            raise RuntimeError(f'Downloaded checkpoint for {model_name} failed integrity validation')
        _report('ready', 100, f'{model_name} downloaded')
        return

    weight_url = _resolve_weight_url(model_name)
    if weight_url:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _report('downloading', 5, f'Downloading {model_name} from {weight_url[:80]}...')
        _download_with_progress(weight_url, str(checkpoint_path), _report)
        if not _validate_checkpoint(checkpoint_path):
            _delete_corrupt_checkpoint(checkpoint_path)
            _report('error', 0, f'{model_name}: download failed integrity check')
            raise RuntimeError(f'Downloaded checkpoint for {model_name} failed integrity validation')
        _report('ready', 100, f'{model_name} downloaded')
        return

    # Last resort: try DetInferencer auto-download (only if mmdet available)
    if HAS_MMDET:
        try:
            from mmdet.apis import DetInferencer
            device = 'cpu'
            _report('downloading', 10, f'Resolving {model_name} from DetInferencer...')
            DetInferencer(model=model_name, device=device)
            _report('ready', 100, f'{model_name} downloaded and cached')
            return
        except Exception as e:
            _report('error', 0, str(e))
            raise

    # HuggingFace model download
    if model_name in HF_MODEL_ZOO:
        zoo_entry = HF_MODEL_ZOO[model_name]
        repo_id = zoo_entry['repo_id']
        _report('downloading', 10, f'Downloading HF model {repo_id}...')
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id,
                cache_dir='/data/huggingface',
                ignore_patterns=['*.msgpack', '*.h5', 'tf_*', 'flax_*'],
            )
            _report('ready', 100, f'{model_name} downloaded from HuggingFace Hub')
            return
        except Exception as e:
            _report('error', 0, str(e))
            raise

    # Try HF if model_name looks like a repo ID (contains '/')
    if HAS_TRANSFORMERS and '/' in model_name:
        _report('downloading', 10, f'Downloading HF model {model_name}...')
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                model_name,
                cache_dir='/data/huggingface',
                ignore_patterns=['*.msgpack', '*.h5', 'tf_*', 'flax_*'],
            )
            _report('ready', 100, f'{model_name} downloaded from HuggingFace Hub')
            return
        except Exception as e:
            _report('error', 0, str(e))
            raise

    _report('error', 0, f'No download method available for {model_name}')
    raise RuntimeError(f'Cannot resolve model: {model_name}')


def _download_with_progress(url: str, dest: str, report_fn):
    """Download a file with progress reporting."""
    import urllib.request

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part = dest_path.with_suffix(dest_path.suffix + '.part')

    try:
        response = urllib.request.urlopen(url)
        total = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 256
        last_pct = -1

        with open(part, 'wb') as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct:
                        last_pct = pct
                        report_fn('downloading', pct, f'{downloaded}/{total} bytes')

        if total > 0 and downloaded != total:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f'Download size mismatch for {url}: expected {total} bytes, got {downloaded}'
            )

        shutil.move(str(part), str(dest_path))
        report_fn('downloading', 100, 'Download complete')
    except Exception:
        part.unlink(missing_ok=True)
        raise


# ── Backend status reporting ────────────────────────────────────────────────

def write_backend_status(cam_stream: str, model_bundle: Dict[str, Any],
                         extra: Dict[str, Any] | None = None,
                         detect_backend: str = 'mmdet',
                         config=None):
    """Update backend status on the config object (and optionally on disk for
    backwards compatibility).

    When *config* is supplied the status dict is stored on
    ``config.backend_status`` and no file is written.  The caller is
    responsible for publishing via ``publisher.publish_stream()``.
    """
    backend = model_bundle.get('backend', 'unknown')
    model_name = model_bundle.get('model_name', '')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if backend == 'tensorrt':
        precision = 'fp16'
        trt_cached = bool(model_bundle.get('trt_cached'))
        message = f'TensorRT FP16 engine active for {model_name}'
        if trt_cached:
            message += ' (loaded from cache)'
    elif backend == 'huggingface':
        precision = 'fp32'
        trt_cached = False
        message = f'HuggingFace Transformers (PyTorch FP32) for {model_name}'
    elif backend == 'mmdet':
        precision = 'fp32'
        trt_cached = False
        requested = detect_backend.lower()
        if requested == 'tensorrt':
            message = f'PyTorch FP32 fallback (TensorRT build failed) for {model_name}'
        else:
            message = f'PyTorch FP32 (MMDetection) for {model_name}'
    else:
        precision = 'n/a'
        trt_cached = False
        message = 'No inference backend'

    status = {
        'backend': backend,
        'model': model_name,
        'precision': precision,
        'device': device,
        'trt_cached': trt_cached,
        'requested_backend': detect_backend.lower(),
        'message': message,
    }
    if extra:
        status.update(extra)

    if config is not None:
        config.backend_status = status
    else:
        # Legacy file-based path (kept for callers that don't pass config)
        status_dir = Path('/data/status')
        status_dir.mkdir(parents=True, exist_ok=True)
        status_file = status_dir / f'{cam_stream}.backend.json'
        with open(status_file, 'w') as f:
            json.dump(status, f)
    logger.info('[status] %s: %s', cam_stream, message)


# ── TensorRT build (user-triggered) ────────────────────────────────────────

def build_trt_for_model(model_name: str, progress_callback=None):
    """Explicitly build a TensorRT engine for a model.

    Triggered by the user via the UI, not automatically on model switch.
    """
    def _report(status, progress=0, message=''):
        if progress_callback:
            progress_callback(status, progress, message)

    if model_name in ('none', ''):
        _report('error', 0, 'Cannot build TRT engine for "none"')
        return

    try:
        import trt_backend
    except ImportError:
        _report('error', 0, 'TensorRT backend module not available')
        return

    if not trt_backend.is_available():
        _report('error', 0, 'TensorRT or CUDA not available on this device')
        return

    if trt_backend.is_engine_cached(model_name):
        _report('ready', 100, f'TensorRT engine already cached for {model_name}')
        return

    _report('building', 5, f'Loading model {model_name}...')

    try:
        if HAS_MMDET and model_name in MMDET_MODEL_ZOO:
            from model_loader import get_mmdet_model
            source_bundle = get_mmdet_model(model_name)
        elif HAS_TRANSFORMERS and (model_name in HF_MODEL_ZOO or '/' in model_name):
            from model_loader import get_huggingface_model
            source_bundle = get_huggingface_model(model_name)
        elif HAS_MMDET:
            from model_loader import get_mmdet_model
            source_bundle = get_mmdet_model(model_name)
        else:
            _report('error', 0, 'No ML framework available for model loading')
            return
    except Exception as e:
        _report('error', 0, f'Failed to load model: {e}')
        return

    input_wh = source_bundle.get('native_input_wh', (640, 640))
    try:
        num_classes = source_bundle.get('inferencer', source_bundle.get('model', None))
        if hasattr(num_classes, 'bbox_head'):
            num_classes = num_classes.bbox_head.num_classes
        else:
            num_classes = 80
    except Exception:
        num_classes = 80

    _report('building', 15, 'Exporting to ONNX...')

    backend = source_bundle.get('backend', '')
    try:
        if backend == 'mmdet':
            onnx_file = trt_backend.export_to_onnx(model_name, source_bundle['inferencer'], input_wh)
        elif backend == 'huggingface':
            # HF models: export via torch.onnx.export
            onnx_file = _export_hf_to_onnx(model_name, source_bundle, input_wh)
        else:
            _report('error', 0, f'ONNX export not supported for backend: {backend}')
            return
        _report('building', 30, 'ONNX export complete. Building TensorRT FP16 engine (this may take 5-20 min)...')
    except Exception as e:
        _report('error', 0, f'ONNX export failed: {e}')
        return

    try:
        trt_backend.build_engine_from_onnx(model_name, onnx_file)
        _report('ready', 100, f'TensorRT FP16 engine built and cached for {model_name}')
    except Exception as e:
        _report('error', 0, f'TensorRT engine build failed: {e}')
        return

    try:
        del source_bundle
        torch.cuda.empty_cache()
    except Exception:
        pass


# ── HuggingFace ONNX export helper ─────────────────────────────────────────

def _export_hf_to_onnx(model_name: str, model_bundle: Dict[str, Any],
                        input_wh: tuple[int, int]) -> str:
    """Export a HuggingFace detection model to ONNX format.

    Returns the path to the exported .onnx file.
    """
    import numpy as np

    trt_cache = Path(os.environ.get('TRT_CACHE_DIR', '/data/tensorrt'))
    trt_cache.mkdir(parents=True, exist_ok=True)
    onnx_path = trt_cache / f'{model_name}.onnx'

    if onnx_path.is_file():
        logger.info('ONNX file already exists: %s', onnx_path)
        return str(onnx_path)

    model = model_bundle['model']
    processor = model_bundle['processor']

    # Create a dummy input through the processor
    w, h = input_wh
    dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    from PIL import Image
    pil_image = Image.fromarray(dummy_image)
    inputs = processor(images=pil_image, return_tensors='pt')

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    input_names = list(inputs.keys())
    output_names = ['logits', 'pred_boxes']

    logger.info('Exporting HF model %s to ONNX (%s)...', model_name, onnx_path)
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            name: {0: 'batch_size'} for name in input_names + output_names
        },
    )
    logger.info('ONNX export complete: %s', onnx_path)
    return str(onnx_path)
