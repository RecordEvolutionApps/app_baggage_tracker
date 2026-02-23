"""Model routes — listing, class queries, status checks, and preparation."""
from __future__ import annotations

import asyncio
import json
import logging
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from model_catalog import discover_all_models, get_model_classes, get_all_tags

logger = logging.getLogger('routes.models')

router = APIRouter()

# ── In-memory class cache (shared across requests) ─────────────────────────
_MODEL_CLASSES_CACHE: dict[str, list[dict]] = {}


# ── Request models ──────────────────────────────────────────────────────────

class PrepareRequest(BaseModel):
    model: str


class BuildTrtRequest(BaseModel):
    model: str


class ValidateRequest(BaseModel):
    model: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/models/validate")
def validate_model(req: ValidateRequest):
    """Check whether a model is usable with the currently installed backends.

    Returns:
      {"valid": true,  "backend": "huggingface", "model": "..."}  on success
      {"valid": false, "backend": "mmdet",       "model": "...", "reason": "..."}  on failure
    """
    import torch
    from model_zoo import MMDET_MODEL_ZOO, HF_MODEL_ZOO, IS_AMD64

    try:
        import mmdet  # noqa: F401
        has_mmdet = True
    except ImportError:
        has_mmdet = False

    # amd64 builds use a modern stack that only supports HuggingFace.
    if IS_AMD64:
        has_mmdet = False

    try:
        import transformers  # noqa: F401
        has_transformers = True
    except ImportError:
        has_transformers = False

    has_cuda = torch.cuda.is_available()
    model = req.model

    # Determine the model's native backend
    if model in MMDET_MODEL_ZOO:
        required_backend = 'mmdet'
        if not has_mmdet:
            return {
                "valid": False,
                "model": model,
                "backend": required_backend,
                "reason": (
                    f"Model '{model}' requires MMDetection which is not installed "
                    "on this device. Choose a HuggingFace model instead."
                ),
            }
    elif model in HF_MODEL_ZOO or '/' in model:
        required_backend = 'huggingface'
        if not has_transformers:
            return {
                "valid": False,
                "model": model,
                "backend": required_backend,
                "reason": (
                    f"Model '{model}' requires HuggingFace Transformers which is "
                    "not installed on this device."
                ),
            }
    else:
        # Unknown model — reject immediately so it doesn't cause a runtime error
        return {
            "valid": False,
            "model": model,
            "backend": "unknown",
            "reason": (
                f"Model '{model}' was not found in any known model zoo "
                "(MMDetection or HuggingFace). "
                "Please select a model from the available list."
            ),
        }

    return {"valid": True, "model": model, "backend": required_backend}


@router.get("/models")
def list_models():
    """Return available detection models (MMDetection + HuggingFace)."""
    return discover_all_models()


@router.get("/models/tags")
def list_tags():
    """Return all available tags grouped by dimension, for building filter UI."""
    models = discover_all_models()
    return get_all_tags(models)


# ── Cache management (must be before {model_id} parameterised routes) ───────

@router.get("/models/cache")
def list_cached_models():
    """Return all locally cached models with their on-disk sizes."""
    from model_zoo import get_cached_models
    return get_cached_models()


@router.delete("/models/cache")
def clear_cache():
    """Delete ALL cached model files (checkpoints + TensorRT engines)."""
    from model_zoo import clear_all_cache
    return clear_all_cache()


@router.delete("/models/{model_id}/cache")
def delete_model_cache(model_id: str):
    """Delete cached files for a specific model."""
    from model_zoo import delete_cached_model
    return delete_cached_model(model_id)


@router.get("/models/{model_id}/classes")
def model_classes(model_id: str):
    """Return the detection class list for a specific model."""
    try:
        return get_model_classes(model_id, _MODEL_CLASSES_CACHE)
    except Exception as exc:
        logger.error("Failed to load classes for '%s': %s", model_id, exc, exc_info=True)
        raise HTTPException(500, f"Could not load classes for '{model_id}': {exc}")


@router.get("/models/{model_id}/status")
def model_status(model_id: str):
    """Check whether a model's checkpoint is already cached locally."""
    from model_zoo import is_model_cached
    cached = is_model_cached(model_id)
    return {"model": model_id, "cached": cached}


@router.post("/models/prepare")
async def prepare_model_endpoint(req: PrepareRequest):
    """Download a model checkpoint if not cached, streaming progress via SSE.

    The response is a text/event-stream with JSON events:
      data: {"status": "checking", "progress": 0, "message": "..."}
      data: {"status": "downloading", "progress": 45, "message": "..."}
      data: {"status": "ready", "progress": 100, "message": "..."}
      data: {"status": "error", "progress": 0, "message": "..."}
    """
    model_name = req.model

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_callback(status: str, progress: int, message: str):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"status": status, "progress": progress, "message": message},
            )

        def run_prepare():
            try:
                from model_zoo import prepare_model
                prepare_model(model_name, progress_callback=progress_callback)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"status": "error", "progress": 0, "message": str(e)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = threading.Thread(target=run_prepare, daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/models/build-trt")
async def build_trt_endpoint(req: BuildTrtRequest):
    """Build a TensorRT FP16 engine for a model, streaming progress via SSE.

    The response is a text/event-stream with JSON events:
      data: {"status": "building", "progress": 15, "message": "..."}
      data: {"status": "ready", "progress": 100, "message": "..."}
      data: {"status": "error", "progress": 0, "message": "..."}
    """
    model_name = req.model

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_callback(status: str, progress: int, message: str):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"status": status, "progress": progress, "message": message},
            )

        def run_build():
            try:
                from model_zoo import build_trt_for_model
                build_trt_for_model(model_name, progress_callback=progress_callback)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"status": "error", "progress": 0, "message": str(e)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = threading.Thread(target=run_build, daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
