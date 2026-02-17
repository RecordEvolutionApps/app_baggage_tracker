"""Model routes — listing, class queries, status checks, and preparation."""
from __future__ import annotations

import asyncio
import json
import logging
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from model_catalog import discover_mmdet_models, get_model_classes, get_all_tags

logger = logging.getLogger('routes.models')

router = APIRouter()

# ── In-memory class cache (shared across requests) ─────────────────────────
_MODEL_CLASSES_CACHE: dict[str, list[dict]] = {}


# ── Request models ──────────────────────────────────────────────────────────

class PrepareRequest(BaseModel):
    model: str


class BuildTrtRequest(BaseModel):
    model: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/models")
def list_models():
    """Return available MMDetection models discovered from the installed package."""
    return discover_mmdet_models()


@router.get("/models/tags")
def list_tags():
    """Return all available tags grouped by dimension, for building filter UI."""
    models = discover_mmdet_models()
    return get_all_tags(models)


# ── Cache management (must be before {model_id} parameterised routes) ───────

@router.get("/models/cache")
def list_cached_models():
    """Return all locally cached models with their on-disk sizes."""
    from model_utils import get_cached_models
    return get_cached_models()


@router.delete("/models/cache")
def clear_cache():
    """Delete ALL cached model files (checkpoints + TensorRT engines)."""
    from model_utils import clear_all_cache
    return clear_all_cache()


@router.delete("/models/{model_id}/cache")
def delete_model_cache(model_id: str):
    """Delete cached files for a specific model."""
    from model_utils import delete_cached_model
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
    from model_utils import is_model_cached
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
                from model_utils import prepare_model
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
                from model_utils import build_trt_for_model
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
