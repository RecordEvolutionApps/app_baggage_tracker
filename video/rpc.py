"""WAMP RPC registration for the video service.

Registers device-scoped functions so the frontend can call them
directly via IronFlock instead of proxying through the web backend.
"""
from __future__ import annotations

import asyncio
import logging
import threading

logger = logging.getLogger('rpc')

# ── In-memory class cache (shared with FastAPI routes) ──────────────────────
from routes.models import _MODEL_CLASSES_CACHE


async def register_all(ironflock) -> None:
    """Register all device RPCs on the given IronFlock instance."""

    # ── Model catalog ───────────────────────────────────────────────────
    from model_catalog import discover_all_models, get_model_classes, get_all_tags

    async def rpc_get_models():
        return discover_all_models()

    async def rpc_get_model_tags():
        models = discover_all_models()
        return get_all_tags(models)

    async def rpc_get_model_classes(model_id: str):
        return get_model_classes(model_id, _MODEL_CLASSES_CACHE)

    # ── Model status & cache ────────────────────────────────────────────
    from model_zoo import is_model_cached, get_cached_models, delete_cached_model, clear_all_cache

    async def rpc_get_model_status(model_id: str):
        return {"model": model_id, "cached": is_model_cached(model_id)}

    async def rpc_get_cached_models():
        return get_cached_models()

    async def rpc_delete_cached_model(model_id: str):
        return delete_cached_model(model_id)

    async def rpc_clear_all_cache():
        return clear_all_cache()

    # ── Camera listing ──────────────────────────────────────────────────
    from routes.cameras import list_cameras

    async def rpc_list_cameras():
        return list_cameras()

    # ── Model validation ────────────────────────────────────────────────
    from routes.models import ValidateRequest

    async def rpc_validate_model(model: str):
        from routes.models import validate_model
        req = ValidateRequest(model=model)
        return validate_model(req)

    # ── Model preparation (progressive results) ────────────────────────
    async def rpc_prepare_model(model_name: str, details=None):
        from model_zoo import prepare_model
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_callback(status: str, progress: int, message: str):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"status": status, "progress": progress, "message": message},
            )

        def run_prepare():
            try:
                prepare_model(model_name, progress_callback=progress_callback)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"status": "error", "progress": 0, "message": str(e)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=run_prepare, daemon=True)
        thread.start()

        final_event = None
        while True:
            event = await queue.get()
            if event is None:
                break
            final_event = event
            if details and hasattr(details, 'progress'):
                details.progress(event)

        return final_event or {"status": "ready", "progress": 100, "message": "Done"}

    # ── TensorRT build (progressive results) ────────────────────────────
    async def rpc_build_trt(model_name: str, details=None):
        from model_zoo import build_trt_for_model
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_callback(status: str, progress: int, message: str):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"status": status, "progress": progress, "message": message},
            )

        def run_build():
            try:
                build_trt_for_model(model_name, progress_callback=progress_callback)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"status": "error", "progress": 0, "message": str(e)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=run_build, daemon=True)
        thread.start()

        final_event = None
        while True:
            event = await queue.get()
            if event is None:
                break
            final_event = event
            if details and hasattr(details, 'progress'):
                details.progress(event)

        return final_event or {"status": "ready", "progress": 100, "message": "Done"}

    # ── Stream lifecycle ───────────────────────────────────────────────
    from routes.streams import (
        StreamRequest,
        start_stream as _start_stream,
        stop_stream as _stop_stream,
        delete_stream as _delete_stream,
    )

    def _resolve_cam_path(source: dict) -> str:
        """Build the camera path string from a source config dict.

        Mirrors the TypeScript backend's credential-injection logic so that
        RTSP passwords never need to leave the server side.
        """
        src_type = source.get('type', '')
        raw_path = source.get('path', '')
        if src_type == 'IP' and raw_path and '://' in raw_path:
            protocol, path = raw_path.split('://', 1)
            username = source.get('username', '')
            password = source.get('password', '')
            userpw = ''
            if username:
                userpw = f'{username}:{password}@' if password else f'{username}@'
            return f'{protocol}://{userpw}{path}'
        return raw_path

    async def rpc_start_stream(cam_stream: str, source: dict | None = None):
        source = source or {}
        cam_path = _resolve_cam_path(source)
        req = StreamRequest(
            camPath=cam_path,
            camStream=cam_stream,
            width=source.get('width'),
            height=source.get('height'),
        )
        return _start_stream(req)

    async def rpc_stop_stream(cam_stream: str):
        try:
            return _stop_stream(cam_stream)
        except Exception:
            # Stream may not be running — return gracefully
            return {"status": "stopped", "camStream": cam_stream}

    async def rpc_delete_stream(cam_stream: str):
        try:
            return _delete_stream(cam_stream)
        except Exception:
            # Stream may not be running — return gracefully
            return {"status": "deleted", "camStream": cam_stream}

    # ── Register all RPCs ───────────────────────────────────────────────
    await ironflock.register_device_function('getModels', rpc_get_models)
    await ironflock.register_device_function('getModelTags', rpc_get_model_tags)
    await ironflock.register_device_function('getModelClasses', rpc_get_model_classes)
    await ironflock.register_device_function('getModelStatus', rpc_get_model_status)
    await ironflock.register_device_function('getCachedModels', rpc_get_cached_models)
    await ironflock.register_device_function('deleteCachedModel', rpc_delete_cached_model)
    await ironflock.register_device_function('clearAllCache', rpc_clear_all_cache)
    await ironflock.register_device_function('listCameras', rpc_list_cameras)
    await ironflock.register_device_function('validateModel', rpc_validate_model)
    await ironflock.register_device_function('prepareModel', rpc_prepare_model,
                                             {'details_arg': 'details'})
    await ironflock.register_device_function('buildTrt', rpc_build_trt,
                                             {'details_arg': 'details'})
    await ironflock.register_device_function('startStream', rpc_start_stream)
    await ironflock.register_device_function('stopStream', rpc_stop_stream)
    await ironflock.register_device_function('deleteStream', rpc_delete_stream)

    logger.info('Registered %d WAMP device RPCs', 14)
