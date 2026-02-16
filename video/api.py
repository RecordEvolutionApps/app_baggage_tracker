"""
Video Service API -- application entrypoint.

Creates the FastAPI app, includes route modules, and provides
the lifespan handler and health endpoint.

Start with:  uvicorn api:app
"""
from __future__ import annotations

import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routes.streams import router as streams_router, processes, delete_mediasoup_ingest
from routes.cameras import router as cameras_router
from routes.models import router as models_router


# -- Lifespan ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # On shutdown, kill all running streams and clean up mediasoup ingests
    for cam_stream, proc in processes.items():
        print(f"[api] Shutting down stream {cam_stream} (pid={proc.pid})")
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        delete_mediasoup_ingest(cam_stream)
    processes.clear()


# -- App ----------------------------------------------------------------------
app = FastAPI(title="Video Stream Service", lifespan=lifespan)

app.include_router(streams_router)
app.include_router(cameras_router)
app.include_router(models_router)


@app.get("/health")
def health():
    """Health check."""
    alive = {k: v.pid for k, v in processes.items() if v.poll() is None}
    return {"ok": True, "streams": alive}
