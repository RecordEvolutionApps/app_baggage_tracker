"""
Video Service API -- application entrypoint.

Creates the FastAPI app, includes route modules, and provides
the lifespan handler and health endpoint.

Start with:  uvicorn api:app
"""
from __future__ import annotations

import logging
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI


# -- Suppress noisy polling endpoints from Uvicorn access log ----------------
class _NoisyEndpointFilter(logging.Filter):
    """Drop access-log records for high-frequency polling routes."""
    _quiet_paths = ("/backend",)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._quiet_paths)

logging.getLogger("uvicorn.access").addFilter(_NoisyEndpointFilter())

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
