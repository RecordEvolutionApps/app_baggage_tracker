"""
Video Service API — replaces SSH-based stream management.

Provides HTTP endpoints for starting/stopping video streams
and listing USB cameras. The web backend calls these endpoints
instead of spawning SSH commands.
"""

import os
import sys
import signal
import subprocess
import asyncio
import urllib.parse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Configuration ───────────────────────────────────────────────────────────
MEDIASOUP_URL = os.environ.get("MEDIASOUP_URL", "http://127.0.0.1:1200")

# ── State ───────────────────────────────────────────────────────────────────
# camStream → subprocess.Popen
processes: dict[str, subprocess.Popen] = {}


# ── Lifespan ────────────────────────────────────────────────────────────────
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
        _delete_mediasoup_ingest(cam_stream)
    processes.clear()


app = FastAPI(title="Video Stream Service", lifespan=lifespan)


# ── Models ──────────────────────────────────────────────────────────────────
class StreamRequest(BaseModel):
    camPath: str
    camStream: str


class MaskUpdate(BaseModel):
    polygons: list = []
    selectedPolygonId: int | None = None


# ── Helpers ─────────────────────────────────────────────────────────────────
def _create_mediasoup_ingest(stream_id: str) -> int:
    """Ask mediasoup to create (or return existing) ingest and return the RTP port."""
    import urllib.request
    import json
    body = json.dumps({"streamId": stream_id}).encode()
    req = urllib.request.Request(
        f"{MEDIASOUP_URL}/ingest",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        return data["port"]


def _delete_mediasoup_ingest(stream_id: str):
    """Tell mediasoup to tear down the ingest for this stream."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{MEDIASOUP_URL}/ingest/{urllib.parse.quote(stream_id, safe='')}",
            method="DELETE",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[api] Could not delete mediasoup ingest for {stream_id}: {e}")


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.post("/streams")
def start_stream(req: StreamRequest):
    """Start a video stream process for the given camera."""
    if req.camStream in processes:
        proc = processes[req.camStream]
        if proc.poll() is None:  # still running
            raise HTTPException(409, f"Stream {req.camStream} is already running (pid={proc.pid})")
        else:
            # Process exited, remove stale entry
            processes.pop(req.camStream, None)

    # Ask mediasoup to create (or reuse) an ingest and give us the RTP port
    try:
        port = _create_mediasoup_ingest(req.camStream)
    except Exception as e:
        raise HTTPException(502, f"Could not create mediasoup ingest: {e}")

    cmd = ["python3", "-u", "/app/video/videoStream.py", req.camPath, req.camStream, "--port", str(port)]

    print(f"[api] Starting stream: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr,
        stdin=subprocess.PIPE,
    )
    processes[req.camStream] = proc

    return {
        "status": "started",
        "camStream": req.camStream,
        "pid": proc.pid,
        "port": port,
    }


@app.delete("/streams/{cam_stream}")
def stop_stream(cam_stream: str):
    """Stop a running video stream and clean up mediasoup ingest."""
    proc = processes.pop(cam_stream, None)
    if not proc:
        raise HTTPException(404, f"No running stream for {cam_stream}")

    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    _delete_mediasoup_ingest(cam_stream)

    return {"status": "stopped", "camStream": cam_stream}


@app.get("/streams")
def list_streams():
    """List all running video streams."""
    alive = {}
    dead = []
    for cam_stream, proc in processes.items():
        if proc.poll() is None:
            alive[cam_stream] = proc.pid
        else:
            dead.append(cam_stream)
    # Clean up dead processes
    for cam_stream in dead:
        processes.pop(cam_stream, None)
    return {"streams": alive}


@app.get("/cameras")
def list_cameras():
    """List USB cameras attached to the system."""
    try:
        result = subprocess.run(
            ["/app/video/list-cameras.sh"],
            capture_output=True, text=True, timeout=10,
        )
        cameras = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            path, name, devpath = parts[0], parts[1], parts[2]
            dev_id = devpath.replace("/devices/platform/", "").split("/video4linux")[0]
            cameras.append({"path": path, "name": name, "id": dev_id})
        return cameras
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Camera listing timed out")


@app.post("/streams/{cam_stream}/mask")
def update_mask(cam_stream: str, mask: MaskUpdate):
    """Send updated mask data to a running stream via stdin."""
    proc = processes.get(cam_stream)
    if not proc or proc.poll() is not None:
        raise HTTPException(404, f"No running stream for {cam_stream}")

    try:
        import json
        data = json.dumps(mask.model_dump()) + "\n"
        proc.stdin.write(data.encode())
        proc.stdin.flush()
        return {"status": "ok", "camStream": cam_stream}
    except Exception as e:
        raise HTTPException(500, f"Failed to send mask: {e}")


@app.get("/health")
def health():
    """Health check."""
    alive = {k: v.pid for k, v in processes.items() if v.poll() is None}
    return {"ok": True, "streams": alive}
