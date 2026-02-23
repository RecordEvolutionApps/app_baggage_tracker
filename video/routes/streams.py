"""Stream management routes — start, stop, list, and backend status."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import urllib.parse
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ── Configuration ───────────────────────────────────────────────────────────
MEDIASOUP_URL = os.environ.get("MEDIASOUP_URL", "http://127.0.0.1:1200")

# ── Shared state ────────────────────────────────────────────────────────────
# camStream → subprocess.Popen  (shared with the main app for lifespan cleanup)
processes: dict[str, subprocess.Popen] = {}


# ── Request models ──────────────────────────────────────────────────────────
class StreamRequest(BaseModel):
    camPath: str
    camStream: str
    width: Optional[int] = None
    height: Optional[int] = None


# ── Mediasoup helpers ───────────────────────────────────────────────────────

def create_mediasoup_ingest(stream_id: str) -> int:
    """Ask mediasoup to create (or return existing) ingest and return the RTP port."""
    import urllib.request
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


def delete_mediasoup_ingest(stream_id: str):
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

@router.post("/streams")
def start_stream(req: StreamRequest):
    """Start a video stream process for the given camera."""
    if req.camStream in processes:
        proc = processes[req.camStream]
        if proc.poll() is None:  # still running
            # Kill existing stream so we can restart cleanly
            print(f"[api] Killing existing stream {req.camStream} (pid={proc.pid}) before restart")
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        processes.pop(req.camStream, None)

    # Clean up any stale mediasoup ingest before creating a new one
    delete_mediasoup_ingest(req.camStream)

    # Ask mediasoup to create (or reuse) an ingest and give us the RTP port
    try:
        port = create_mediasoup_ingest(req.camStream)
    except Exception as e:
        raise HTTPException(502, f"Could not create mediasoup ingest: {e}")

    cmd = ["python3", "-u", "/app/video/videoStream.py", req.camPath, req.camStream, "--port", str(port)]

    if req.width and req.height:
        cmd.extend(["--width", str(req.width), "--height", str(req.height)])

    print(f"[api] Starting stream: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    processes[req.camStream] = proc

    return {
        "status": "started",
        "camStream": req.camStream,
        "pid": proc.pid,
        "port": port,
    }


@router.post("/streams/{cam_stream}/stop")
def stop_stream(cam_stream: str):
    """Stop a running video stream (keeps it stoppable, not deleted)."""
    proc = processes.pop(cam_stream, None)
    if not proc:
        raise HTTPException(404, f"No running stream for {cam_stream}")

    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    delete_mediasoup_ingest(cam_stream)

    return {"status": "stopped", "camStream": cam_stream}


@router.delete("/streams/{cam_stream}")
def delete_stream(cam_stream: str):
    """Delete a running video stream and clean up mediasoup ingest."""
    proc = processes.pop(cam_stream, None)
    if not proc:
        raise HTTPException(404, f"No running stream for {cam_stream}")

    try:
        # Signal the stream process that this is an explicit deletion
        proc.send_signal(signal.SIGUSR1)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    delete_mediasoup_ingest(cam_stream)

    return {"status": "deleted", "camStream": cam_stream}


@router.get("/streams")
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


@router.get("/streams/{cam_stream}/backend")
def stream_backend_status(cam_stream: str):
    """Return the inference backend status for a running stream.

    Reads from /data/status/<camStream>.backend.json written by the
    video process at startup.
    """
    status_file = f'/data/status/{cam_stream}.backend.json'
    if not os.path.isfile(status_file):
        return {
            "backend": "unknown",
            "model": "",
            "precision": "n/a",
            "device": "unknown",
            "trt_cached": False,
            "message": "Stream has not reported backend status yet",
        }
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read backend status: {e}")
