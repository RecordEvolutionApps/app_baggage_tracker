"""Camera listing routes."""
from __future__ import annotations

import subprocess

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/cameras")
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
