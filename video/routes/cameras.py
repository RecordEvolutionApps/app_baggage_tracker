"""Camera listing routes."""
from __future__ import annotations

import subprocess

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/cameras")
def list_cameras():
    """List local cameras (USB, MIPI CSI, GMSL) with supported resolutions."""
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
            path = parts[0]
            name = parts[1]
            devpath = parts[2]
            resolutions_str = parts[3] if len(parts) > 3 else ""
            interface = parts[4] if len(parts) > 4 else "usb"
            dev_id = devpath.replace("/devices/platform/", "").split("/video4linux")[0]

            resolutions = []
            if resolutions_str:
                for res in resolutions_str.split(","):
                    res = res.strip()
                    if "x" in res:
                        try:
                            w, h = res.split("x")
                            resolutions.append({"width": int(w), "height": int(h)})
                        except ValueError:
                            continue

            cameras.append({
                "path": path,
                "name": name,
                "id": dev_id,
                "resolutions": resolutions,
                "interface": interface,
            })
        return cameras
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Camera listing timed out")
