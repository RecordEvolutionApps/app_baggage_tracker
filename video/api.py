"""
Video Service API — replaces SSH-based stream management.

Provides HTTP endpoints for starting/stopping video streams
and listing USB cameras. The web backend calls these endpoints
instead of spawning SSH commands.
"""

import os
import sys
import json
import signal
import subprocess
import asyncio
import urllib.parse
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

# ── Model classes cache ────────────────────────────────────────────────────
_MODEL_CLASSES_CACHE: dict[str, list[dict]] = {}


def _extract_model_classes(inferencer) -> list[dict]:
    classes = None
    try:
        classes = inferencer.model.dataset_meta.get('classes')
    except Exception:
        classes = None
    if not classes:
        try:
            classes = inferencer.dataset_meta.get('classes')
        except Exception:
            classes = None
    if not classes:
        raise RuntimeError('Model classes not found in dataset metadata')

    return [{"id": idx, "name": str(name)} for idx, name in enumerate(classes)]


def _get_model_classes(model_id: str) -> list[dict]:
    if model_id in ('none', ''):
        return []
    if model_id in _MODEL_CLASSES_CACHE:
        return _MODEL_CLASSES_CACHE[model_id]

    from model_utils import get_mmdet_model
    bundle = get_mmdet_model(model_id)
    inferencer = bundle.get('inferencer')
    if inferencer is None:
        raise RuntimeError('MMDetection inferencer unavailable')
    classes = _extract_model_classes(inferencer)
    _MODEL_CLASSES_CACHE[model_id] = classes
    return classes


# ── Models ──────────────────────────────────────────────────────────────────
class StreamRequest(BaseModel):
    camPath: str
    camStream: str



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
    _delete_mediasoup_ingest(req.camStream)

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


@app.get("/models")
def list_models():
    """Return available MMDetection models discovered from the installed package."""
    return _discover_mmdet_models()


@app.get("/models/{model_id}/classes")
def get_model_classes(model_id: str):
    """Return the detection class list for a specific model."""
    try:
        return _get_model_classes(model_id)
    except Exception as exc:
        raise HTTPException(500, f"Could not load classes for '{model_id}': {exc}")


def _head_content_length(url: str) -> int | None:
    """Return Content-Length from a HEAD request, or None on failure."""
    import urllib.request
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=5) as resp:
            cl = resp.headers.get('Content-Length')
            return int(cl) if cl else None
    except Exception:
        return None


def _populate_file_sizes(models: list[dict]) -> None:
    """Fetch checkpoint file sizes for all models using concurrent HEAD requests."""
    from concurrent.futures import ThreadPoolExecutor

    url_map: dict[int, str] = {}
    for i, m in enumerate(models):
        url = m.get('_weight_url', '')
        if url and url.startswith('http'):
            url_map[i] = url

    if not url_map:
        return

    results: dict[int, int | None] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_head_content_length, url): idx for idx, url in url_map.items()}
        for future in futures:
            idx = futures[future]
            results[idx] = future.result()

    for idx, size_bytes in results.items():
        if size_bytes and size_bytes > 0:
            size_mb = round(size_bytes / (1024 * 1024), 1)
            models[idx]['fileSize'] = size_mb


def _discover_mmdet_models() -> list[dict]:
    """
    Discover COCO-pretrained detection models from MMDetection's metafile
    registry via openmim. Results are cached after first call.
    """
    if hasattr(_discover_mmdet_models, '_cache'):
        return _discover_mmdet_models._cache

    models: list[dict] = []

    try:
        import os
        from mim.commands.search import get_model_info
        # Field names are lowercase in openmim 0.3.x:
        #   model, weight, config, training_data, architecture, paper, readme
        df = get_model_info(
            'mmdet',
            shown_fields=[
                'model',
                'weight',
                'config',
                'training_data',
                'architecture',
                'paper',
                'readme',
            ],
        )
        seen: set[str] = set()
        for _, row in df.iterrows():
            import pandas as pd
            
            config_path = row.get('config', '')
            weight = row.get('weight', '')
            
            # Handle NaN values from pandas properly
            training_data_raw = row.get('training_data', '')
            training_data = str(training_data_raw).lower() if pd.notna(training_data_raw) else ''
            
            model_label_raw = row.get('model', '')
            model_label = str(model_label_raw) if pd.notna(model_label_raw) else ''
            
            architecture_raw = row.get('architecture', '')
            architecture = str(architecture_raw) if pd.notna(architecture_raw) else ''
            
            readme_raw = row.get('readme', '')
            readme = str(readme_raw) if pd.notna(readme_raw) else ''
            
            paper_raw = row.get('paper', '')
            # Extract real paper URL (skip NaN and placeholder 'URL,Title')
            paper = ''
            if pd.notna(paper_raw):
                paper_str = str(paper_raw).strip()
                if paper_str and paper_str != 'URL,Title' and ('http://' in paper_str or 'https://' in paper_str):
                    # If there's a comma, take the first part (URL)
                    paper = paper_str.split(',')[0].strip()
            
            # Fallback: generate GitHub link to model README or config
            if not paper and readme:
                paper = f"https://github.com/open-mmlab/mmdetection/tree/main/{readme}"
            elif not paper and config_path:
                paper = f"https://github.com/open-mmlab/mmdetection/tree/main/{config_path}"
            
            # Use model label and architecture as summary/description
            if model_label and architecture and training_data:
                # Replace commas with comma-space for readability
                arch_display = architecture.replace(',', ', ')
                summary = f"{model_label} is a {arch_display} model trained on {training_data.upper()}."
            else:
                summary = model_label

            # Only include models that have a checkpoint
            if not config_path or not weight:
                continue

            # The config basename (without .py) is the model ID that
            # DetInferencer accepts, e.g. "rtmdet_tiny_8xb32-300e_coco"
            name = os.path.basename(config_path).replace('.py', '')
            if not name or name in seen:
                continue
            seen.add(name)

            # Build human-readable label from config name
            parts = name.replace('_', ' ').replace('-', ' ').split()
            label = ' '.join(parts).title()
            # Architecture family = first segment before '_'
            arch = name.split('_')[0] if '_' in name else name.split('-')[0]

            # Detect open-vocabulary models by architecture/config name
            ov_keywords = ('grounding-dino', 'grounding_dino', 'glip', 'detic', 'yolo-world', 'yolo_world')
            is_open_vocab = any(kw in name.lower() for kw in ov_keywords) or any(kw in architecture.lower() for kw in ov_keywords)

            if is_open_vocab:
                label = f'{label} (Open Vocab)'

            models.append({
                'id': name,
                'label': label,
                'arch': arch,
                'dataset': training_data,
                'architecture': architecture,
                'task': 'object_detection',
                'paper': paper,
                'summary': summary,
                'openVocab': is_open_vocab,
                '_weight_url': weight if pd.notna(weight) else '',
            })

        # Fetch checkpoint file sizes via concurrent HEAD requests
        _populate_file_sizes(models)

        # Sort by architecture, then by name
        models.sort(key=lambda m: (m['arch'], m['id']))
        print(f'[api] Discovered {len(models)} MMDetection COCO models from metafiles')

    except Exception as e:
        print(f'[api] mim-based model discovery failed: {e}, falling back to curated list')
        import traceback
        traceback.print_exc()
        # Fallback: return the curated list from model_utils
        try:
            from model_utils import MMDET_MODEL_ZOO
            models = [
                {
                    'id': k,
                    'label': k.replace('_', ' ').replace('-', ' ').title(),
                    'arch': k.split('_')[0],
                    'dataset': 'coco',
                    'architecture': 'object detection',
                    'task': 'object_detection',
                    'paper': '',
                    'summary': f"{k.replace('_', ' ').title()} trained on COCO dataset",
                }
                for k in MMDET_MODEL_ZOO
            ]
        except ImportError:
            models = [
                {
                    'id': 'rtmdet_tiny_8xb32-300e_coco',
                    'label': 'RTMDet Tiny',
                    'arch': 'rtmdet',
                    'dataset': 'coco',
                    'architecture': 'real-time detection',
                    'task': 'object_detection',
                    'paper': '',
                    'summary': 'RTMDet Tiny is a real-time detection model trained on COCO.',
                },
                {
                    'id': 'rtmdet_s_8xb32-300e_coco',
                    'label': 'RTMDet Small',
                    'arch': 'rtmdet',
                    'dataset': 'coco',
                    'architecture': 'real-time detection',
                    'task': 'object_detection',
                    'paper': '',
                    'summary': 'RTMDet Small is a real-time detection model trained on COCO.',
                },
                {
                    'id': 'rtmdet_m_8xb32-300e_coco',
                    'label': 'RTMDet Medium',
                    'arch': 'rtmdet',
                    'dataset': 'coco',
                    'architecture': 'real-time detection',
                    'task': 'object_detection',
                    'paper': '',
                    'summary': 'RTMDet Medium is a real-time detection model trained on COCO.',
                },
            ]

    # Remove internal _weight_url key before caching
    for m in models:
        m.pop('_weight_url', None)

    _discover_mmdet_models._cache = models
    return models


# ── Model preparation ───────────────────────────────────────────────────────

class PrepareRequest(BaseModel):
    model: str


@app.get("/models/{model_id}/status")
def model_status(model_id: str):
    """Check whether a model's checkpoint is already cached locally."""
    from model_utils import is_model_cached
    cached = is_model_cached(model_id)
    return {"model": model_id, "cached": cached}


@app.post("/models/prepare")
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


@app.get("/health")
def health():
    """Health check."""
    alive = {k: v.pid for k, v in processes.items() if v.poll() is None}
    return {"ok": True, "streams": alive}
