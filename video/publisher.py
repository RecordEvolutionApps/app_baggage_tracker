"""IronFlock publishing — images, camera hubs, streams, detection counts, and line counts."""
from __future__ import annotations

import base64
import json
import logging
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2

from config import StreamConfig

logger = logging.getLogger('publisher')

_publish_pool = ThreadPoolExecutor(max_workers=1)


def _encode_image(frame):
    """WebP + base64 encoding — ~30% smaller than JPEG at equivalent quality."""
    _, encoded_frame = cv2.imencode('.webp', frame, [cv2.IMWRITE_WEBP_QUALITY, 80])
    return base64.b64encode(encoded_frame.tobytes()).decode('utf-8')


class Publisher:
    """Wraps an IronFlock (or stub) instance for async data publishing."""

    def __init__(self, ironflock, config: StreamConfig):
        self._ironflock = ironflock
        self._config = config

    def publish_image(self, frame):
        """Publish an annotated frame to the ``images`` table."""
        ironflock = self._ironflock

        async def _publish():
            loop = get_event_loop()
            base64_encoded_frame = await loop.run_in_executor(
                _publish_pool, _encode_image, frame.copy(),
            )
            now = datetime.now().astimezone().isoformat()
            await ironflock.publish_to_table(
                'images',
                {"tsp": now, "image": 'data:image/webp;base64,' + base64_encoded_frame},
            )

        get_event_loop().create_task(_publish())

    def publish_cameras(self):
        """Publish camera hub heartbeat to the ``camera_hubs`` table."""
        now = datetime.now().astimezone().isoformat()
        payload = {"tsp": now}
        payload["videolink"] = f"https://{self._config.device_key}-baggagetracker-1100.app.ironflock.com"
        payload["devicelink"] = self._config.device_url
        get_event_loop().create_task(
            self._ironflock.publish_to_table('camera_hubs', payload),
        )

    def publish_class_count(self, zone_name, result):
        """Publish per-zone class counts to the ``detections`` table."""
        now = datetime.now().astimezone().isoformat()
        detections = {str(class_id): result.get(class_id, 0)
                      for class_id in self._config.class_list}
        payload = {
            "tsp": now,
            "zone_name": zone_name,
            "detections": detections,
        }

        get_event_loop().create_task(
            self._ironflock.publish_to_table('detections', payload),
        )

    def publish_line_count(self, line_name, num_in, num_out):
        """Publish per-line crossing counts to the ``linecounts`` table."""
        now = datetime.now().astimezone().isoformat()
        payload = {
            "tsp": now,
            "line_name": line_name,
            "num_in": num_in,
            "num_out": num_out,
        }
        get_event_loop().create_task(
            self._ironflock.publish_to_table('linecounts', payload),
        )

    def publish_stream(self, *, status: str, deleted: bool = False):
        """Publish stream info to the ``streams`` table."""
        now = datetime.now().astimezone().isoformat()
        payload = {
            "tsp": now,
            "cam_stream": self._config.cam_stream,
            "cam_path": self._config.device,
            "stream_config": json.dumps(self._config.stream_settings),
            "status": status,
            "deleted": deleted,
        }
        get_event_loop().create_task(
            self._ironflock.publish_to_table('streams', payload),
        )


class StubIronFlock:
    """No-op IronFlock stub for ``ENV=DEV``."""

    async def publish_to_table(self, table, data):
        pass
