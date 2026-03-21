"""IronFlock publishing — images, camera hubs, streams, detection counts, and line counts."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
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
        stream_name = self._config.cam_stream

        async def _publish():
            loop = get_event_loop()
            base64_encoded_frame = await loop.run_in_executor(
                _publish_pool, _encode_image, frame.copy(),
            )
            now = datetime.now().astimezone().isoformat()
            await ironflock.publish_to_table(
                'images',
                {"tsp": now, "image": 'data:image/webp;base64,' + base64_encoded_frame, "stream_name": stream_name},
            )

        get_event_loop().create_task(_publish())

    def publish_camera_hubs(self):
        """Publish camera hub heartbeat to the ``camera_hubs`` table."""
        now = datetime.now().astimezone().isoformat()
        payload = {"tsp": now, "deleted": False}
        payload["webpage"] = f"https://{self._config.device_key}-visionai-1100.app.ironflock.com"
        payload["devicelink"] = self._config.device_url or ''
        get_event_loop().create_task(
            self._ironflock.publish_to_table('camera_hubs', payload),
        )

    def publish_class_count(self, zone_name, result):
        """Publish per-zone class counts to the ``zone_counts`` table."""
        now = datetime.now().astimezone().isoformat()
        detections = {str(class_id): result.get(class_id, 0)
                      for class_id in self._config.class_list}
        payload = {
            "tsp": now,
            "zone_name": zone_name,
            "detections": detections,
            "stream_name": self._config.cam_stream,
        }

        get_event_loop().create_task(
            self._ironflock.publish_to_table('zone_counts', payload),
        )

    def publish_line_count(self, line_name, num_in, num_out):
        """Publish per-line crossing counts to the ``linecounts`` table."""
        now = datetime.now().astimezone().isoformat()
        payload = {
            "tsp": now,
            "linename": line_name,
            "num_in": num_in,
            "num_out": num_out,
            "stream_name": self._config.cam_stream,
        }
        get_event_loop().create_task(
            self._ironflock.publish_to_table('linecounts', payload),
        )

    def publish_stream(self, *, status: str, deleted: bool = False):
        """Publish stream info to the ``streams`` table."""
        now = datetime.now().astimezone().isoformat()
        full = {**self._config._full_config, **self._config.stream_settings}
        stream_config = json.dumps(full)
        payload = {
            "tsp": now,
            "stream_name": self._config.cam_stream,
            "stream_url": f"https://{self._config.device_key}-visionai-1100.app.ironflock.com",
            "cam_path": self._config.device,
            "stream_config": stream_config,
            "status": status,
            "deleted": deleted,
        }
        get_event_loop().create_task(
            self._ironflock.publish_to_table('streams', payload, exclude_me=True),
        )


class StubIronFlock:
    """File-backed IronFlock stub for ``ENV=DEV``.

    Persists rows to ``/data/stub/<table>.json`` so the web backend
    (running in a separate container) can share state.
    """

    _STUB_DIR = '/data/stub'

    def __init__(self):
        self._subs: dict[str, list] = {}

    def _file_path(self, table: str) -> str:
        return os.path.join(self._STUB_DIR, f'{table}.json')

    def _read_table(self, table: str) -> list[dict]:
        try:
            with open(self._file_path(table), 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_table(self, table: str, rows: list[dict]) -> None:
        os.makedirs(self._STUB_DIR, exist_ok=True)
        tmp = self._file_path(table) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(rows, f, indent=2)
        os.replace(tmp, self._file_path(table))

    async def publish_to_table(self, table, data, **kwargs):
        rows = self._read_table(table)
        max_id = max((r.get('_rowId', 0) for r in rows), default=0)
        row = {**data, '_rowId': max_id + 1, '_publisher': 'py', 'latest_flag': True}

        # Maintain latest_flag for streams table
        if table == 'streams' and data.get('stream_name'):
            for r in rows:
                if r.get('stream_name') == data['stream_name']:
                    r['latest_flag'] = False

        rows.append(row)
        self._write_table(table, rows)

        # Fire local subscription callbacks (skip own if exclude_me)
        exclude_me = kwargs.get('exclude_me', False)
        if not exclude_me:
            for cb in self._subs.get(table, []):
                clean = {k: v for k, v in row.items() if k not in ('_rowId', '_publisher')}
                cb(clean)

    async def getHistory(self, table, params=None):
        rows = self._read_table(table)
        params = params or {}

        for f in params.get('filterAnd', []):
            col, op, val = f['column'], f['operator'], f['value']
            if op == '=':
                rows = [r for r in rows if r.get(col) == val]
            elif op == '!=':
                rows = [r for r in rows if r.get(col) != val]

        rows.sort(key=lambda r: r.get('_rowId', 0), reverse=True)
        rows = [{k: v for k, v in r.items() if k not in ('_rowId', '_publisher')} for r in rows]

        limit = params.get('limit')
        if limit:
            rows = rows[:limit]
        return rows

    async def subscribe_to_table(self, table, callback):
        if table not in self._subs:
            self._subs[table] = []
        self._subs[table].append(callback)

        # Poll the file for cross-process changes (rows published by the TS web backend)
        rows = self._read_table(table)
        last_id = max((r.get('_rowId', 0) for r in rows), default=0)

        async def _poll():
            nonlocal last_id
            while True:
                await asyncio.sleep(1.5)
                try:
                    current = self._read_table(table)
                    new_rows = [r for r in current
                                if r.get('_rowId', 0) > last_id and r.get('_publisher') != 'py']
                    if new_rows:
                        last_id = max(r.get('_rowId', 0) for r in current)
                        for r in new_rows:
                            clean = {k: v for k, v in r.items() if k not in ('_rowId', '_publisher')}
                            callback(clean)
                except Exception:
                    pass

        asyncio.get_event_loop().create_task(_poll())
