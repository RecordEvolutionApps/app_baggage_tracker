"""Async file watcher for the unified stream config JSON file."""
from __future__ import annotations

import json
import logging
import os
from asyncio import sleep

from config import StreamConfig
from masks import prepMasks

logger = logging.getLogger('watchers')

# Settings keys that the video process cares about
_SETTINGS_KEYS = frozenset([
    'model', 'useSahi', 'useSmoothing', 'confidence', 'frameBuffer',
    'nmsIou', 'sahiIou', 'overlapRatio', 'classList', 'classNames',
])


async def watchStreamConfig(config: StreamConfig, poll_interval: float = 1.0,
                            on_change=None):
    """Poll the unified stream config file for both settings and mask changes.

    The backend writes ``/data/streams/<camStream>.json`` on every update.
    This watcher replaces the old separate ``watchSettingsFile`` and
    ``watchMaskFile`` coroutines.
    """
    config_path = f'/data/streams/{config.cam_stream}.json'
    last_mtime = 0.0
    last_masks_repr = ''

    logger.info('[watcher] Watching %s', config_path)

    while True:
        try:
            if os.path.exists(config_path):
                mtime = os.path.getmtime(config_path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(config_path, 'r') as f:
                        data = json.load(f)

                    # ── Settings update ─────────────────────────────────
                    new_settings = {k: data[k] for k in _SETTINGS_KEYS if k in data}
                    changed = {}
                    for k, v in new_settings.items():
                        old_v = config.stream_settings.get(k)
                        if old_v != v:
                            changed[k] = {'from': old_v, 'to': v}
                    config.stream_settings.update(new_settings)
                    if changed:
                        for k, diff in changed.items():
                            logger.info('[settings] %s: %s: %s -> %s',
                                        config.cam_stream, k, diff['from'], diff['to'])

                    # ── Masks update ────────────────────────────────────
                    masks_data = data.get('masks', {})
                    masks_repr = json.dumps(masks_data.get('polygons', []), sort_keys=True)
                    masks_changed = masks_repr != last_masks_repr
                    if masks_changed:
                        last_masks_repr = masks_repr
                        config.saved_masks[:] = []
                        config.saved_masks.extend(
                            prepMasks(masks_data, config.resolution_x, config.resolution_y)
                        )
                        logger.info('[masks] Reloaded masks from %s', config_path)

                    if changed or masks_changed:
                        if on_change is not None:
                            on_change()
                    else:
                        logger.debug('[watcher] %s: file touched (no changes)',
                                     config.cam_stream)
        except Exception as e:
            logger.error('[watcher] Error reading stream config: %s', e, exc_info=True)

        await sleep(poll_interval)
