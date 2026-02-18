"""Async file watchers for mask and settings JSON files."""
from __future__ import annotations

import json
import logging
import os
from asyncio import sleep

from config import StreamConfig
from masks import prepMasks

logger = logging.getLogger('watchers')


async def watchMaskFile(config: StreamConfig, poll_interval: float = 1.0):
    """Poll the per-stream mask file for changes and reload when modified.

    Mutates ``config.saved_masks`` in-place.
    """
    mask_path = f'/data/masks/{config.cam_stream}.json'
    legacy_path = '/data/mask.json'
    last_mtime = 0.0

    logger.info('Watching mask file: %s', mask_path)

    while True:
        try:
            path = mask_path if os.path.exists(mask_path) else legacy_path

            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(path, 'r') as f:
                        loaded_masks = json.load(f)
                    config.saved_masks[:] = []
                    config.saved_masks.extend(
                        prepMasks(loaded_masks, config.resolution_x, config.resolution_y)
                    )
                    logger.info('Reloaded masks from %s', path)
        except Exception as e:
            logger.error('Error reading mask file: %s', e, exc_info=True)

        await sleep(poll_interval)


async def watchSettingsFile(config: StreamConfig, poll_interval: float = 1.0):
    """Poll the per-stream settings file for changes and update ``config.stream_settings``.

    The backend writes ``/data/settings/<camStream>.json`` when settings change.
    """
    settings_path = f'/data/settings/{config.cam_stream}.json'
    last_mtime = 0.0

    logger.info('[settings] Watching %s', settings_path)

    while True:
        try:
            if os.path.exists(settings_path):
                mtime = os.path.getmtime(settings_path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(settings_path, 'r') as f:
                        new_settings = json.load(f)
                    # Log which keys changed
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
                    else:
                        logger.debug('[settings] %s: settings file touched (no changes)',
                                     config.cam_stream)
        except Exception as e:
            logger.error('[settings] Error reading settings file: %s', e, exc_info=True)

        await sleep(poll_interval)
