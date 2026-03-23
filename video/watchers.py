"""Stream config watcher via IronFlock backend table subscription."""
from __future__ import annotations

import json
import logging

from config import StreamConfig
from masks import prepMasks

logger = logging.getLogger('watchers')

# Settings keys that the video process cares about (inside 'inference' section)
_INFERENCE_KEYS = frozenset([
    'model', 'useSahi', 'useSmoothing', 'confidence', 'frameBuffer',
    'nmsIou', 'sahiIou', 'overlapRatio',
])

# Processing keys (inside 'processing' section)
_PROCESSING_KEYS = frozenset(['classList', 'classNames'])


async def watchStreamConfig(ironflock, config: StreamConfig, on_change=None):
    """Subscribe to the ``streams`` table for live config updates.

    Replaces the old file-polling watcher. The callback filters for rows
    matching the current ``cam_stream`` and applies settings/mask changes.
    """
    last_config_repr = json.dumps(config.stream_settings, sort_keys=True)
    last_masks_repr = ''

    def _on_row(row):
        nonlocal last_config_repr, last_masks_repr

        # Filter: only process rows for our stream
        row_stream = row.get('stream_name', '')
        if row_stream != config.cam_stream:
            return

        # Ignore deleted rows
        if row.get('deleted'):
            return

        try:
            raw = row.get('stream_config', '{}')
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            logger.warning('[watcher] Failed to parse stream_config: %s', e)
            return

        # ── Settings update ─────────────────────────────────────
        # Keep full config blob in sync for the publisher
        config._full_config.update(data)

        # Extract settings from nested sections (with flat fallback for compat)
        inference = data.get('inference', {})
        processing = data.get('processing', {})
        new_settings = {}
        for k in _INFERENCE_KEYS:
            if k in inference:
                new_settings[k] = inference[k]
            elif k in data:  # flat fallback
                new_settings[k] = data[k]
        for k in _PROCESSING_KEYS:
            if k in processing:
                new_settings[k] = processing[k]
            elif k in data:  # flat fallback
                new_settings[k] = data[k]
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

        # ── Masks update ────────────────────────────────────────
        masks_data = processing.get('masks', data.get('masks', {}))
        masks_repr = json.dumps(masks_data.get('polygons', []), sort_keys=True)
        masks_changed = masks_repr != last_masks_repr
        if masks_changed:
            last_masks_repr = masks_repr
            config.saved_masks[:] = []
            config.saved_masks.extend(
                prepMasks(masks_data, config.resolution_x, config.resolution_y)
            )
            logger.info('[masks] Reloaded masks from backend table subscription')

        if changed or masks_changed:
            if on_change is not None:
                on_change()
        else:
            logger.debug('[watcher] %s: row received but no changes', config.cam_stream)

    logger.info('[watcher] Subscribing to streams table for %s', config.cam_stream)
    await ironflock.subscribe_to_table('streams', _on_row)
