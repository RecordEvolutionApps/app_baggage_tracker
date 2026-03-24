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

# Source fields that require re-opening the video capture
_SOURCE_FIELDS = ('type', 'path', 'username', 'password', 'width', 'height')


def _resolve_device(source: dict) -> str:
    """Convert a source config dict into a device string (same logic as the TS backend)."""
    src_type = source.get('type', '')
    path = source.get('path', '') or ''
    if src_type == 'IP' and path and '://' in path:
        protocol, rest = path.split('://', 1)
        username = source.get('username', '')
        password = source.get('password', '')
        userpw = (username + (':' + password if password else '') + '@') if username else ''
        return f'{protocol}://{userpw}{rest}'
    return path


async def watchStreamConfig(ironflock, config: StreamConfig, on_change=None):
    """Subscribe to the ``streams`` table for live config updates.

    Replaces the old file-polling watcher. The callback filters for rows
    matching the current ``cam_stream`` and applies settings/mask changes.
    """
    last_config_repr = json.dumps(config.stream_settings, sort_keys=True)
    last_masks_repr = ''

    def _on_row(row):
        nonlocal last_config_repr, last_masks_repr

        try:
            _on_row_inner(row)
        except Exception:
            logger.error('[watcher] Unhandled error in _on_row callback', exc_info=True)

    def _on_row_inner(row):
        nonlocal last_config_repr, last_masks_repr

        row_stream = row.get('stream_name', '')
        row_status = row.get('status', '')
        logger.info('[watcher] Row received: stream=%s status=%s (our stream=%s)',
                    row_stream, row_status, config.cam_stream)

        # Filter: only process rows for our stream
        if row_stream != config.cam_stream:
            logger.debug('[watcher] Ignoring row for different stream: %s', row_stream)
            return

        # Ignore deleted rows
        if row.get('deleted'):
            logger.info('[watcher] Ignoring deleted row for %s', row_stream)
            return

        try:
            raw = row.get('stream_config', '{}')
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            logger.warning('[watcher] Failed to parse stream_config: %s', e)
            return

        # ── Source change detection ─────────────────────────────
        source = data.get('source', {})
        old_source = config._full_config.get('source', {})
        if any(source.get(k) != old_source.get(k) for k in _SOURCE_FIELDS):
            new_device = _resolve_device(source)
            if new_device != config.device:
                logger.info('[watcher] Source changed: %s -> %s', config.device, new_device)
                config._pending_source = {
                    'device': new_device,
                    'width': source.get('width'),
                    'height': source.get('height'),
                }

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

        # Log extracted model value for diagnostics
        if 'model' in new_settings:
            logger.info('[watcher] Extracted model=%s (from inference=%s, flat=%s), '
                        'current stream_settings.model=%s, current_model_name=%s',
                        new_settings['model'],
                        'model' in inference,
                        'model' in data and 'model' not in inference,
                        config.stream_settings.get('model'),
                        config.current_model_name)
        else:
            logger.info('[watcher] No model key found in incoming config '
                        '(inference keys: %s, top-level keys: %s)',
                        list(inference.keys()), [k for k in data.keys()
                         if k in _INFERENCE_KEYS])

        # Debug: log what classList the watcher extracted
        if 'classList' in new_settings:
            logger.debug('[watcher] Extracted classList=%s from %s (nested=%s, flat=%s)',
                         new_settings['classList'],
                         config.cam_stream,
                         'classList' in processing,
                         'classList' in data and 'classList' not in processing)

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
