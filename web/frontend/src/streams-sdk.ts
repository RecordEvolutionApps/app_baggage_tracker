/**
 * SDK wrapper functions for stream config CRUD and subscriptions.
 *
 * These mirror the backend's shared.ts helpers but run in the browser,
 * talking directly to the IronFlock backend (or the DEV REST stub).
 */
import { ironflock, ironflockReady, deviceKey } from './ironflock.js'
import { type StreamConfig, normalizeStreamConfig } from './utils.js'

// ── Camera Hub ─────────────────────────────────────────────────────────────

export type CameraHub = {
    tsp?: string
    devname?: string
    webpage?: string
    devicelink?: string
    deleted?: boolean
}

/** Read the latest (non-deleted) camera hub row. */
export async function getCameraHub(): Promise<CameraHub | null> {
    await ironflockReady
    const rows = await ironflock.getHistory('camera_hubs', {
        limit: 1,
        filterAnd: [
            { column: 'latest_flag', operator: '=', value: true },
            { column: 'deleted', operator: '!=', value: true },
        ],
    })
    return (rows && rows.length > 0) ? rows[0] as CameraHub : null
}

/** Subscribe to camera hub updates (heartbeats + config changes). */
export async function subscribeCameraHub(callback: (hub: CameraHub) => void): Promise<void> {
    await ironflockReady
    ironflock.subscribeToTable('camera_hubs', (row: any) => {
        callback(row as CameraHub)
    })
}

// ── Row ↔ StreamConfig conversion ──────────────────────────────────────────

function rowToStreamConfig(row: any): StreamConfig {
    const config = typeof row.stream_config === 'string'
        ? JSON.parse(row.stream_config)
        : row.stream_config ?? {}
    config.camStream = row.stream_name ?? config.camStream
    return normalizeStreamConfig(config)
}

// ── CRUD operations ────────────────────────────────────────────────────────

/** List all active (non-deleted) stream configs. */
export async function listStreams(): Promise<StreamConfig[]> {
    await ironflockReady
    const rows = await ironflock.getHistory('streams', {
        limit: 10000,
        filterAnd: [
            { column: 'latest_flag', operator: '=', value: true },
            { column: 'deleted', operator: '!=', value: true },
        ],
    })
    return (rows ?? []).map(rowToStreamConfig)
}

/** Read a single stream config. Returns null if not found. */
export async function readStream(camStream: string): Promise<StreamConfig | null> {
    await ironflockReady
    const rows = await ironflock.getHistory('streams', {
        limit: 1,
        filterAnd: [
            { column: 'stream_name', operator: '=', value: camStream },
            { column: 'latest_flag', operator: '=', value: true },
            { column: 'deleted', operator: '!=', value: true },
        ],
    })
    if (rows && rows.length > 0) return rowToStreamConfig(rows[0])
    return null
}

/** Write (create or update) a stream config to the backend table.
 *
 * Always performs a read-modify-write: reads the current persisted row first
 * and deep-merges the incoming config on top of it before writing.  This
 * guarantees that every row in the table is always a complete StreamConfig,
 * regardless of which component (camera-dialog, inference-setup, polygon
 * manager, …) triggered the write and how much of the config it knew about.
 */
export async function writeStream(camStream: string, config: StreamConfig, status = 'configured'): Promise<void> {
    await ironflockReady

    // Read the current persisted state so we never lose sections the caller
    // didn't touch (e.g. changing the source must not wipe inference settings,
    // and saving inference settings must not wipe masks).
    const existing = await readStream(camStream)

    const merged: StreamConfig = existing ? {
        // Scalar top-level fields: incoming wins (name, stopped, camStream)
        ...existing,
        ...config,
        // Deep-merge each structured section independently.
        source: { ...(existing.source ?? {}), ...(config.source ?? {}) },
        inference: { ...(existing.inference ?? {}), ...(config.inference ?? {}) },
        processing: {
            ...(existing.processing ?? {}),
            ...(config.processing ?? {}),
        },
    } : config

    const now = new Date().toISOString()
    await ironflock.appendToTable('streams', [{
        tsp: now,
        stream_name: camStream,
        stream_url: `https://${deviceKey}-visionai-1100.app.ironflock.com/#view/${encodeURIComponent(camStream)}`,
        cam_path: merged.source?.path ?? '',
        stream_config: JSON.stringify(merged),
        status,
        deleted: false,
    }])
}

/** Mark a stream config as deleted in the backend table. */
export async function deleteStream(camStream: string): Promise<void> {
    await ironflockReady
    const now = new Date().toISOString()
    await ironflock.appendToTable('streams', [{
        tsp: now,
        stream_name: camStream,
        stream_url: `https://${deviceKey}-visionai-1100.app.ironflock.com/#view/${encodeURIComponent(camStream)}`,
        cam_path: '',
        stream_config: '{}',
        status: 'deleted',
        deleted: true,
    }])
}

// ── Subscriptions ──────────────────────────────────────────────────────────

/**
 * Subscribe to stream config changes. The callback receives a StreamConfig
 * for each changed stream, or a config with `{ deleted: true }` for removals.
 *
 * Returns an unsubscribe function (currently a no-op since the SDK doesn't
 * expose unsubscribe — the subscription lives for the page lifetime).
 */
export async function subscribeStreams(callback: (config: StreamConfig & { deleted?: boolean }) => void): Promise<void> {
    await ironflockReady
    ironflock.subscribeToTable('streams', (row: any) => {
        if (row.deleted) {
            callback({ ...rowToStreamConfig(row), deleted: true } as any)
        } else {
            callback(rowToStreamConfig(row))
        }
    })
}
