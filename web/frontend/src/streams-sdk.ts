/**
 * SDK wrapper functions for stream config CRUD and subscriptions.
 *
 * These mirror the backend's shared.ts helpers but run in the browser,
 * talking directly to the IronFlock backend (or the DEV REST stub).
 */
import { ironflock, ironflockReady, deviceKey } from './ironflock.js'
import type { StreamConfig } from './utils.js'

// ── Row ↔ StreamConfig conversion ──────────────────────────────────────────

function rowToStreamConfig(row: any): StreamConfig {
    const config = typeof row.stream_config === 'string'
        ? JSON.parse(row.stream_config)
        : row.stream_config ?? {}
    return {
        ...config,
        camStream: row.stream_name ?? config.camStream,
    } as StreamConfig
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

/** Write (create or update) a stream config to the backend table. */
export async function writeStream(camStream: string, config: StreamConfig, status = 'configured'): Promise<void> {
    await ironflockReady
    const now = new Date().toISOString()
    await ironflock.appendToTable('streams', [{
        tsp: now,
        stream_name: camStream,
        stream_url: `https://${deviceKey}-visionai-1100.app.ironflock.com/#view/${encodeURIComponent(camStream)}`,
        cam_path: config.path ?? '',
        stream_config: JSON.stringify(config),
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
