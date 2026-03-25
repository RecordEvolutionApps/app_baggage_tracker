import { ironflock, getIronFlockConfig } from './ironflock.js'

// ── Config constants ───────────────────────────────────────────────────────

export const VIDEO_API = Bun.env.VIDEO_API || "http://video:8000";
export const MEDIASOUP_WS = Bun.env.MEDIASOUP_WS || "ws://mediasoup:1200";

// ── Types ──────────────────────────────────────────────────────────────────

export type MaskPolygon = {
    id: number
    label: string
    type: 'ZONE' | 'LINE'
    lineColor: string
    fillColor: string
    committed: boolean
    points: { x: number; y: number }[]
}

export type MaskData = {
    selectedPolygonId?: number
    polygons: MaskPolygon[]
}

export type SourceConfig = {
    id: string
    type: 'USB' | 'IP' | 'Demo' | 'YouTube' | 'Image'
    path?: string
    username?: string
    password?: string
    width?: number
    height?: number
}

export type InferenceConfig = {
    model?: string
    useSahi?: boolean
    useSmoothing?: boolean
    confidence?: number
    nmsIou?: number
    sahiIou?: number
    frameBuffer?: number
    overlapRatio?: number
}

export type ProcessingConfig = {
    masks?: MaskData
    classList?: number[]
    classNames?: string[]
}

export type StreamConfig = {
    camStream: string
    name: string
    stopped?: boolean
    source: SourceConfig
    inference?: InferenceConfig
    processing?: ProcessingConfig
}

/** @deprecated Use StreamConfig instead */
export type Camera = StreamConfig

export type USBCameraInfo = DeviceCameraInfo   // backward compat alias
export type DeviceCameraInfo = {
    path: string
    name: string
    id: string
    resolutions: { width: number; height: number }[]
    interface: 'usb' | 'csi' | 'gmsl' | 'other'
}

// ── Normalize legacy flat configs into nested structure ─────────────────────

/** Auto-migrate a flat (legacy) or nested config into the canonical nested shape. */
export function normalizeStreamConfig(raw: any): StreamConfig {
    // Already normalized — has a `source` object
    if (raw.source && typeof raw.source === 'object') {
        return raw as StreamConfig
    }

    // Flat → nested migration
    const source: SourceConfig = {
        id: raw.id ?? '',
        type: raw.type ?? 'IP',
        path: raw.path,
        username: raw.username,
        password: raw.password,
        width: raw.width,
        height: raw.height,
    }

    const inference: InferenceConfig = {
        model: raw.model,
        useSahi: raw.useSahi,
        useSmoothing: raw.useSmoothing,
        confidence: raw.confidence,
        nmsIou: raw.nmsIou,
        sahiIou: raw.sahiIou,
        frameBuffer: raw.frameBuffer,
        overlapRatio: raw.overlapRatio,
    }

    const processing: ProcessingConfig = {
        masks: raw.masks,
        classList: raw.classList,
        classNames: raw.classNames,
    }

    return {
        camStream: raw.camStream ?? '',
        name: raw.name ?? '',
        stopped: raw.stopped,
        source,
        inference,
        processing,
    }
}

// ── Shared mutable state ───────────────────────────────────────────────────

export const ports = new Map<string, any>()

// ── Stream config persistence (IronFlock backend table) ────────────────────

function rowToStreamConfig(row: any): StreamConfig {
    const config = typeof row.stream_config === 'string'
        ? JSON.parse(row.stream_config)
        : row.stream_config ?? {}
    config.camStream = row.stream_name ?? config.camStream
    return normalizeStreamConfig(config)
}

/** Read a single stream config from the backend table. Returns null if not found. */
export async function readStreamConfig(camStream: string): Promise<StreamConfig | null> {
    try {
        const rows = await ironflock.getHistory('streams', {
            limit: 1,
            filterAnd: [
                { column: 'stream_name', operator: '=', value: camStream },
                { column: 'latest_flag', operator: '=', value: true },
                { column: 'deleted', operator: '!=', value: true },
            ],
        }) as any[] | null
        if (rows && rows.length > 0) return rowToStreamConfig(rows[0])
    } catch (err) {
        console.error(`Failed to read stream config for ${camStream}:`, err)
    }
    return null
}

/** Write a full stream config to the backend table.
 *
 * Always performs a read-modify-write: reads the current row first and
 * deep-merges the incoming config on top of it, so no caller can accidentally
 * drop sections it didn't touch (source, inference, or processing/masks).
 */
export async function writeStreamConfig(camStream: string, config: StreamConfig, status = 'configured'): Promise<void> {
    const existing = await readStreamConfig(camStream)

    const merged: StreamConfig = existing ? {
        ...existing,
        ...config,
        source: { ...(existing.source ?? {}), ...(config.source ?? {}) },
        inference: { ...(existing.inference ?? {}), ...(config.inference ?? {}) },
        processing: {
            ...(existing.processing ?? {}),
            ...(config.processing ?? {}),
        },
    } : config

    const now = new Date().toISOString()
    const { deviceKey } = getIronFlockConfig()
    await ironflock.appendToTable('streams', [{
        tsp: now,
        stream_name: camStream,
        stream_url: `https://${deviceKey}-visionai-1100.app.ironflock.com/#view/${encodeURIComponent(camStream)}`,
        cam_path: merged.source?.path ?? '',
        stream_config: JSON.stringify(merged),
        status,
        deleted: false,
    }], { exclude_me: true })
}

/** Mark a stream as deleted in the backend table. */
export async function deleteStreamConfig(camStream: string): Promise<void> {
    const now = new Date().toISOString()
    const { deviceKey } = getIronFlockConfig()
    await ironflock.appendToTable('streams', [{
        tsp: now,
        stream_name: camStream,
        stream_url: `https://${deviceKey}-visionai-1100.app.ironflock.com/#view/${encodeURIComponent(camStream)}`,
        cam_path: '',
        stream_config: '{}',
        status: 'deleted',
        deleted: true,
    }], { exclude_me: true })
}

/** List all active stream configs from the backend table. */
export async function listStreamConfigs(): Promise<StreamConfig[]> {
    try {
        const rows = await ironflock.getHistory('streams', {
            limit: 10000,
            filterAnd: [
                { column: 'latest_flag', operator: '=', value: true },
                { column: 'deleted', operator: '!=', value: true },
            ],
        }) as any[] | null
        return (rows ?? []).map(rowToStreamConfig)
    } catch (err) {
        console.error('Failed to list stream configs:', err)
        return []
    }
}

// ── Source field comparison ─────────────────────────────────────────────────

const SOURCE_FIELDS: (keyof SourceConfig)[] = ['type', 'path', 'username', 'password', 'width', 'height']

/** Returns true if any camera-source field changed between old and new config. */
export function sourceChanged(prev: StreamConfig | null, next: StreamConfig): boolean {
    if (!prev) return true
    return SOURCE_FIELDS.some(k => prev.source?.[k] !== next.source?.[k])
}

// ── Migration from legacy files ────────────────────────────────────────────

export async function migrateFromLegacy(): Promise<void> {
    const { readdir, rename } = await import('node:fs/promises')
    const { join } = await import('node:path')

    // Check if legacy streamSetup.json exists
    const legacySetupFile = Bun.file('/data/streamSetup.json')
    if (!(await legacySetupFile.exists())) return

    let legacy: Record<string, any>
    try {
        legacy = await legacySetupFile.json()
    } catch (err) {
        console.error('Failed to read legacy streamSetup.json:', err)
        return
    }

    console.log('Migrating legacy stream configs to backend table...')

    for (const [camStream, cam] of Object.entries(legacy)) {
        // Read per-stream settings file if it exists
        let settings: Record<string, any> = {}
        try {
            const settingsFile = Bun.file(`/data/settings/${camStream}.json`)
            if (await settingsFile.exists()) settings = await settingsFile.json()
        } catch { /* no settings file */ }

        // Read per-stream mask file if it exists
        let masks: MaskData = { polygons: [] }
        try {
            const maskFile = Bun.file(`/data/masks/${camStream}.json`)
            if (await maskFile.exists()) masks = await maskFile.json()
        } catch { /* no mask file */ }

        // Also try legacy single mask file for frontCam
        if (camStream === 'frontCam' && masks.polygons.length === 0) {
            try {
                const legacyMask = Bun.file('/data/mask.json')
                if (await legacyMask.exists()) {
                    const data = await legacyMask.json()
                    if (data?.polygons) masks = data
                }
            } catch { /* ignore */ }
        }

        const config: StreamConfig = normalizeStreamConfig({
            ...cam,
            camStream,
            model: settings.model ?? cam.model ?? undefined,
            useSahi: settings.useSahi ?? cam.useSahi,
            useSmoothing: settings.useSmoothing ?? cam.useSmoothing,
            confidence: settings.confidence ?? cam.confidence,
            frameBuffer: settings.frameBuffer ?? cam.frameBuffer,
            nmsIou: settings.nmsIou ?? cam.nmsIou,
            sahiIou: settings.sahiIou ?? cam.sahiIou,
            overlapRatio: settings.overlapRatio ?? cam.overlapRatio,
            classList: settings.classList ?? cam.classList,
            classNames: settings.classNames ?? cam.classNames,
            masks,
        })

        await writeStreamConfig(camStream, config)
        console.log(`  Migrated stream: ${camStream}`)
    }

    // Rename legacy file to .bak
    try { await rename('/data/streamSetup.json', '/data/streamSetup.json.bak') } catch { }
    console.log('Legacy migration complete')
}

// ── Also migrate from /data/streams/*.json files to backend table ──────────

export async function migrateFromFiles(): Promise<void> {
    const { readdir } = await import('node:fs/promises')
    const { join } = await import('node:path')
    const streamsDir = '/data/streams'

    let files: string[]
    try {
        files = await readdir(streamsDir)
    } catch {
        return // no streams dir
    }

    const jsonFiles = files.filter(f => f.endsWith('.json'))
    if (jsonFiles.length === 0) return

    // Check if the backend table already has configs — if so, skip file migration
    const existing = await listStreamConfigs()
    if (existing.length > 0) return

    console.log('Migrating stream configs from files to backend table...')
    for (const file of jsonFiles) {
        try {
            const data = await Bun.file(join(streamsDir, file)).json()
            const camStream = data.camStream ?? file.replace('.json', '')
            await writeStreamConfig(camStream, normalizeStreamConfig(data))
            console.log(`  Migrated file: ${file}`)
        } catch (err) {
            console.error(`Failed to migrate ${file}:`, err)
        }
    }
    console.log('File migration complete')
}

// ── Utilities ──────────────────────────────────────────────────────────────

/**
 * Wait for a service to respond to HTTP requests.
 * Retries every `intervalMs` up to `maxRetries` times.
 */
export async function waitForService(url: string, name: string, maxRetries = 30, intervalMs = 2000): Promise<boolean> {
    for (let i = 1; i <= maxRetries; i++) {
        try {
            const res = await fetch(url, { signal: AbortSignal.timeout(2000) })
            if (res.ok || res.status < 500) {
                console.log(`✓ ${name} is ready (attempt ${i})`)
                return true
            }
        } catch (_) { /* not ready yet */ }
        console.log(`Waiting for ${name}... (attempt ${i}/${maxRetries})`)
        await new Promise(r => setTimeout(r, intervalMs))
    }
    console.error(`✗ ${name} not reachable after ${maxRetries} attempts`)
    return false
}
