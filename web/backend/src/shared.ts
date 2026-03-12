import { mkdir, readdir } from 'node:fs/promises'
import { join } from 'node:path'

// ── Config constants ───────────────────────────────────────────────────────

export const VIDEO_API = Bun.env.VIDEO_API || "http://video:8000";
export const MEDIASOUP_WS = Bun.env.MEDIASOUP_WS || "ws://mediasoup:1200";
export const streamsDir = '/data/streams'

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

export type StreamConfig = {
    id: string
    type: 'USB' | 'IP' | 'Demo' | 'YouTube' | 'Image'
    name: string
    path?: string
    username?: string
    password?: string
    camStream: string
    width?: number
    height?: number
    model?: string
    useSahi?: boolean
    useSmoothing?: boolean
    confidence?: number
    frameBuffer?: number
    nmsIou?: number
    sahiIou?: number
    overlapRatio?: number
    classList?: number[]
    classNames?: string[]
    stopped?: boolean
    masks?: MaskData
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

// ── Shared mutable state ───────────────────────────────────────────────────

export const ports = new Map<string, any>()

// ── Stream config persistence (one file per stream) ────────────────────────

async function ensureStreamsDir() {
    await mkdir(streamsDir, { recursive: true })
}

function streamPath(camStream: string): string {
    return join(streamsDir, `${camStream}.json`)
}

/** Read a single stream config from disk. Returns null if not found. */
export async function readStreamConfig(camStream: string): Promise<StreamConfig | null> {
    try {
        const file = Bun.file(streamPath(camStream))
        if (await file.exists()) return await file.json()
    } catch (err) {
        console.error(`Failed to read stream config for ${camStream}:`, err)
    }
    return null
}

/** Write a full stream config to disk. */
export async function writeStreamConfig(camStream: string, config: StreamConfig): Promise<void> {
    await ensureStreamsDir()
    await Bun.write(streamPath(camStream), JSON.stringify(config, null, 2))
}

/** Delete a stream config file from disk. */
export async function deleteStreamConfig(camStream: string): Promise<void> {
    const { unlink } = await import('node:fs/promises')
    try {
        await unlink(streamPath(camStream))
    } catch (err: any) {
        if (err.code !== 'ENOENT') throw err
    }
}

/** List all stream configs by reading all JSON files in the streams dir. */
export async function listStreamConfigs(): Promise<StreamConfig[]> {
    await ensureStreamsDir()
    const files = await readdir(streamsDir)
    const configs: StreamConfig[] = []
    for (const file of files) {
        if (!file.endsWith('.json')) continue
        try {
            const data = await Bun.file(join(streamsDir, file)).json()
            configs.push(data)
        } catch (err) {
            console.error(`Failed to read stream config ${file}:`, err)
        }
    }
    return configs
}

// ── Source field comparison ─────────────────────────────────────────────────

const SOURCE_FIELDS: (keyof StreamConfig)[] = ['type', 'path', 'username', 'password', 'width', 'height']

/** Returns true if any camera-source field changed between old and new config. */
export function sourceChanged(prev: StreamConfig | null, next: StreamConfig): boolean {
    if (!prev) return true
    return SOURCE_FIELDS.some(k => prev[k] !== next[k])
}

// ── Migration from legacy files ────────────────────────────────────────────

export async function migrateFromLegacy(): Promise<void> {
    await ensureStreamsDir()

    // Check if migration is needed
    const existing = await readdir(streamsDir)
    if (existing.some(f => f.endsWith('.json'))) {
        console.log('Streams dir already has config files, skipping legacy migration')
        return
    }

    const legacySetupFile = Bun.file('/data/streamSetup.json')
    if (!(await legacySetupFile.exists())) {
        console.log('No legacy streamSetup.json found, nothing to migrate')
        return
    }

    let legacy: Record<string, any>
    try {
        legacy = await legacySetupFile.json()
    } catch (err) {
        console.error('Failed to read legacy streamSetup.json:', err)
        return
    }

    console.log('Migrating legacy stream configs...')

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

        // Merge: cam fields + settings overrides + masks
        const config: StreamConfig = {
            ...(cam as StreamConfig),
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
        }

        await writeStreamConfig(camStream, config)
        console.log(`  Migrated stream: ${camStream}`)
    }

    // Rename legacy files to .bak
    const { rename } = await import('node:fs/promises')
    try { await rename('/data/streamSetup.json', '/data/streamSetup.json.bak') } catch { }
    console.log('Legacy migration complete')
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
