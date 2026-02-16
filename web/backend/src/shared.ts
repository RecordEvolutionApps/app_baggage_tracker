import { BunFile } from "bun";
import { mkdir } from 'node:fs/promises'

// ── Config constants ───────────────────────────────────────────────────────

export const VIDEO_API = Bun.env.VIDEO_API || "http://video:8000";
export const MEDIASOUP_WS = Bun.env.MEDIASOUP_WS || "ws://mediasoup:1200";
export const settingsDir = '/data/settings'

// ── Types ──────────────────────────────────────────────────────────────────

export type Camera = {
    id: string
    type: 'USB' | 'IP'
    name: string
    path?: string
    username?: string
    password?: string
    camStream: string
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
}

// ── Shared mutable state ───────────────────────────────────────────────────

export const ports = new Map<string, any>()
export let streamSetup: Record<string, Camera> = {}
export const streamSetupFile: BunFile = Bun.file("/data/streamSetup.json");

export function setStreamSetup(value: Record<string, Camera>) {
    streamSetup = value
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

export async function writeStreamSettings(camStream: string, cam: Camera) {
    await mkdir(settingsDir, { recursive: true })
    const settings: Record<string, any> = {
        model: cam.model ?? 'rtmdet_tiny_8xb32-300e_coco',
        useSahi: cam.useSahi ?? true,
        useSmoothing: cam.useSmoothing ?? true,
        confidence: cam.confidence ?? 0.1,
        frameBuffer: cam.frameBuffer ?? 64,
        nmsIou: cam.nmsIou ?? 0.5,
        sahiIou: cam.sahiIou ?? 0.5,
        overlapRatio: cam.overlapRatio ?? 0.2,
        classList: cam.classList ?? [],
        classNames: cam.classNames ?? [],
    }
    await Bun.write(`${settingsDir}/${camStream}.json`, JSON.stringify(settings))
}
