import { BunFile } from "bun";
import type { Context } from "elysia";
import { mkdir } from 'node:fs/promises'

const ports = new Map()
let streamSetup: Record<string, Camera> = {}
const streamSetupFile: BunFile = Bun.file("/data/streamSetup.json");
const VIDEO_API = Bun.env.VIDEO_API || "http://video:8000";
const MEDIASOUP_WS = Bun.env.MEDIASOUP_WS || "ws://mediasoup:1200";

/**
 * Wait for a service to respond to HTTP requests.
 * Retries every `intervalMs` up to `maxRetries` times.
 */
async function waitForService(url: string, name: string, maxRetries = 30, intervalMs = 2000): Promise<boolean> {
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


type Camera = {
    id: string
    type: 'USB' | 'IP'
    name: string
    path?: string
    username?: string
    password?: string
    camStream: string
    model?: string
    useSahi?: boolean
    frameBuffer?: number
  }

const AVAILABLE_MODELS = [
    { id: 'none', label: 'No Inference' },
    { id: 'rtmdet_tiny_8xb32-300e_coco', label: 'RTMDet Tiny' },
    { id: 'rtmdet_s_8xb32-300e_coco', label: 'RTMDet Small' },
    { id: 'rtmdet_m_8xb32-300e_coco', label: 'RTMDet Medium' },
]

const settingsDir = '/data/settings'

async function writeStreamSettings(camStream: string, cam: Camera) {
    await mkdir(settingsDir, { recursive: true })
    const settings = {
        model: cam.model ?? 'rtmdet_tiny_8xb32-300e_coco',
        useSahi: cam.useSahi ?? true,
        frameBuffer: cam.frameBuffer ?? 64,
    }
    await Bun.write(`${settingsDir}/${camStream}.json`, JSON.stringify(settings))
}

async function initStreams() {
    let camList: Camera[] = []
    try {
        camList = await getUSBCameras()
    } catch (error) {
        console.error('Failed to load USB cameras', error)
    }
    console.log("CAMERA LIST", { camList })

    const firstCam = camList?.[0]

    const exists: boolean = await streamSetupFile.exists();
    if (!exists && firstCam) {
        await Bun.write(streamSetupFile, JSON.stringify({ frontCam: firstCam }))
    }

    try {
        streamSetup = await streamSetupFile.json()
    } catch (err) {
        console.error('error loading streamSetup file:', err)
        await Bun.write(streamSetupFile, JSON.stringify({}))
        streamSetup = {}
    }

    if (Object.keys(streamSetup).length === 0 && !firstCam) {
        const demoCam: Camera = {
            id: 'demoVideo',
            type: 'IP',
            name: 'Demo Video',
            path: 'demoVideo',
            camStream: 'frontCam',
        }
        streamSetup = { frontCam: demoCam }
        await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    }

    console.log('StreamSetup', streamSetup)

    // Wait for video API and mediasoup to be ready before starting streams
    const videoReady = await waitForService(`${VIDEO_API}/cameras`, 'Video API')
    if (!videoReady) {
        console.error('Video API not available — streams will not be started')
        return
    }

    for (const [camStream, cam] of Object.entries(streamSetup)) {
        if (camStream && cam) {
            writeStreamSettings(camStream, cam)
            startVideoStream(cam, camStream)
        }
    }
}

initStreams()

export function getStreamSetup(ctx: Context): any {
    const params = new URLSearchParams(ctx.request.url.split('?')[1])
    console.log('getStreamSetup', params.get('camStream'))
    return { "camera": streamSetup[params.get('camStream') ?? ''], "width": process.env.RESOLUTION_X, "height": process.env.RESOLUTION_Y }
}

export function getModels(): any {
    return AVAILABLE_MODELS
}

export async function updateStreamModel(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, model } = body
    if (!camStream || !model) {
        ctx.set.status = 400
        return { error: 'camStream and model are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.model = model
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, model }
}

export async function updateStreamSahi(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, useSahi } = body
    if (!camStream || typeof useSahi !== 'boolean') {
        ctx.set.status = 400
        return { error: 'camStream (string) and useSahi (boolean) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.useSahi = useSahi
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, useSahi }
}

export async function updateStreamFrameBuffer(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, frameBuffer } = body
    if (!camStream || typeof frameBuffer !== 'number' || frameBuffer < 0) {
        ctx.set.status = 400
        return { error: 'camStream (string) and frameBuffer (non-negative number) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.frameBuffer = frameBuffer
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, frameBuffer }
}

// ── Stream CRUD ────────────────────────────────────────────────────────────

export function listStreams(): Camera[] {
    return Object.values(streamSetup)
}

export async function createStream(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch (err) {
        console.error('createStream: failed to parse body', ctx.body, err)
        ctx.set.status = 400
        return { error: 'Invalid JSON body' }
    }
    const { name, camStream } = body
    if (!camStream) {
        ctx.set.status = 400
        return { error: 'camStream is required' }
    }
    if (streamSetup[camStream]) {
        ctx.set.status = 409
        return { error: `Stream "${camStream}" already exists` }
    }
    const newCam: Camera = {
        id: camStream,
        type: 'IP',
        name: name || camStream,
        path: '',
        camStream,
    }
    streamSetup[camStream] = newCam
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    return newCam
}

export async function deleteStream(ctx: Context) {
    const url = new URL(ctx.request.url)
    const camStream = url.pathname.split('/cameras/streams/')[1]
    if (!camStream) {
        ctx.set.status = 400
        return { error: 'camStream is required' }
    }
    const decoded = decodeURIComponent(camStream)
    const cam = streamSetup[decoded]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${decoded}" not found` }
    }
    await killVideoStream(cam.path ?? '', decoded)
    delete streamSetup[decoded]
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    return { status: 'deleted', camStream: decoded }
}

export const selectCamera = async (ctx: Context) => {
    const cam: Camera = JSON.parse(ctx.body as any)
    console.log('selected camera', cam)
    if (cam.type === 'IP') {
        streamSetup[cam.camStream] = cam
    }
    else {
        const camList = await getUSBCameras()
        const cameraDev = camList.find((c) => c.id === cam.id)

        streamSetup[cam.camStream] = cameraDev
    }

    const startCam = streamSetup[cam.camStream]
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup));
    await writeStreamSettings(cam.camStream, startCam)

    await killVideoStream(startCam.path, cam.camStream)

    startVideoStream(startCam, cam.camStream)
}

async function startVideoStream(cam: Camera, camStream: string) {
    console.log('starting video stream for', cam, camStream)

    let camPath: string
    if (cam.type === 'IP') {
        const rawPath = cam.path ?? ''
        if (!rawPath || !rawPath.includes('://')) {
            // Allow bare values like "demoVideo" without forcing a scheme.
            camPath = rawPath
        } else {
            const [protocol, path] = rawPath.split('://')
            const userpw = cam.username ? (cam.username + (cam.password ? `:${cam.password}@`: '@')) : ''
            camPath = `${protocol}://${userpw}${path}`
        }
    } else {
        camPath = cam.path ?? ''
    }

    try {
        const res = await fetch(`${VIDEO_API}/streams`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camPath, camStream }),
        })
        if (!res.ok) {
            const text = await res.text()
            console.error(`Failed to start stream ${camStream}:`, res.status, text)
            // Retry once after a delay (mediasoup may not be ready yet)
            console.log(`Retrying stream ${camStream} in 5s...`)
            await new Promise(r => setTimeout(r, 5000))
            const retry = await fetch(`${VIDEO_API}/streams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ camPath, camStream }),
            })
            if (!retry.ok) {
                console.error(`Retry failed for ${camStream}:`, retry.status, await retry.text())
                return
            }
            const data: any = await retry.json()
            console.log(`Stream ${camStream} started on retry:`, data)
            ports.set(camStream, data.pid)
            return
        }
        const data: any = await res.json()
        console.log(`Stream ${camStream} started:`, data)
        ports.set(camStream, data.pid)
    } catch (err) {
        console.error(`Error starting stream ${camStream}:`, err)
        // Retry once after a delay
        console.log(`Retrying stream ${camStream} in 5s...`)
        await new Promise(r => setTimeout(r, 5000))
        try {
            const retry = await fetch(`${VIDEO_API}/streams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ camPath, camStream }),
            })
            if (retry.ok) {
                const data: any = await retry.json()
                console.log(`Stream ${camStream} started on retry:`, data)
                ports.set(camStream, data.pid)
            } else {
                console.error(`Retry failed for ${camStream}:`, retry.status, await retry.text())
            }
        } catch (retryErr) {
            console.error(`Retry also failed for ${camStream}:`, retryErr)
        }
    }
}

async function killVideoStream(camPath: string, camStream: string) {
    console.log('Stopping video stream (if exists) for', camStream)

    try {
        const res = await fetch(`${VIDEO_API}/streams/${camStream}`, {
            method: 'DELETE',
        })
        if (res.ok) {
            const data: any = await res.json()
            console.log(`Stream ${camStream} stopped:`, data)
        } else if (res.status !== 404) {
            console.error(`Failed to stop stream ${camStream}:`, res.status, await res.text())
        }
    } catch (err) {
        console.error(`Error stopping stream ${camStream}:`, err)
    }

    ports.delete(camStream)
}

export async function getUSBCameras(): Promise<Camera[]> {
    try {
        const res = await fetch(`${VIDEO_API}/cameras`)
        if (!res.ok) {
            console.error('Failed to fetch cameras:', res.status, await res.text())
            return []
        }
        return await res.json() as Camera[]
    } catch (err) {
        console.error('Error fetching cameras:', err)
        return []
    }
}

