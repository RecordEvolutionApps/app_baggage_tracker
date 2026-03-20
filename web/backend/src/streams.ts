import type { Context } from "elysia";
import {
    type StreamConfig,
    type Camera,
    VIDEO_API,
    ports,
    readStreamConfig,
    writeStreamConfig,
    deleteStreamConfig,
    listStreamConfigs,
    sourceChanged,
    waitForService,
    migrateFromLegacy,
    migrateFromFiles,
} from './shared.js'

// ── Stream lifecycle ───────────────────────────────────────────────────────

export async function startVideoStream(cam: StreamConfig, camStream: string) {
    console.log('starting video stream for', cam, camStream)

    let camPath: string
    if (cam.type === 'IP') {
        const rawPath = cam.path ?? ''
        if (!rawPath || !rawPath.includes('://')) {
            camPath = rawPath
        } else {
            const [protocol, path] = rawPath.split('://')
            const userpw = cam.username ? (cam.username + (cam.password ? `:${cam.password}@`: '@')) : ''
            camPath = `${protocol}://${userpw}${path}`
        }
    } else if (cam.type === 'Image') {
        camPath = cam.path ?? ''
    } else {
        camPath = cam.path ?? ''
    }

    const body: Record<string, any> = { camPath, camStream }
    if (cam.width) body.width = cam.width
    if (cam.height) body.height = cam.height

    try {
        const res = await fetch(`${VIDEO_API}/streams`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        })
        if (!res.ok) {
            const text = await res.text()
            console.error(`Failed to start stream ${camStream}:`, res.status, text)
            console.log(`Retrying stream ${camStream} in 5s...`)
            await new Promise(r => setTimeout(r, 5000))
            const retry = await fetch(`${VIDEO_API}/streams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
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
        console.log(`Retrying stream ${camStream} in 5s...`)
        await new Promise(r => setTimeout(r, 5000))
        try {
            const retry = await fetch(`${VIDEO_API}/streams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
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

export async function killVideoStream(camPath: string, camStream: string, method: 'stop' | 'delete' = 'delete') {
    console.log(`${method === 'delete' ? 'Deleting' : 'Stopping'} video stream (if exists) for`, camStream)

    try {
        const url = method === 'stop'
            ? `${VIDEO_API}/streams/${camStream}/stop`
            : `${VIDEO_API}/streams/${camStream}`
        const httpMethod = method === 'stop' ? 'POST' : 'DELETE'
        const res = await fetch(url, { method: httpMethod })
        if (res.ok) {
            const data: any = await res.json()
            console.log(`Stream ${camStream} ${method === 'delete' ? 'deleted' : 'stopped'}:`, data)
        } else if (res.status !== 404) {
            console.error(`Failed to ${method} stream ${camStream}:`, res.status, await res.text())
        }
    } catch (err) {
        console.error(`Error ${method === 'delete' ? 'deleting' : 'stopping'} stream ${camStream}:`, err)
    }

    ports.delete(camStream)
}

// ── Backend status ─────────────────────────────────────────────────────────

export async function getStreamBackendStatus(ctx: Context): Promise<any> {
    const url = new URL(ctx.request.url)
    const camStream = url.pathname.split('/streams/')[1]?.split('/')[0]
    if (!camStream) {
        ctx.set.status = 400
        return { error: 'camStream is required' }
    }
    try {
        const res = await fetch(
            `${VIDEO_API}/streams/${encodeURIComponent(camStream)}/backend`,
            { signal: AbortSignal.timeout(5000) },
        )
        if (res.ok) return await res.json()
        ctx.set.status = res.status
        return { error: `Video API returned ${res.status}` }
    } catch (err) {
        console.error('Failed to fetch backend status:', err)
        ctx.set.status = 502
        return { error: 'Could not reach video API' }
    }
}

// ── Initialization ─────────────────────────────────────────────────────────

async function initStreams() {
    // Migrate from legacy file layouts if needed
    await migrateFromLegacy()
    await migrateFromFiles()

    const configs = await listStreamConfigs()

    if (configs.length === 0) {
        console.log('No stream configs found in backend table — waiting for streams to be created')
        return
    }

    console.log('Stream configs:', configs.map(c => c.camStream))

    const videoReady = await waitForService(`${VIDEO_API}/cameras`, 'Video API')
    if (!videoReady) {
        console.error('Video API not available — streams will not be started')
        return
    }

    for (const config of configs) {
        if (config.camStream && !config.stopped) {
            startVideoStream(config, config.camStream)
        }
    }
}

initStreams()

// ── Route handlers ─────────────────────────────────────────────────────────

/** GET /streams — list all stream configs */
export async function handleListStreams(): Promise<StreamConfig[]> {
    return listStreamConfigs()
}

/** GET /streams/:camStream — read a single stream config from the backend table */
export async function handleGetStream(ctx: Context): Promise<any> {
    const camStream = (ctx.params as any)?.camStream
    if (!camStream) {
        ctx.set.status = 400
        return { error: 'camStream is required' }
    }
    const config = await readStreamConfig(decodeURIComponent(camStream))
    if (!config) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }

    let width = config.width
    let height = config.height

    // If the camera config doesn't have a resolution, read it from the
    // status file written by the video process after opening the source.
    if (!width || !height) {
        try {
            const resFile = Bun.file(`/data/status/${camStream}.resolution.json`)
            if (await resFile.exists()) {
                const res = await resFile.json()
                width = res.width ?? width
                height = res.height ?? height
            }
        } catch (_) { /* resolution file not written yet */ }
    }

    return {
        ...config,
        width: width ?? 640,
        height: height ?? 480,
    }
}

/** POST /streams — create a new stream */
export async function handleCreateStream(ctx: Context) {
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
    const existing = await readStreamConfig(camStream)
    if (existing) {
        ctx.set.status = 409
        return { error: `Stream "${camStream}" already exists` }
    }
    const config: StreamConfig = {
        id: camStream,
        type: 'IP',
        name: name || camStream,
        path: '',
        camStream,
        masks: { polygons: [] },
    }
    await writeStreamConfig(camStream, config)
    return config
}

/** PUT /streams/:camStream — update a stream config */
export async function handleUpdateStream(ctx: Context) {
    const camStream = (ctx.params as any)?.camStream
    if (!camStream) {
        ctx.set.status = 400
        return { error: 'camStream is required' }
    }
    const decoded = decodeURIComponent(camStream)

    let incoming: StreamConfig
    try {
        incoming = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body as StreamConfig
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }

    // Ensure camStream is consistent
    incoming.camStream = decoded

    // Migrate legacy type
    if (incoming.type === 'IP' && incoming.path === 'demoVideo') {
        incoming.type = 'Demo'
    }

    // Read previous config to detect source changes
    const prev = await readStreamConfig(decoded)

    // For USB cameras, resolve device info from the video API
    if (incoming.type === 'USB') {
        try {
            const camList = await getUSBCameras()
            const cameraDev = camList.find((c: any) => c.id === incoming.id)
            if (cameraDev) {
                incoming.path = (cameraDev as any).path ?? incoming.path
                incoming.name = (cameraDev as any).name ?? incoming.name
            }
        } catch { /* proceed with what we have */ }
    }

    // Ensure masks default
    if (!incoming.masks) {
        incoming.masks = prev?.masks ?? { polygons: [] }
    }

    // Write the full config to the backend table
    const status = incoming.stopped ? 'stopped' : 'configured'
    await writeStreamConfig(decoded, incoming, status)

    // Determine if we need to restart the video process
    const needsRestart = sourceChanged(prev, incoming)

    if (needsRestart && prev) {
        console.log(`Source changed for ${decoded}, restarting video process`)
        await killVideoStream(prev.path ?? '', decoded, 'stop')
    }

    if (needsRestart && incoming.path && !incoming.stopped) {
        startVideoStream(incoming, decoded)
    }

    return { status: 'ok', camStream: decoded }
}

/** DELETE /streams/:camStream — delete a stream */
export async function handleDeleteStream(ctx: Context) {
    const camStream = (ctx.params as any)?.camStream
    if (!camStream) {
        ctx.set.status = 400
        return { error: 'camStream is required' }
    }
    const decoded = decodeURIComponent(camStream)
    const config = await readStreamConfig(decoded)
    if (!config) {
        ctx.set.status = 404
        return { error: `Stream "${decoded}" not found` }
    }
    await killVideoStream(config.path ?? '', decoded, 'delete')
    await deleteStreamConfig(decoded)
    return { status: 'deleted', camStream: decoded }
}

/** POST /streams/:camStream/stop — pause a stream */
export async function handleStopStream(ctx: Context) {
    const camStream = (ctx.params as any)?.camStream
    if (!camStream) {
        ctx.set.status = 400
        return { error: 'camStream is required' }
    }
    const decoded = decodeURIComponent(camStream)
    const config = await readStreamConfig(decoded)
    if (!config) {
        ctx.set.status = 404
        return { error: `Stream "${decoded}" not found` }
    }
    await killVideoStream(config.path ?? '', decoded, 'stop')
    config.stopped = true
    await writeStreamConfig(decoded, config, 'stopped')
    return { status: 'stopped', camStream: decoded }
}

/** POST /streams/:camStream/start — resume a stream */
export async function handleStartStream(ctx: Context) {
    const camStream = (ctx.params as any)?.camStream
    if (!camStream) {
        ctx.set.status = 400
        return { error: 'camStream is required' }
    }
    const decoded = decodeURIComponent(camStream)
    const config = await readStreamConfig(decoded)
    if (!config) {
        ctx.set.status = 404
        return { error: `Stream "${decoded}" not found` }
    }
    config.stopped = false
    await writeStreamConfig(decoded, config, 'started')
    startVideoStream(config, decoded)
    return { status: 'started', camStream: decoded }
}

// ── Camera discovery ───────────────────────────────────────────────────────

export async function getUSBCameras(): Promise<Camera[]> {
    return getDeviceCameras()
}

export async function getDeviceCameras(): Promise<Camera[]> {
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
