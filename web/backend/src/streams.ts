import type { Context } from "elysia";
import {
    type Camera,
    VIDEO_API,
    ports,
    streamSetup,
    streamSetupFile,
    setStreamSetup,
    waitForService,
    writeStreamSettings,
} from './shared.js'

// ── Stream lifecycle ───────────────────────────────────────────────────────

export async function startVideoStream(cam: Camera, camStream: string) {
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
    } else {
        camPath = cam.path ?? ''
    }

    try {
        const body: Record<string, any> = { camPath, camStream }
        if (cam.width) body.width = cam.width
        if (cam.height) body.height = cam.height

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

export async function killVideoStream(camPath: string, camStream: string) {
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

// ── Backend status ─────────────────────────────────────────────────────────

export async function getStreamBackendStatus(ctx: Context): Promise<any> {
    const url = new URL(ctx.request.url)
    const camStream = url.pathname.split('/cameras/streams/')[1]?.split('/')[0]
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
    let camList: Camera[] = []
    try {
        camList = await getUSBCameras()
    } catch (error) {
        console.error('Failed to load cameras', error)
    }
    console.log("CAMERA LIST", { camList })

    const firstCam = camList?.[0]

    const exists: boolean = await streamSetupFile.exists();
    if (!exists && firstCam) {
        await Bun.write(streamSetupFile, JSON.stringify({ frontCam: firstCam }))
    }

    let loaded: Record<string, Camera> = {}
    try {
        loaded = await streamSetupFile.json()
    } catch (err) {
        console.error('error loading streamSetup file:', err)
        await Bun.write(streamSetupFile, JSON.stringify({}))
        loaded = {}
    }

    if (Object.keys(loaded).length === 0 && !firstCam) {
        const demoCam: Camera = {
            id: 'frontCam',
            type: 'Demo',
            name: 'Demo Video',
            path: 'demoVideo',
            camStream: 'frontCam',
        }
        loaded = { frontCam: demoCam }
        await Bun.write(streamSetupFile, JSON.stringify(loaded))
    }

    // Migrate legacy entries: IP + demoVideo → Demo type
    for (const cam of Object.values(loaded)) {
        if (cam.type === 'IP' && cam.path === 'demoVideo') {
            (cam as any).type = 'Demo'
        }
    }

    setStreamSetup(loaded)
    console.log('StreamSetup', streamSetup)

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

// ── Route handlers ─────────────────────────────────────────────────────────

export async function getStreamSetup(ctx: Context): Promise<any> {
    const params = new URLSearchParams(ctx.request.url.split('?')[1])
    const camStream = params.get('camStream') ?? ''
    console.log('getStreamSetup', camStream)
    const cam = streamSetup[camStream]

    let width = cam?.width
    let height = cam?.height

    // If the camera config doesn't have a resolution (Demo, YouTube, IP/RTSP),
    // read the actual resolution written by the video process after it opened
    // the source.
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
        "camera": cam,
        "width": width ?? 640,
        "height": height ?? 480,
    }
}

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
    const cam: Camera = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body as Camera
    console.log('selected camera', cam)

    // Migrate legacy type: demoVideo path → Demo type
    if (cam.type === 'IP' && cam.path === 'demoVideo') {
        cam.type = 'Demo'
    }

    if (cam.type === 'IP' || cam.type === 'YouTube' || cam.type === 'Demo') {
        streamSetup[cam.camStream] = cam
    }
    else {
        const camList = await getUSBCameras()
        const cameraDev = camList.find((c) => c.id === cam.id)
        if (!cameraDev) {
            ctx.set.status = 404
            return { error: `Camera "${cam.id}" not found` }
        }
        // Merge USB device info with user-chosen resolution
        streamSetup[cam.camStream] = {
            ...cameraDev,
            camStream: cam.camStream,
            type: 'USB',
            width: cam.width,
            height: cam.height,
        }
    }

    const startCam = streamSetup[cam.camStream]
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup));
    await writeStreamSettings(cam.camStream, startCam)

    await killVideoStream(startCam.path ?? '', cam.camStream)

    startVideoStream(startCam, cam.camStream)
}

export const getUSBCameras = getDeviceCameras   // backward compat alias

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
