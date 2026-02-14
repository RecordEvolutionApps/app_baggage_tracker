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
        const res = await fetch(`${VIDEO_API}/streams`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camPath, camStream }),
        })
        if (!res.ok) {
            const text = await res.text()
            console.error(`Failed to start stream ${camStream}:`, res.status, text)
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

// ── Initialization ─────────────────────────────────────────────────────────

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
            id: 'demoVideo',
            type: 'IP',
            name: 'Demo Video',
            path: 'demoVideo',
            camStream: 'frontCam',
        }
        loaded = { frontCam: demoCam }
        await Bun.write(streamSetupFile, JSON.stringify(loaded))
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

export function getStreamSetup(ctx: Context): any {
    const params = new URLSearchParams(ctx.request.url.split('?')[1])
    console.log('getStreamSetup', params.get('camStream'))
    return { "camera": streamSetup[params.get('camStream') ?? ''], "width": process.env.RESOLUTION_X, "height": process.env.RESOLUTION_Y }
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
    const cam: Camera = JSON.parse(ctx.body as any)
    console.log('selected camera', cam)
    if (cam.type === 'IP') {
        streamSetup[cam.camStream] = cam
    }
    else {
        const camList = await getUSBCameras()
        const cameraDev = camList.find((c) => c.id === cam.id)
        if (!cameraDev) {
            ctx.set.status = 404
            return { error: `Camera "${cam.id}" not found` }
        }
        streamSetup[cam.camStream] = cameraDev
    }

    const startCam = streamSetup[cam.camStream]
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup));
    await writeStreamSettings(cam.camStream, startCam)

    await killVideoStream(startCam.path ?? '', cam.camStream)

    startVideoStream(startCam, cam.camStream)
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
