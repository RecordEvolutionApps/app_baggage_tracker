


import { BunFile } from "bun";
import type { Context } from "elysia";

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
        if (camStream && cam) startVideoStream(cam, camStream)
    }
}

initStreams()

export function getStreamSetup(ctx: Context): any {
    // const params = ctx.query as any
    const params = new URLSearchParams(ctx.request.url.split('?')[1])
    console.log('getStreamSetup', params.get('camStream'))
    return { "camera": streamSetup[params.get('camStream') ?? ''], "width": process.env.RESOLUTION_X, "height": process.env.RESOLUTION_Y }
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

export function updateStreamMask(maskData: any) {
    // Send mask update to all running streams via the video API
    for (const camStream of ports.keys()) {
        fetch(`${VIDEO_API}/streams/${camStream}/mask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(maskData),
        }).catch(err => console.error(`Error sending mask to ${camStream}:`, err))
    }
}
