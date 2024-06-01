


import { BunFile, Subprocess } from "bun";
import { $ } from "bun";
import type { Context } from "elysia";

export const streams: Map<string, Subprocess<"ignore", "pipe", "inherit">> = new Map()
const ports = new Map()
let streamSetup: Record<string, Camera> = {}
const streamSetupFile: BunFile = Bun.file("/data/streamSetup.json");
const camStreams = ['frontCam', 'backCam', 'leftCam', 'rightCam']


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
    const camList = await getUSBCameras()
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

    console.log('StreamSetup', streamSetup)
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
    if (!camStreams.includes(camStream)) {
        console.error('Error camStream is illegal:', camStream)
        return
    }
    console.log('running video stream for ', cam, camStream)

    let camPath: string
    if (cam.type === 'IP') {
        const [protocol, path] = cam.path?.split('://') ?? []
        const userpw =  cam.username ? (cam.username + (cam.password ? `:${cam.password}@`: '@')) : ''
        camPath = `${protocol}://${userpw}${path}`
    } else {
        camPath = cam.path ?? ''
    }

    camPath = `"${camPath}"`

    const proc = Bun.spawn(["ssh", "-o", "StrictHostKeyChecking=no", "video", "source",  "~/env_vars.txt", "&&", "python3", "-u", "/app/video/videoStream.py", camPath, camStream], {
        env: { ...process.env },
        onExit: async (proc, exitCode, signalCode, error) => {
            console.log("Proccess exited with", { exitCode, signalCode, error })
            if (exitCode > 0) {
                await startVideoStream(cam, camStream)
            }
        },
        stderr: "inherit",
        stdout: "inherit",
        stdin: "pipe",
    });

    streams.set(cam.path ?? 'xxx', proc)
    ports.set(camStream, proc)

    await proc.exited

    console.error('Python video stream exited', proc.exitCode, proc.signalCode)
}

async function killVideoStream(camPath: string, camStream: string) {
    console.log('Killing video stream (if exists) on ', camPath, camStream)

    const proc: Subprocess = streams.get(camPath)

    if (proc) {
        console.log('killing device process', proc)
        streams.delete(camPath)
        proc.kill()
        await proc.exited

        const rproc = Bun.spawn(["ssh", "-o", "StrictHostKeyChecking=no", "video", "pkill", "-9", "python3"], {
            env: { ...process.env },
            onExit: async (proc, exitCode, signalCode, error) => {
                console.log("Proccess killed with", { exitCode, signalCode, error })
            },
            stderr: "inherit",
            stdout: "inherit",
            stdin: "pipe",
        });
        await rproc.exited
    }

    const pproc: Subprocess = ports.get(camStream)

    if (pproc) {
        console.log('killing cam process', pproc)
        ports.delete(camStream)
        pproc.kill()
        await pproc.exited

        const rproc = Bun.spawn(["ssh", "-o", "StrictHostKeyChecking=no", "video", "pkill", "-9", "python3"], {
            env: { ...process.env },
            onExit: async (proc, exitCode, signalCode, error) => {
                console.log("Proccess killed with", { exitCode, signalCode, error })
            },
            stderr: "inherit",
            stdout: "inherit",
            stdin: "pipe",
        });
        await rproc.exited
    }
}

export async function getUSBCameras(): Promise<Camera[]> {
    const camerasOutput = await $`ssh -o StrictHostKeyChecking=no video /app/video/list-cameras.sh`.text()
    return camerasOutput.split("\n").slice(0, -1).map((v: any) => {
        const [path, name, DEVPATH] = v.split(":")
        const [id] = DEVPATH.replace("/devices/platform/", "").split("/video4linux") ?? []
        return { path, name, id }
    })
}
