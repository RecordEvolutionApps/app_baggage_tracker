


import { BunFile, Subprocess } from "bun";
import { $ } from "bun";
import type { Context } from "elysia";

export const streams: Map<string, Subprocess<"ignore", "pipe", "inherit">> = new Map()
const ports = new Map()
let streamSetup: any = {}
const streamSetupFile: BunFile = Bun.file("/data/streamSetup.json");
const camStreams = ['frontCam', 'backCam', 'leftCam', 'rightCam']


async function initStreams() {
    const camList = await getCameras()
    console.log("CAMERA LIST", { camList })

    if (!camList.length) {
        console.log('NO CAMERA DETECTED')
        return
    }
    const firstCam = camList[0]

    const exists: boolean = await streamSetupFile.exists();
    console.log("setup file exists: ", exists)
    if (!exists) {
        await Bun.write(streamSetupFile, JSON.stringify({ frontCam: firstCam }))
    }

    try {
        streamSetup = await streamSetupFile.json()
    } catch (err) {
        console.error('errrrrr', err)
        await Bun.write(streamSetupFile, JSON.stringify({ frontCam: firstCam }))
        streamSetup = await streamSetupFile.json()
    }

    console.log('StreamSetup', streamSetup)
    for (const [camStream, { id }] of Object.entries(streamSetup)) {
        if (id && camStream) runVideoStream(id, camStream)
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
    const { id, camStream }: { id: string, camStream: string } = JSON.parse(ctx.body as any)
    console.log('selected camera', { id, camStream })

    const camList = await getCameras()
    const cameraDev = camList.find((c) => c.id === id)

    streamSetup[camStream] = cameraDev
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup));

    await killVideoStream(id, camStream)

    runVideoStream(id, camStream)
}

async function runVideoStream(deviceId: string, camStream: string) {
    if (!camStreams.includes(camStream)) {
        console.error('Error camStream is illegal:', camStream)
        return
    }
    console.log('running video stream for ', deviceId, camStream)

    await startVideoStream(deviceId, camStream)
}

async function startVideoStream(deviceId: string, camStream: string) {
    const camList = await getCameras()
    const cameraDev = camList.find((c: any) => c.id === deviceId)

    if (!cameraDev) {
        console.log("Unable to find Device path for ID", deviceId)
        return
    }

    const proc = Bun.spawn(["ssh", "-o", "StrictHostKeyChecking=no", "video", "source",  "~/env_vars.txt", "&&", "python3", "-u", "/app/video/videoStream.py", cameraDev.path, camStream], {
        env: { ...process.env },
        onExit: async (proc, exitCode, signalCode, error) => {
            console.log("Proccess exited with", { exitCode, signalCode, error })
            if (exitCode > 0) {
                await startVideoStream(deviceId, camStream)
            }
        },
        stderr: "inherit",
        stdout: "inherit",
        stdin: "pipe",
    });

    streams.set(deviceId, proc)
    ports.set(camStream, proc)

    await proc.exited

    console.error('Python video stream exited', proc.exitCode, proc.signalCode)
}

async function killVideoStream(deviceId: string, camStream: string) {
    console.log('Killing video stream (if exists) on ', deviceId, camStream)

    const proc: Subprocess = streams.get(deviceId)

    if (proc) {
        console.log('killing device process', proc)
        streams.delete(deviceId)
        proc.kill()
        await proc.exited
    }

    const pproc: Subprocess = ports.get(camStream)

    if (pproc) {
        console.log('killing cam process', pproc)
        ports.delete(camStream)
        pproc.kill()
        await pproc.exited
    }
}

export async function getCameras() {
    const camerasOutput = await $`ssh -o StrictHostKeyChecking=no video /app/video/list-cameras.sh`.text()
    return camerasOutput.split("\n").slice(0, -1).map((v: any) => {
        const [path, name, DEVPATH] = v.split(":")
        const [id] = DEVPATH.replace("/devices/platform/", "").split("/video4linux") ?? []
        return { path, name, id }
    })
}
