


import { BunFile, Subprocess } from "bun";
import { $ } from "bun";
import type { Context } from "elysia";

const streams = new Map()
const ports = new Map()
let streamSetup: any = {}
const streamSetupFile: BunFile = Bun.file("/data/streamSetup.json");

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
    } catch(err) {
        console.error('errrrrr', err)
        await Bun.write(streamSetupFile, JSON.stringify({ frontCam: firstCam }))
        streamSetup = await streamSetupFile.json()
    }

    console.log('StreamSetup', streamSetup)
    for (const [camName, { id }] of Object.entries(streamSetup)) {
        runVideoStream(id, camName)
    }
}

initStreams()

export function getStreamSetup(ctx: Context): any {
    const params = ctx.query as any
    console.log('getStreamSetup', params)
    return { "device": streamSetup[params.cam] }
}

export const selectCamera = async (ctx: Context) => {
    const { id, deviceName }: { id: string, deviceName: string } = JSON.parse(ctx.body as any)
    console.log('selected camera', { id, deviceName })

    const camList = await getCameras()
    const cameraDev = camList.find((c) => c.id === id)

    streamSetup[deviceName] = cameraDev
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup));

    await killVideoStream(id, deviceName)

    runVideoStream(id, deviceName)
}

async function runVideoStream(deviceId: string, camName: string) {
    console.log('running video stream for ', deviceId, camName)

    await startVideoStream(deviceId, camName)
}

async function startVideoStream(deviceId: string, camName: string) {
    const camList = await getCameras()
    const cameraDev = camList.find((c: any) => c.id === deviceId)

    if (!cameraDev) {
        console.log("Unable to find Device path for ID", deviceId)
        return
    }

    const proc = Bun.spawn(["python3", "-u", "video/videoStream.py", cameraDev.path, camName], {
        env: { ...process.env },
        onExit: async (proc, exitCode, signalCode, error) => {
            console.log("Proccess exited with", { exitCode, signalCode, error })
            if (exitCode > 0) {
                await startVideoStream(deviceId, camName)
            }
        },
        stderr: "inherit",
        stdout: "inherit"
    });

    streams.set(deviceId, proc)
    ports.set(camName, proc)

    await proc.exited

    console.error('Python video stream exited', proc.exitCode, proc.signalCode)
}

async function killVideoStream(deviceId: string, deviceName: string) {
    console.log('Killing video stream (if exists) on ', deviceId, deviceName)

    const proc: Subprocess = streams.get(deviceId)

    if (proc) {
        console.log('killing device process', proc)
        streams.delete(deviceId)
        proc.kill()
        await proc.exited
    }

    const pproc: Subprocess = ports.get(deviceName)

    if (pproc) {
        console.log('killing cam process', pproc)
        ports.delete(deviceName)
        pproc.kill()
        await pproc.exited
    }
}

export async function getCameras() {
    const camerasOutput = await $`video/list-cameras.sh`.text()
    return camerasOutput.split("\n").slice(0, -1).map((v: any) => {
        const [path, name, DEVPATH] = v.split(":")
        const [id] = DEVPATH.replace("/devices/platform/", "").split("/video4linux") ?? []
        return { path, name, id }
    })
}
