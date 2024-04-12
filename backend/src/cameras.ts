


import { BunFile, Subprocess } from "bun";
import type { BunFile, Subprocess } from "bun";
import { $ } from "bun";
import type { Context } from "elysia";

const streams = new Map()
const ports = new Map()
let streamSetup: any = {}
const streamSetupFile: BunFile = Bun.file("/data/streamSetup.json");

async function initStreams() {
    const camList = await getCameras()
    console.log({ camList })

    if (!camList.length) {
        console.log('NO CAMERA DETECTED')
        return
    }
    const firstCam = camList[0]
    
    const exists: boolean = await streamSetupFile.exists();
    if (!exists) {
        Bun.write(streamSetupFile, JSON.stringify({frontCam: firstCam.id}))
    }
    
    try {
        streamSetup = await streamSetupFile.json()
    } catch(err) {
        console.error('errrrrr', err)
        Bun.write(streamSetupFile, JSON.stringify({frontCam: firstCam.id}))
        streamSetup = await streamSetupFile.json()
    }

    console.log('StreamSetup', streamSetup)
    for (const [cam, dev] of Object.entries(streamSetup)) {
        runVideoStream(dev as string, cam as string)
    }
}

initStreams()

export function getStreamSetup(ctx: Context): any {
    const params = ctx.query as any
    console.log('getStreamSetup', params)
    return { "device": streamSetup[params.cam] }
}

export const selectCamera = async (ctx: Context) => {
    const { device, cam }: { device: string, cam: string } = JSON.parse(ctx.body as any)
    console.log('selected camera', { device, cam }, [...streams.keys()], [...ports.keys()])

    streamSetup[cam] = device
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup));

    await killVideoStream(device, cam)

    runVideoStream(device, cam)
}

async function runVideoStream(device: string, cam: string) {
    console.log('running video stream for ', device, cam)
    streams.set(device, 'starting')
    ports.set(cam, 'starting')
    while(streams.get(device) || ports.get(cam)) {
        await startVideoStream(device, cam)
    }
}

async function startVideoStream(device: string, cam: string) {
    const proc = Bun.spawn(["python3", "-u", "video/videoStream.py", device, cam], {
        env: { ...process.env},
        stderr: "inherit",
        stdout: "inherit"
    });

    streams.set(device, proc)
    ports.set(cam, proc)
   
    await proc.exited

    console.error('Python video stream exited', proc.exitCode, proc.signalCode)

}

async function killVideoStream(device: string, cam: string) {
    console.log('Killing video stream (if exists) on ', device, cam)

    const proc: Subprocess = streams.get(device)

    if (proc) {
        console.log('killing device process', proc)
        streams.delete(device)
        proc.kill()
        await proc.exited
    }

    const pproc: Subprocess = ports.get(cam)

    if (pproc) {
        console.log('killing cam process', pproc)
        ports.delete(cam)
        pproc.kill()
        await pproc.exited
    }
}

export async function getCameras() {
    const camerasOutput = await $`video/list-cameras.sh`.text()
    return camerasOutput.split("\n").slice(0, -1).map((v: any) => {
        const [path, name, DEVPATH] = v.split(":")
        const id = DEVPATH.replace("/devices/platform/", "").split("/video4linux")
        return { path, name, id }
    })
}
