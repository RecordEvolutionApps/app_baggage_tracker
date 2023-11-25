// import v4l2camera from 'v4l2camera';

import type { BunFile, Subprocess } from "bun";
import type { Context } from "elysia";

const streams = new Map()
let streamSetup: any = {}
const streamSetupFile: BunFile = Bun.file("streamSetup.json");

async function initStreams() {
    const exists = await streamSetupFile.exists();
    if (!exists) {
        const camList = getCameras()
        if (!camList.length) {
            console.log('NO CAMERA DETECTED')
            return
        }
        const firstCam = camList[0]
        await startVideoStream(firstCam.path, 'frontCam')

    } 
    streamSetup = await streamSetupFile.json();

    console.log('StreamSetup', streamSetup)
    for (const [cam, dev] of Object.entries(streamSetup)) {
        await startVideoStream(dev as string, cam as string)
    }
}

initStreams()

export const getCameras = (): any[] => {
    console.log('getting cameras')
    const cameraList = [];

    // // Iterate over video devices (assumed to be cameras)
    // for (let i = 0; i < 10; i++) {
    //   const devicePath = `/dev/video${i}`;
    //   const camera = new v4l2camera.Camera(devicePath);
  
    //   // Try to open the camera
    //   if (camera.open()) {
    //     // Get camera information
    //     const cameraInfo = {
    //       path: devicePath,
    //       name: camera.configGet().name,
    //     };
  
    //     // Add camera information to the list
    //     cameraList.push(cameraInfo);
  
    //     // Close the camera
    //     camera.close();
    //   }
    // }

    return [
        {
          path: '/dev/video0',
          name: 'Logitech HD Webcam',
        },
        {
          path: '/dev/video1',
          name: 'Canon EOS 5D Mark IV',
        },
        {
          path: '/dev/video2',
          name: 'Microsoft LifeCam HD-3000',
        },
      ]

    return cameraList
}

export const selectCamera = async (ctx: Context) => {
    
    const {device, cam}: {device: string, cam: string} = JSON.parse(ctx.body as any)
    console.log('selected camera', {device, cam})

    streamSetup[cam] = device
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup));

    const proc: Subprocess = streams.get(device)

    if (proc) {
        proc.kill()
        await proc.exited
    }

    startVideoStream(device, cam)
}

async function startVideoStream(device: string, cam: string) {
    const proc = Bun.spawn(["python", "-u", "index.py", device, cam], {
        cwd: "src", // specify a working directory
        env: { ...process.env}, // specify environment variables
        onExit(proc, exitCode, signalCode, error) {
          console.log('Failed to run python video stream', error)
          streams.delete(device)
        },
      });
      
      streams.set(device, proc)
}
