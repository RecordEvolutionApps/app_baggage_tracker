// import v4l2camera from 'v4l2camera';

import { Subprocess } from "bun";
import { Context } from "elysia";

const streams = new Map()

export const getCameras = (ctx: Context) => {
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
    console.log('selected camera', ctx.body)

    const device: string = ctx.body?.selected

    const proc: Subprocess = streams.get(device)

    if (proc) {
        proc.kill()
        await proc.exited
    }

    startVideoStream(device)
}

async function startVideoStream(device: string) {
    const proc2 = Bun.spawn(["pwd"]);
    const text = await new Response(proc2.stdout).text();
    console.log(text); // => "hello"

    const proc = Bun.spawn(["python", "index.py", device], {
        cwd: "src", // specify a working directory
        env: { ...process.env}, // specify environment variables
        onExit(proc, exitCode, signalCode, error) {
          console.log('Failed to run python video stream', error)
          streams.delete(device)
        },
      });
      
      streams.set(device, proc)
}
