import json
import os
import asyncio
from aiohttp import web

import cv2
import json
import pyudev

streams = {}
ports = {}
stream_setup = {}
stream_setup_file_path = "/data/streamSetup.json"

async def get_cameras():

    context = pyudev.Context()

    result = []

    for device in context.list_devices(subsystem='video4linux'):
        device_path = device.device_node
        cap = cv2.VideoCapture(device_path)
        #
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #
            # Get the USB device name using pyudev
            usb_device_name = device.get('ID_MODEL', 'Unknown')
            usb_serial_number = device.get('ID_SERIAL_SHORT', 'Unknown')
        #
        else:
            # If unable to open, retrieve available information
            width = 0
            height = 0
            usb_device_name = device.get('ID_MODEL', 'Unknown')
            usb_serial_number = device.get('ID_SERIAL_SHORT', 'Unknown')
        
        # if width > 0:
        result.append({
            "path": device_path,
            "name": usb_device_name,
            "serial": usb_serial_number,
            "width": width,
            "height": height
        })
        #
        cap.release()
    result = [{"path": '/dev/video0'}, {"path": '/dev/video1'}, {"path": '/dev/video2'}, {"path": '/dev/video3'}]
    print('CAMLIST', result)
    return web.json_response(result)

async def init_streams():
    try:
        with open(stream_setup_file_path, 'r') as file:
            stream_setup = json.load(file)
    except FileNotFoundError:
        cam_list = await get_cameras()
        if not cam_list:
            print('NO CAMERA DETECTED')
            return
        first_cam = cam_list[0]
        kill_stream(dev, cam)
        asyncio.create_task(start_video_stream(first_cam['path'], 'frontCam'))

    print('StreamSetup', stream_setup)
    for cam, dev in stream_setup.items():
        kill_stream(dev, cam)
        asyncio.create_task(start_video_stream(dev, cam))

async def start_video_stream(device, cam):
    try:
        print('Starting video stream on ', device, cam)
        proc = await asyncio.create_subprocess_exec(
            "python3", "-u", "videoStream.py", device, cam,
            cwd="/app/video", 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE, 
            env=dict(**os.environ)
        )

        streams[device] = proc
        ports[cam] = proc

        while True:
            if proc.returncode is not None:
                print('VIDEOSTREAM Exited with code', proc.returncode)
                break
            chunk = await proc.stdout.readline()
            error = await proc.stderr.readline()
            if chunk:
                print('VIDEOSTREAM:', chunk.decode())
            if error:
                print('VIDEOSTREAM ERROR:', error.decode())
            await asyncio.sleep(1)
        print('############### exiting stream', device, cam)
    except Exception as err:
        print('################ Failed video process ##################################', device, cam)
        print(err)

def kill_stream(device, cam):
    print('Killing video stream (if exists) on ', device, cam)

    proc = streams.get(device)
    if proc:
        print('killing device process', proc, device)
        try:
            proc.kill()
        except:
            print('device kill failed', device , proc.pid)
        print('device process killed', device, proc.pid)
        # await asyncio.sleep(2)
        # proc._transport.close()
        # print('device process killed finally', device)
    if device in streams: del streams[device]

    pproc = ports.get(cam)
    if pproc and proc != pproc:
        print('killing cam process', pproc, cam)
        try:
            pproc.kill()
        except:
            print('cam kill failed', cam , pproc.pid)
        print('cam process killed', cam, pproc.pid)
        # await asyncio.sleep(2)
        # pproc._transport.close()
        # print('cam process killed finally', cam)
    if cam in ports: del ports[cam]
    print('kill_stream done')

# Exported functions
def get_stream_setup(request):
    params = request.query
    print('getStreamSetup', params)
    return web.json_response({"device": stream_setup.get(params['cam'])})

async def select_camera(request):
    data = await request.json()
    device, cam = data['device'], data['cam']
    print('selected camera', {'device': device, 'cam': cam}, list(streams.keys()), list(ports.keys()))

    stream_setup[cam] = device
    with open(stream_setup_file_path, 'w') as file:
        json.dump(stream_setup, file)

    kill_stream(device, cam)

    asyncio.create_task(start_video_stream(device, cam))
