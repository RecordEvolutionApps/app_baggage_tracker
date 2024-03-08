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
        
        if width > 0:
            result.append({
                "path": device_path,
                "name": usb_device_name,
                "serial": usb_serial_number,
                "width": width,
                "height": height
            })
        #
        cap.release()

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
        await start_video_stream(first_cam['path'], 'frontCam')

    print('StreamSetup', stream_setup)
    for cam, dev in stream_setup.items():
        await start_video_stream(dev, cam)

async def start_video_stream(device, cam):
    proc = await asyncio.create_subprocess_exec(
        "python3", "-u", "videoStream.py", device, cam,
        cwd="/app/video", stdout=asyncio.subprocess.PIPE, env=dict(**os.environ)
    )

    streams[device] = proc
    ports[cam] = proc

    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        print('VIDEOSTREAM:', chunk.decode())

# Exported functions
async def get_stream_setup(request):
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

    proc = streams.get(device)
    if proc:
        print('killing device process', proc)
        proc.terminate()
        await proc.wait()
    if device in streams: del streams[device]

    pproc = ports.get(cam)
    if pproc and proc != pproc:
        print('killing cam process', pproc)
        pproc.terminate()
        await pproc.wait()
    if cam in ports: del ports[cam]

    await start_video_stream(device, cam)
