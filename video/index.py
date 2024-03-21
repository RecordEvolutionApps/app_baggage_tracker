from aiohttp import web
import asyncio
import aiohttp_cors
from camera_handler import get_cameras, get_stream_setup, select_camera, init_streams
from reswarm import Reswarm
import functools
print = functools.partial(print, flush=True)

async def index(request):
    return web.FileResponse('web/dist/index.html')

@web.middleware
async def middleware(request: web.Request, handler):

    print(f"Received request: {request.method} {request.path}")

    resp = await handler(request)
    return resp

app = web.Application(middlewares=[middleware])

# Define routes
app.router.add_get('/cameras', lambda x: get_cameras())
app.router.add_get('/cameras/setup', get_stream_setup)
app.router.add_post('/cameras/select', select_camera)
# app.router.add_post('/mir/status', get_status)
app.router.add_get('/', index)
app.router.add_static('/', path='./web/dist')

# Setup CORS
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    ),
})

# Configure CORS on all routes
for route in list(app.router.routes()):
    cors.add(route)


async def main():
    # Start the application
    rw = Reswarm(mainFunc=init_streams)
    rw._component.start()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=1100)    
    await site.start()
    print('WEB SERVER STARTED on PORT 1100')

    await asyncio.Event().wait()

asyncio.run(main())