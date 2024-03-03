from aiohttp import web
import asyncio
import aiohttp_cors
from camera_handler import get_cameras, get_stream_setup, select_camera, init_streams
from reswarm import Reswarm

async def index(request):
    return web.FileResponse('web/dist/index.html')

app = web.Application()

# Define routes
app.router.add_get('/cameras', get_cameras)
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
    # add stuff to the loop, e.g. using asyncio.create_task()
    # Start the application
    rw = Reswarm(mainFunc=init_streams)
    rw._component.start()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=1100)    
    await site.start()
    print('WEB SERVER STARTED on PORT 1100')

    # add more stuff to the loop, if needed

    # asyncio.create_task(init_streams())

    # wait forever
    await asyncio.Event().wait()

asyncio.run(main())