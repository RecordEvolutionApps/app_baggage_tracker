import asyncio
from reswarm import Reswarm

# create a reswarm instance, which auto connects to the Record Evolution Platform
# the reswarm instance handles authentication and reconnects when connection is lost
rw = Reswarm()

async def main():
    """Publishes sample data every 2 seconds to the 're.hello.world' topic
    """
    while True:
        data = {"temperature": 20}
        # publish an event (if connection is not established the publish is skipped)
        await rw.publish('re.hello.world', data)
        print(f'Published {data} to topic re.hello.world')
        await asyncio.sleep(2)


if __name__ == "__main__":
    # run the main coroutine
    asyncio.get_event_loop().create_task(main())
    # run the reswarm component
    rw.run()
