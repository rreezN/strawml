import asyncio
from asyncua import Client

async def main(url, sensor_node_id):
    queue = asyncio.Queue()
    await asyncio.gather(
        read_sensor(url, sensor_node_id, queue),
        process_sensor_values(queue),
    )


async def read_sensor(url, sensor_node_id, queue):
    print(f'Connecting to {url}...')
    async with Client(url) as client:
        print(f'Connected to {url}!')
        sensor_node = client.get_node(sensor_node_id)
        while True:
            value = await sensor_node.get_value()
            await queue.put(value)
            await asyncio.sleep(0.5)

async def process_sensor_values(queue):
    while True:
        value = await queue.get()
        handle_sensor_value(value)


def handle_sensor_value(value):
    print(f'Value: {value}')
    # can plot and save the value to a file here

if __name__ == '__main__':
    data_path = 'data/opcua_server.txt'
    txt_file = open(data_path, 'r')
    url = txt_file.readline().strip()
    print(f'Read url: {url} from: {data_path}')
    sensor_node_id = txt_file.readline().strip()
    print(f'Read sensor node id: {sensor_node_id} from: {data_path}')
    asyncio.run(main(url, sensor_node_id))