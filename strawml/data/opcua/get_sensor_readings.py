import asyncio
from asyncua import Client

async def main(url, sensor_node_id):
    print(f'Connecting to {url}')
    async with Client(url) as client:
        print('Connected')
        sensor_node = client.get_node(sensor_node_id)
        while True:
            value = await sensor_node.get_value()
            print(f'Sensor value: {value}')
            await asyncio.sleep(1)

if __name__ == '__main__':
    data_path = 'data/opcua_server.txt'
    txt_file = open(data_path, 'r')
    url = txt_file.readline().strip()
    print(f'Read url: {url} from: {data_path}')
    sensor_node_id = txt_file.readline().strip()
    print(f'Read sensor node id: {sensor_node_id} from: {data_path}')
    asyncio.run(main(url, sensor_node_id))