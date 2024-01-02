import asyncio
import json
import websocket_connector as wsc
import dino_extraction as dino
import base64
from PIL import Image
from io import BytesIO
import os
import numpy as np


def make_callback(image_consumer=None):

    image_count = 0

    async def callback(message):
        '''
        The message objects are json.
        The image looks like this:
        {
            imageData":
            {
                "time":3520152.700000003,
                "dt":0,
                "data_uri":"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA...",
                "viewSizes":[
                    {"name":"map","left":0,"bottom":0,"width":0.5,"height":1},
                    {"name":"gimbal","left":0.5,"bottom":0.333,"width":0.5,"height":0.667}
                ]}
        }
        '''
        # decode the json
        message = json.loads(message)

        # see if we can decode the image
        try:
            data_uri = message['imageData']['dataUri']
        except:
            print("Error: Could not find image.")
            return

        if data_uri:
            print(f"Received image: {len(data_uri)}")

            # decode the jpeg image and save the pixels into a torch tensor
            base64_image = data_uri.split(",")[1]
            image = Image.open(BytesIO(base64.b64decode(base64_image)))

            # get width and height
            width, height = image.size

            # print the image size
            print(f"Image size: {width}x{height}")

            # print the number of channels
            print(f"Number of channels: {len(image.getbands())}")

        else:
            print("Error: Could not find image data uri.")
            return

        named_rectangles = {}

        # increment the image count
        nonlocal image_count
        image_count += 1

        view_infos = message['imageData']['viewInfos']

        # print the view sizes, one per line, preceded by a tab and the view name
        for view in view_infos:
            x_start = int(view['left'] * width + 1)
            x_end = int((view['left'] + view['width']) * width)

            # y_start is a bit weird, because the origin is the top left corner,
            # but the image position is defined by the bottom offset
            rel_y_start = 1 - view['bottom'] - view['height']
            y_start = int(rel_y_start * height + 1)
            y_end = int((rel_y_start + view['height']) * height)

            tile_width = x_end - x_start
            tile_height = y_end - y_start

            name = view['name']
            print(f"  {name}: {tile_width}x{tile_height}")

            # cut the named view from the image and save it as a file with the pattern:
            # output/<name>_<index>.jpg
            # format the index with leading zeros: 000, 001, 002, ...

            path = f"output/{name}_{image_count:03}.jpg"

            # make sure that the output directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # if the file already exists, delete it first
            if os.path.exists(path):
                os.remove(path)

            tile = image.crop((x_start, y_start, x_end, y_end))

            named_rectangles[name] = tile

            # save for debugging
            # tile.save(f"output/{name}_{image_count:03}.jpg")

        # create a new data object that contains all the fields from the original message,
        # but replace the image data with the named rectangles

        # create a new message object
        new_message = {}

        # replace the image data with the named rectangles
        new_message['imageData'] = {
            'namedRectangles': named_rectangles,
            'time': message['imageData']['time'],
            'dt': message['imageData']['dt'],
            'viewInfos': view_infos
        }

        # iterate over the other fields and copy them
        for key in message:
            if key != 'imageData':
                new_message[key] = message[key]

        # call the image consumer
        if image_consumer:
            await image_consumer(new_message, send_semantic_rays)

        # use open cv to extract the sift features

    return callback


async def send_semantic_rays(phrase_dict, matrix_world: np.ndarray):
    '''
    This function takes a point cloud and sends it to the websocket server.
    matrix_world is a 4x4 matrix stored linearly in column-major order.
    '''

    # print the phrases which are the keys of the dictionary
    print(f"Sending semantic rays: {len(phrase_dict)} phrases: {list(phrase_dict.keys())}")


    # create a new message object
    message = {
        'type': 'semantic_rays',

        # the matrix elements are already in a flat column-major format,
        # so we can just copy the list
        'matrix_world_column_major': [
            float(x) for x in matrix_world
        ],

        'phrase_dict': phrase_dict
    }

    # send the message
    await connector.send(message)


async def attempt_to_connect():
    # Connect to the websocket server in a loop, until it works.
    while True:
        try:
            await connector.connect_websocket()
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying in 2 seconds...")
            await asyncio.sleep(2)

if __name__ == '__main__':

    # create a dino object
    dino_extractor = dino.DinoExtraction()

    # create a connector object
    connector = wsc.WebSocketConnector(
        host='localhost',
        http_port=3000,
        ws_port=18999,
        subscribe='fang-1-sensors',
        publish='fang-1-stereo-cloud',
        callback=make_callback(dino_extractor.make_on_message())
    )

    asyncio.run(attempt_to_connect())
