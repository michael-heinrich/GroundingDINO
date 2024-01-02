import websockets
import requests
import json


class WebSocketConnector:
    def __init__(self, host, http_port=80, ws_port=8080, subscribe=None, publish=None, callback=None):
        # if neither subscribe nor publish is specified, raise an error
        if not subscribe and not publish:
            raise ValueError(
                "Either 'subscribe' or 'publish' must be specified.")

        self.subscribe = subscribe
        self.publish = publish
        self.http_port = http_port
        self.ws_port = ws_port
        self.host = host
        self.callback = callback
        self.websocket = None

    async def connect_websocket(self):
        # first, print the permissible values
        server_topics = self.query_topics()

        # If server topics are not None, check if the subscribe topic is present.
        # If not, print a warning.

        if server_topics:
            if self.subscribe not in server_topics:
                print(
                    f"Warning: '{self.subscribe}' is not a valid topic. Permissible values are: {server_topics}")
            else:
                print(f"Subscribing to topic '{self.subscribe}'")
        else:
            print(
                "Warning: Could not retrieve server topics. Please check your connection.")

        async with websockets.connect(f'ws://{self.host}:{self.ws_port}?subscribe={self.subscribe}&publish={self.publish}') as websocket:
            if self.websocket:
                self.websocket.close()
            
            self.websocket = websocket

            while True:
                
                # wait for the next message
                message = await websocket.recv()

                while len(websocket.messages) > 0:
                    # Consume and drop all messages that were in the queue.
                    # The algorithm might be slower than the incoming messages,
                    # so we have to drop potentially outdated messages to remain
                    # in real-time.
                    message = await websocket.recv()



                print(f"Received message: {len(message)}")

                if self.callback:
                    await self.callback(message)

    def query_topics(self):
        # check if http_port is present and omit it from the url if not
        if self.http_port:
            url = f'http://{self.host}:{self.http_port}/topics'
        else:
            url = f'http://{self.host}/topics'

        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError as e:
            print(f"Error: Could not connect to server: {e}")
            return None
        if response.status_code == 200:
            topics = response.json()
            return topics
        else:
            # print, what went wrong
            print(f"Error: {response.status_code}")
            return None
        
    def close(self):
        if self.websocket:
            self.websocket.close()
            self.websocket = None

    async def send(self, json_message):
        if self.websocket:
            # translate the json object to a string
            message = json.dumps(json_message)
            print(f"Sending message: {len(message)}")
            await self.websocket.send(message)
            print("Message sent.")
        else:
            print("Error: No websocket connection.")
