import json
import os

# in notebook
try:
    from IPython.display import Image, display
except:
    print("Not in notebook")

class HiveAgent():
    """
    Use the Hive services to complete a task.
    The agent takes text, audio and image as parameters.
    If the text is a json then we will assume it comes from a device.
    In this case the json should contains the text, audio and image extracted informations.
    ```
    {
        text: from asr,
        language: from asr,
        voices: from voice_reco,
        faces: from face_reco
    }
    ```
    """
    def __init__(self, hive, name="HiveAgent"):
        self.hive = hive
        self.name = name
        self.buffer = []


    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        # client.subscribe("$SYS/#")


    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        # print(msg.topic) # +" "+str(msg.payload))
        payload = json.loads(msg.payload.decode('utf8').replace("'", '"'))
        if msg.topic in ["samples"]:
            if payload.get("data"):
                buffer.extend(payload.get("data", []))
                print("{} buffer size {}".format(counter, len(buffer)))
                if payload.get("last"):
                    print("create file and start inference")
                    self.hive.get_client("mqtt").publish("asr_result", "{}")
                    # then reset the buffer
                    self.buffer = []


    def run(self):
        self.hive.start_mqtt(on_connect=self.on_connect, on_message=self.on_message)
        self.hive.get_client("mqtt").subscribe("samples", 0)
        self.hive.get_client("mqtt").run_forever()
