import paho.mqtt.client as mqtt
import numpy as np
import cv2
import os
import time
import json
import base64
import io
import sys
from PIL import Image
import logging


start_time = None
client = None
# image_data_buffer = []

logging.basicConfig(level=logging.DEBUG)
handler = logging.FileHandler('mqtt_client.log')
logger = logging.getLogger()
logger.addHandler(handler)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)


def on_disconnect(client, userdata, rc):
    print(str(rc))
    client.loop_stop()
    # client.disconnect()

    # 다시 연결
    time.sleep(1)
    mq_start()


def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribed: " + str(mid) + " " + str(granted_qos))


def on_message(client, userdata, msg):
    # -----------text------------
    # print(msg.topic, " : ", str(msg.payload.decode("utf-8")))


    # -----------img(full)------------
    mqtt_msg = np.frombuffer(msg.payload, dtype=np.uint8)
    cam = cv2.imdecode(mqtt_msg, cv2.IMREAD_COLOR)

    # mqtt_msg = json.loads(msg.payload)
    # cam = np.array(mqtt_msg)

    # encoded_img = np.fromstring(msg.payload, dtype = np.uint8)
    # cam = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    save_path = msg.topic
    os.makedirs(save_path, exist_ok=True)
    save_filename = f'triplet.jpg'
    full_path = os.path.join(save_path, save_filename)

    cv2.imwrite(full_path, cam)
    print(msg.topic, ': img saved')


    # -----------img(packet)------------
    # global image_data_buffer

    # packet_data = base64.b64decode(msg.payload)

    # image_data_buffer.append(packet_data)

    # num_packets = 73  # test
    # if len(image_data_buffer) == num_packets:
    #     # imgdata = base64.b64decode(b"".join(image_data_buffer))
    #     # dataBytesIO = io.BytesIO(imgdata)
    #     # image = Image.open(dataBytesIO)
    #     # image_data = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    #     image_base64 = b"".join(image_data_buffer)
    #     image_data = base64.b64decode(image_base64)

    #     with open("triplet_packet_test.jpg", "wb") as image_file:
    #         image_file.write(image_data)
    #     print(msg.topic, ': img saved')

        
    #     image_data_buffer = []


def mq_start():
    global client
    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe = on_subscribe
    client.on_message = on_message

    client.connect('10.10.10.12', 1883)

    client.subscribe('pub_1', 2)  # topic, qos
    client.subscribe('pub_2', 2)
    client.subscribe('pub_3', 2)

    # client.loop_start()
    client.loop_forever()


# main
mq_start()
