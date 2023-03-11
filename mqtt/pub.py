import paho.mqtt.client as mqtt
from videocaptureasync import VideoCaptureAsync
import cv2
import time
import json
import sys
import base64

topic = sys.argv[1]
client = None

rtsp_addr = 'rtsp://admin:a123456789@59.8.116.237:555/Streaming/Channels/101/?tranportmode=multicast'

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)


def on_disconnect(client, userdata, rc):
    print(str(rc))
    client.loop_stop()

    # 다시 연결
    time.sleep(1)
    mq_start()


def on_publish(client, userdata, mid):
    print("In on_pub callback mid= ", mid)


def send_image(img_data, topic, client):
    jpg_img = cv2.imencode('.jpg', img_data)  # [0]: retval(압축 결과), [1]: buf(인코딩된 이미지)
    img_base64 = base64.b64encode(jpg_img[1]).decode('utf-8')

    # img = cv2.imread('test/triplet.jpg')
    # jpg_img = cv2.imencode('.jpg', img)
    # img_base64 = base64.b64encode(jpg_img[1]).decode('utf-8')

    packet_size = 10000
    # packet_size = 1000
    num_packets = len(img_base64) // packet_size + 1
    for i in range(num_packets):
        packet_start = i * packet_size
        packet_end = min((i+1) * packet_size, len(img_base64))
        packet_data = img_base64[packet_start:packet_end]

        client.publish(topic, packet_data)
    print(num_packets)


def mq_start():
    global client
    client = mqtt.Client()

    cap = VideoCaptureAsync(rtsp_addr)
    cap.start()
    
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish

    client.connect('10.10.10.12', 1883)

    client.loop_start()

    while(True):

        # -----------text------------
        # client.publish(topic, 'test', 1)
        # time.sleep(3)

        # -----------img(full)------------
        ret, frame = cap.read()

        # ret == False면 capture 객체 다시 생성
        if ret == False:
            print('ret == False')
            # client.disconnect()  # on_disconnect 호출
            # break

            # time.sleep(3)
            # continue
            cap = VideoCaptureAsync(rtsp_addr)
            cap.start()
            ret, frame = cap.read()

        else:
            _, mqtt_msg = cv2.imencode(".jpg", frame)

            # frame_list = frame.tolist()
            # mqtt_msg = json.dumps(frame_list)

            client.publish(topic, mqtt_msg.tobytes(), 2)  # topic, payload, qos
            # client.publish(topic, '테스트', 1)
            time.sleep(3)

        # -----------img(packet)------------
        # ret, frame = cap.read()

        # # ret == False면 capture 객체 다시 생성
        # if ret == False:
        #     cap = VideoCaptureAsync(rtsp_addr)
        #     cap.start()
        #     ret, frame = cap.read()

        # else:
        #     send_image(frame, topic, client)  # type(frame): np.ndarray
        #     time.sleep(3)


# main
mq_start()
