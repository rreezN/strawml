from __init__ import *
import cv2
from strawml.models.chute_finder.video_stream import VideoStreamCustom

# NOTE Ensure that you are connected to the VPN
with open('data/hkvision_credentials.txt', 'r') as f:
    credentials = f.read().splitlines()
    username = credentials[0]
    password = credentials[1]
    ip = credentials[2]
    rtsp_port = credentials[3]

# objectdetect = ObjectDetect('models\yolov8n-obb-chute.pt', 0.8)
video_stream = VideoStreamCustom(model_name="models/yolov8n-obb-chute-ext-adam.pt", object_detect=True, yolo_threshold=0.5, verbose=False)

cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_FPS, 25)
cap.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
video_stream(cap)

