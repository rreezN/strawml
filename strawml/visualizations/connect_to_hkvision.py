from __init__ import *
import numpy as np
import cv2
from ultralytics import YOLO
from strawml.models.chute_finder.video_stream import VideoStreamCustom

# NOTE Ensure that you are connected to the VPN
with open('data/hkvision_credentials.txt', 'r') as f:
    credentials = f.read().splitlines()
    username = credentials[0]
    password = credentials[1]
    ip = credentials[2]
    rtsp_port = credentials[3]

# objectdetect = ObjectDetect('models\yolov8n-obb-chute.pt', 0.8)
video_stream = VideoStreamCustom(model_name="models/yolov8n-obb-chute.pt", object_detect=True, yolo_threshold=0.2)

# Run the model on the live stream and display the results
cap = cv2.VideoCapture()
cap.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
video_stream(cap)