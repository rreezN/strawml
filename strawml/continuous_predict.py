from __init__ import *
from strawml.models.chute_finder.video_stream import VideoStreamCustom
import cv2
# video_stream = VideoStreamCustom(model_name="models/yolov11n-obb-test.pt", object_detect=True, yolo_threshold=0.3)

# # Run the model on the live stream and display the results
with open('data/hkvision_credentials.txt', 'r') as f:
    credentials = f.read().splitlines()
    username = credentials[0]
    password = credentials[1]
    ip = credentials[2]
    rtsp_port = credentials[3]
cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_FPS, 25)
cap.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
# video_stream(cap)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break