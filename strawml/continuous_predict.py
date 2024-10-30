from __init__ import *
from strawml.models.chute_finder.video_stream import VideoStreamCustom
import cv2
video_stream = VideoStreamCustom(model_name="models/yolov11n-obb-test.pt", object_detect=True, yolo_threshold=0.3)

# Run the model on the live stream and display the results
cap = cv2.VideoCapture(0)
video_stream(cap)