from __init__ import *
from strawml.models.chute_finder.video_stream import VideoStreamCustom

video_stream = VideoStreamCustom(model_name="models/yolov8n-obb.pt", object_detect=True, yolo_threshold=0.3)

# Run the model on the live stream and display the results
video_stream(0)