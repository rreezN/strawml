from __init__ import *
import torch
from ultralytics import YOLO

class ObjectDetect:
    def __init__(self, model_name, yolo_threshold=0.5, device="cuda", verbose=False):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.yolo_threshold = yolo_threshold
        self.verbose = verbose
        
    def load_model(self, model_name):
        """
        Loads a local .pt model file and returns the model.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = YOLO(model_name)
        else:
            model = YOLO("models/yolov8n-obb-chute.pt")
            
        model.to(self.device)

        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        results = self.model(frame, verbose=self.verbose)
        # check if obb is in the model_name string
        if 'obb' in self.model_name:
            conf = results[0].obb.conf
            labels = results[0].obb.cls
            coords = results[0].obb.xyxyxyxy
            mask = conf >= self.yolo_threshold
            angle = results[0].obb.xywhr[:, 4]
            return labels[mask], coords[mask], conf[mask], angle[mask]
        else:
            conf = results[0].boxes.conf
            labels = results[0].boxes.cls
            coords = results[0].boxes.xyxy
            mask = conf >= self.yolo_threshold
            return labels[mask], coords[mask], conf[mask]
