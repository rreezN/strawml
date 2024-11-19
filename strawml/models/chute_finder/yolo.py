from __init__ import *
import torch
from ultralytics import YOLO
import numpy as np

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
    
    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        The purpose of this is to have the corners in the following order:
        top_left, top_right, bottom_right, bottom_left

        and we do this based on the coordiantes, knowing that topleft of the image is (0, 0)
        and bottom right is (width, height).

        Params:
        -------
        corners: np.ndarray
            The corners of the detected AprilTag
            
        Returns:
        --------
        np.ndarray
            The ordered corners of the detected AprilTag
        """
        center = (np.mean(corners[:, 0].cpu().numpy()), np.mean(corners[:, 1].cpu().numpy()))
        c = {}
        for i, corner in enumerate(corners):
            x, y = corner
            if x > center[0] and y < center[1]:
                c[0] = corner.cpu()  # top_right
            elif x > center[0] and y > center[1]:
                c[1] = corner.cpu()  # bottom_right
            elif x < center[0] and y > center[1]:
                c[2] = corner.cpu()  # bottom_left
            elif x < center[0] and y < center[1]:
                c[3] = corner.cpu()  # top_left
        if len(c) != 4:
            raise ValueError("The corners must have 4 points with the center in the middle")
        return torch.from_numpy(np.array(list(dict(sorted(c.items())).values())))
    
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
            # order the corners
            coords_ = [self.order_corners(c) for c in coords[mask]]
            return labels[mask], coords_, conf[mask], angle[mask]
        else:
            conf = results[0].boxes.conf
            labels = results[0].boxes.cls
            coords = results[0].boxes.xyxy
            mask = conf >= self.yolo_threshold
            return labels[mask], coords[mask], conf[mask]
