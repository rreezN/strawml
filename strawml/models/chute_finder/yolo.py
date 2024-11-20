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

    def order_corners(self, points):
        """
        Orders the points of a bounding box (tensor) as: top-right, bottom-right, bottom-left, top-left.
        The points are temporarily rotated to identify their roles, but the output is the original coordinates.
        
        Args:
            points (torch.Tensor): A (4, 2) tensor where each row is (x, y).
        
        Returns:
            torch.Tensor: Reordered points (4, 2) in their original coordinates.
        """
        # Compute the centroid
        centroid = points.mean(dim=0)

        # Compute the angle of rotation based on the first edge (assume points[0] and points[1])
        p1, p2 = points[0], points[1]
        delta_x, delta_y = p2[0] - p1[0], p2[1] - p1[1]
        angle = torch.atan2(delta_y, delta_x)  # Angle in radians

        # Rotation matrix for -angle (to align the bbox with axes)
        cos_a, sin_a = torch.cos(-angle), torch.sin(-angle)
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])

        # Rotate all points
        rotated_points = (points - centroid) @ rotation_matrix.T

        # Identify the points in rotated space
        # Top-right: Largest x, smallest y
        # Bottom-right: Largest x, largest y
        # Bottom-left: Smallest x, largest y
        # Top-left: Smallest x, smallest y
        top_right_idx = torch.argmin(rotated_points[:, 1] - rotated_points[:, 0])
        bottom_right_idx = torch.argmax(rotated_points[:, 0] + rotated_points[:, 1])
        bottom_left_idx = torch.argmax(-rotated_points[:, 0] + rotated_points[:, 1])
        top_left_idx = torch.argmin(rotated_points[:, 1] + rotated_points[:, 0])

        # Collect the points in desired order using original coordinates
        ordered_points = points[torch.tensor([top_right_idx, bottom_right_idx, bottom_left_idx, top_left_idx])]

        return ordered_points

    # def order_corners(self, corners: torch.Tensor) -> torch.Tensor:
    #     """
    #     The purpose of this is to have the corners in the following order:
    #     top_left, top_right, bottom_right, bottom_left

    #     and we do this based on the coordiantes, knowing that topleft of the image is (0, 0)
    #     and bottom right is (width, height).

    #     Params:
    #     -------
    #     corners: np.ndarray
    #         The corners of the detected AprilTag
            
    #     Returns:
    #     --------
    #     np.ndarray
    #         The ordered corners of the detected AprilTag
    #     """
    #     center = (np.mean(corners[:, 0].cpu().numpy()), np.mean(corners[:, 1].cpu().numpy()))
    #     c = {}
    #     for i, corner in enumerate(corners):
    #         x, y = corner
    #         if x > center[0] and y < center[1]:
    #             c[0] = corner.cpu()  # top_right
    #         elif x > center[0] and y > center[1]:
    #             c[1] = corner.cpu()  # bottom_right
    #         elif x < center[0] and y > center[1]:
    #             c[2] = corner.cpu()  # bottom_left
    #         elif x < center[0] and y < center[1]:
    #             c[3] = corner.cpu()  # top_left
    #     if len(c) != 4:
    #         raise ValueError("The corners must have 4 points with the center in the middle")
    #     return torch.from_numpy(np.array(list(dict(sorted(c.items())).values())))
    
    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        results = self.model(frame, verbose=self.verbose)
        # check if obb is in the model_name string
        # try:
        if 'obb' in self.model_name:
            conf = results[0].obb.conf
            labels = results[0].obb.cls
            coords = results[0].obb.xyxyxyxy
            mask = conf >= self.yolo_threshold
            angle = results[0].obb.xywhr[:, 4]
            # order the corners
            coords_ = [self.order_corners(c.detach().cpu()).to(self.device) for c in coords[mask]]
            return labels[mask], coords_, conf[mask], angle[mask]
        else:
            conf = results[0].boxes.conf
            labels = results[0].boxes.cls
            coords = results[0].boxes.xyxy
            mask = conf >= self.yolo_threshold
            return labels[mask], coords[mask], conf[mask]
        # except Exception as e:
        #     print(e)
        #     return "NA"