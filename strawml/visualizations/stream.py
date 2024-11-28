from __init__ import *
import cv2
import numpy as np
from typing import Tuple, Optional, Any
import pupil_apriltags
import time
import threading
import queue
import torch
import timm
from torchvision.transforms import v2 as transforms
import h5py
import psutil
from sklearn.linear_model import LinearRegression
import keyboard

from strawml.models.straw_classifier import chute_cropper as cc
from strawml.models.chute_finder.yolo import ObjectDetect
import strawml.models.straw_classifier.cnn_classifier as cnn
import strawml.models.straw_classifier.feature_model as feature_model


class TagGraphWithPositionsCV:
    """
    A class to represent a graph of connected tags with inferred positions for missing tags. This class is designed to be used
    in conjunction with the AprilDetector class to detect AprilTags in a real-time video stream. The class provides methods to
    interpolate the position of a missing tag based on its connected neighbors, account for missing tags, crop the image to the
    size of the chute system, and draw the detected and inferred tags on the image. 

    NOTE This class is highly customized to detect AprilTags in the chute system af meliora bio, and may not be suitable for other
    applications without modification as it has hard-coded tag ids and corner tags.
    """

    def __init__(self, connections, detected_tags):
        """
        Class to represent a graph of connected tags with inferred positions for missing tags.

        Params:
        -------
        connections: 
            List of tuples defining the connected tag ids (e.g., (11, 12), (12, 13))
        detected_tags: 
            List of tuples (center_x, center_y, tag_id) for detected tags
        image: 
            The image to overlay on
        corner_tags: 
            List of tag ids that are corner points
        """
        self.connections = connections
        self.detected_tags = {tag_id: (int(x), int(y)) for (x, y, tag_id) in detected_tags}  # Detected tag positions
        self.inferred_tags = {}  # Store inferred tag positions
        self.corner_tags = set([11, 15, 26, 22])  # Store corner tags
    
    def intersection_point(self, x1, y1, x2, y2, x3, y3):
        """
        Finds the intersection between a line created by two points (x1, y1) and (x2, y2) and a 
        line created by a point (x3, y3) that is perpendicular to the first line.

        Params:
        -------
        x1, y1: int, int
            The x and y coordinates of the first point
        x2, y2: int, int
            The x and y coordinates of the second point
        x3, y3: int, int
            The x and y coordinates of the third point
    
        Returns:
        --------
        Tuple
            The intersection point (x, y)
        """
        # Calculate the slope of the original line passing through (x1, y1) and (x2, y2)
        if x2 == x1:  # Original line is vertical
            slope_original = None
            intercept_original = None
        else:
            slope_original = (y2 - y1) / (x2 - x1)
            intercept_original = y1 - slope_original * x1  # y-intercept of the original line
        
        # Determine the slope and intercept of the orthogonal line through (x3, y3)
        if slope_original is None:  # Original line is vertical, orthogonal line is horizontal
            # The intersection point is at x = x1, y = y3
            intersection_x = x1
            intersection_y = y3
        elif slope_original == 0:  # Original line is horizontal, orthogonal line is vertical
            # The intersection point is at y = y1, x = x3
            intersection_x = x3
            intersection_y = y1
        else:
            # Slope of the orthogonal line
            slope_orthogonal = -1 / slope_original
            intercept_orthogonal = y3 - slope_orthogonal * x3  # y-intercept of the orthogonal line
            
            # Solve for x by setting the two line equations equal
            # m * x + b = m' * x + b'
            intersection_x = (intercept_orthogonal - intercept_original) / (slope_original - slope_orthogonal)
            
            # Substitute x back into one of the line equations to find y
            intersection_y = slope_original * intersection_x + intercept_original
        
        return (intersection_x, intersection_y)


    def interpolate_position(self, tag_id, neighbor_ids, is_corner):
        """
        Interpolate the position of a missing tag based on its connected neighbors.
        For corners, we prioritize direct neighbors (no averaging all around). It also accounts for slanted images.

        Params:
        -------
        tag_id: int
            The tag id of the missing tag
        neighbor_ids: List
            The tag ids of the connected neighbors
        is_corner: bool
            A boolean flag indicating if the missing tag is a corner tag
        
        Returns:
        --------
        Tuple
            The interpolated position (x, y)
        """
        neighbor_positions = [self.detected_tags.get(n) for n in neighbor_ids if n in self.detected_tags]
        
        if not neighbor_positions:
            return None
        
        if is_corner:
            if tag_id == 11:
                try:
                    # draw a line between tag 12+13 and find the intersection with a line drawn from 15 such that it is perpendicular to the line between 12+13
                    x_value, y_value = self.intersection_point(self.detected_tags[12][0], 
                                                               self.detected_tags[12][1], 
                                                               self.detected_tags[13][0], 
                                                               self.detected_tags[13][1], 
                                                               self.detected_tags[15][0], 
                                                               self.detected_tags[15][1])
                except KeyError:
                    return None
            if tag_id == 15:
                try:
                    x_value, y_value = self.intersection_point(self.detected_tags[16][0], 
                                                               self.detected_tags[16][1], 
                                                               self.detected_tags[17][0], 
                                                               self.detected_tags[17][1], 
                                                               self.detected_tags[11][0], 
                                                               self.detected_tags[11][1])
                except KeyError:
                    return None
            if tag_id == 22:
                try:
                    x_value, y_value = self.intersection_point(self.detected_tags[20][0], 
                                                               self.detected_tags[20][1], 
                                                               self.detected_tags[21][0], 
                                                               self.detected_tags[21][1], 
                                                               self.detected_tags[26][0], 
                                                               self.detected_tags[26][1])
                except KeyError:
                    return None
            if tag_id == 26:
                try:
                    x_value, y_value = self.intersection_point(self.detected_tags[24][0], 
                                                               self.detected_tags[24][1], 
                                                               self.detected_tags[25][0], 
                                                               self.detected_tags[25][1], 
                                                               self.detected_tags[22][0], 
                                                               self.detected_tags[22][1])
                except KeyError:
                    return None
            return (int(x_value), int(y_value))
        else:
            # For edge tags, we average the position
            avg_x = np.mean([pos[0] for pos in neighbor_positions])
            avg_y = np.mean([pos[1] for pos in neighbor_positions])
        
            return (int(avg_x), int(avg_y))

    def account_for_missing_tags(self):
        """
        Infer the missing tags by interpolating positions based on neighboring connected tags.

        Returns:
        --------
        None
            The detected tags are updated with the inferred positions
        """
        for tag1, tag2 in self.connections:
            # Infer missing tags using neighboring connections
            if tag1 not in self.detected_tags:
                neighbors = [n2 if n1 == tag1 else n1 for n1, n2 in self.connections if tag1 in (n1, n2)]
                is_corner = tag1 in self.corner_tags
                inferred_position = self.interpolate_position(tag1, neighbors, is_corner)
                if inferred_position:
                    self.detected_tags[tag1] = inferred_position
                    self.inferred_tags[tag1] = inferred_position

            if tag2 not in self.detected_tags:
                neighbors = [n2 if n1 == tag2 else n1 for n1, n2 in self.connections if tag2 in (n1, n2)]
                is_corner = tag2 in self.corner_tags
                inferred_position = self.interpolate_position(tag2, neighbors, is_corner)
                if inferred_position:
                    self.detected_tags[tag2] = inferred_position
                    self.inferred_tags[tag2] = inferred_position

    def crop_to_size(self, image):
        """
        Based on opencv's homography tutorial: https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html

        Seeks to perform persepective corrections on the detected tags in the frame. The tags are assumed to be
        quadrilaterals, and the function uses the four corners of the tags to calculate the homography matrix.

        Params:
        -------
        image: np.ndarray
            The image to be cropped
        
        Returns:
        --------
        np.ndarray
            The cropped image
        """
        # see if tag 11, 15, 22, 26 are detected
        if 11 not in self.detected_tags or 15 not in self.detected_tags or 22 not in self.detected_tags or 26 not in self.detected_tags:
            return None
        
        # now we use the coordinates of the detected tags to crop the image
        x1, y1 = self.detected_tags[11][0], self.detected_tags[11][1]
        x2, y2 = self.detected_tags[15][0], self.detected_tags[15][1]
        x3, y3 = self.detected_tags[22][0], self.detected_tags[22][1]
        x4, y4 = self.detected_tags[26][0], self.detected_tags[26][1]

        pts_src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')

        # Ensure the output is a rectangular
        width = max(abs(x2 - x1), abs(x4 - x3))  # Largest horizontal difference
        height = max(abs(y3 - y1), abs(y4 - y2)) # Largest vertical difference

        # Define the destination points for the square output
        pts_dst = np.array([[0, 0], 
                            [width - 1, 0], 
                            [0, height - 1], 
                            [width - 1, height - 1]]
                            , dtype='float32')
        # Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # Apply the perspective warp to create a square cutout
        cutout = cv2.warpPerspective(image, M, (width, height))
        print(width, height)
        return np.array(cutout)
    
    def draw_overlay(self, image):
        """
        Draws the detected and inferred tags on the image along with the connections.

        """
        for tag1, tag2 in self.connections:
            # Get the positions of the two tags
            pos1 = self.detected_tags.get(tag1)
            pos2 = self.detected_tags.get(tag2)

            if pos1 and pos2:
                # Draw circles at the tag positions
                cv2.circle(image, (int(pos1[0]), int(pos1[1])), 5, (0, 255, 0), -1)  # Green for detected
                cv2.circle(image, (int(pos2[0]), int(pos2[1])), 5, (0, 255, 0), -1)  # Green for detected

                # Draw line between the connected tags
                cv2.line(image, (int(pos1[0]), int(pos1[1])), (int(pos2[0]), int(pos2[1])), (255, 0, 0), 2)

        return image
            
class AprilDetector:
    """
    NOTE This class is highly customized to detect AprilTags in the chute system af meliora bio. 

    AprilTag detector class that uses the pupil_apriltags library to detect AprilTags in a frame. The class is designed
    to be used in conjunction with the RTSPStream class to detect AprilTags in a real-time video stream. The class
    provides methods to detect AprilTags in a frame, draw the detected tags on the frame, and given a predicted straw level
    value performs inverse linear interpolation to get the corresponding pixel value on the frame.
    """
    def __init__(self, detector: pupil_apriltags.bindings.Detector, ids: dict, window: bool=False, od_model_name=None, object_detect=True, yolo_threshold=0.5, device="cuda", frame_shape: tuple = (1440, 2560), with_predictor: bool =False, model_load_path: str = "models/vit_regressor/", regressor: bool = True, predictor_model: str = "vit", edges=True, heatmap=False) -> None:
        self.detector = detector
        self.window = window
        self.ids = ids
        self.model_name = od_model_name
        self.object_detect = object_detect
        self.yolo_threshold = yolo_threshold
        self.device = device
        self.edges = edges
        self.heatmap = heatmap
        if self.object_detect:
            self.OD = ObjectDetect(self.model_name, yolo_threshold=yolo_threshold, device=device, verbose=False)

        # Initialize a queue to store the frames from the video stream
        self.q = queue.Queue()
        # Initialize the tags and tag_ids arrays to store the detected tags and their corresponding IDs
        self.tags = np.array([])
        self.tag_ids = np.array([])
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_params = self.load_camera_params()
        self.tag_connections = [(11, 12), (12, 13), (13, 14), (14, 19), (19, 20), (20, 21), 
                                (21, 22), (22, 26), (26, 25), (25, 24), (24, 23), (23, 18), 
                                (18, 17), (17, 16), (16, 15), (15, 11)]
        self.processed_tags = set()  # Track tags that have already been re-centered
        self.detected_tags = []  # Store all detected tags during each frame
        self.with_predictor = with_predictor
        self.model = None
        if with_predictor:
            self.mean, self.std = self.load_normalisation_constants()
            self.img_unnormalize = transforms.Normalize(mean = [-m/s for m, s in zip(self.mean, self.std)],
                                                        std = [1/s for s in self.std])
            self.transform = transforms.Compose([transforms.ToImage(), 
                                                 transforms.ToDtype(torch.float32, scale=True),
                                                 transforms.Normalize(mean=self.mean, std=self.std)])
            num_classes = 1 if regressor else 11
            input_channels = 3
            if edges: input_channels += 1
            if heatmap: input_channels += 3
            image_size = (384,384)
            print(f"Loading model {predictor_model} from {model_load_path} and {num_classes} classes")
            match predictor_model:
                case 'cnn':
                    image_size = (384, 384)
                    self.model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels, output_size=num_classes)
                case 'convnextv2':
                    image_size = (224, 224)
                    self.model = timm.create_model('convnextv2_atto', pretrained=False, in_chans=input_channels, num_classes=num_classes)
                case 'vit':
                    image_size = (384, 384)
                    self.model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', pretrained=False, in_chans=input_channels, num_classes=num_classes)
                case 'eva02':
                    image_size = (448, 448)
                    self.model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=False, in_chans=input_channels, num_classes=num_classes)
                case 'caformer':
                    image_size = (384, 384)
                    self.model = timm.create_model('caformer_m36.sail_in22k_ft_in1k_384', in_chans=input_channels, num_classes=num_classes, pretrained=False)
            
            self.resize = transforms.Resize(image_size)
            
            if regressor:
                if predictor_model != 'cnn':
                    features = self.model.forward_features(torch.randn(1, input_channels, image_size[0], image_size[1]))
                    feature_size = torch.flatten(features, 1).shape[1]
                    self.regressor_model = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1)
                    
                    self.model.load_state_dict(torch.load(f'{model_load_path}/{predictor_model}_feature_extractor_overall_best.pth', weights_only=True))
                    self.regressor_model.load_state_dict(torch.load(f'{model_load_path}/{predictor_model}_regressor_overall_best.pth', weights_only=True))
                    self.regressor_model.to(self.device)
                else:
                    self.model.load_state_dict(torch.load(model_load_path, weights_only=True))
            else:
                self.regressor_model = None
                self.model.load_state_dict(torch.load(model_load_path, weights_only=True))
            self.model.to(self.device)

    def load_normalisation_constants(self):
        # Loads the normalisation constants from 
        # TODO: Figure out where to save this file
        with h5py.File("data/processed/augmented/chute_detection.hdf5", 'r') as f:
            mean = f.attrs['mean']
            std = f.attrs['std']
        return mean, std
        
    def load_camera_params(self):
        # open npz file
        with np.load("fiducial_marker/calibration.npz") as data:
            cameraMatrix = data['cameraMatrix']
            distCoeffs = data['distCoeffs']
            rvecs = data['rvecs']
            tvecs = data['tvecs']
        # load an image to get shape 
        self.fx = cameraMatrix[0, 0]
        self.fy = cameraMatrix[1, 1]
        self.cx = cameraMatrix[0, 2]
        self.cy = cameraMatrix[1, 2]
        return {"cameraMatrix": cameraMatrix, "distCoeffs": distCoeffs, "rvecs": rvecs, "tvecs": tvecs}
    
    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        # First check if the results are empty
        if not results:
            return frame
        if 'obb' in self.model_name:
            labels, cord, labels_conf, angle_rad = results
        else:
            labels, cord, labels_conf = results
        n = len(labels)

        for i in range(n):  
            # plot polygon around the object based on the coordinates cord
            if 'obb' in self.model_name:
                x1, y1, x2, y2, x3, y3, x4, y4 = cord[i].flatten()
                x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
                # draw lines between the corners
                cv2.line(frame, (x1, y1), (x2, y2), (138,43,226), 2)
                cv2.line(frame, (x2, y2), (x3, y3), (138,43,226), 2)
                cv2.line(frame, (x3, y3), (x4, y4), (138,43,226), 2)
                cv2.line(frame, (x4, y4), (x1, y1), (138,43,226), 2)
                # plot label on the object
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                x1, y1, x2, y2 = cord[i].flatten()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # plot label on the object
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.OD.classes[int(x)]
       
    def fix_frame(self, frame: np.ndarray, blur: bool = False, balance: int = 1) -> np.ndarray:
        """
        Fix the frame by undistorting it using the camera parameters.

        Params:
        -------
        frame: np.ndarray
            The frame to be undistorted
        blur: bool
            A boolean flag to indicate if the frame is to be blurred
        balance: int
            The balance factor to be used for undistortion
        
        Returns:
        --------
        np.ndarray
            The undistorted frame
        """
        if blur:
            image = cv2.GaussianBlur(frame, (7, 7), 0.7) 
            return image.astype(np.uint8)
        K, D = self.camera_params["cameraMatrix"], self.camera_params["distCoeffs"]
        h,  w = frame.shape[:2]
        new_K = K.copy()
        new_K[0,0] *= balance  # Scale fx
        new_K[1,1] *= balance  # Scale fy 
        undistorted_image = cv2.fisheye.undistortImage(frame, K, D, Knew=new_K,new_size=(w,h))
        return undistorted_image  
       
    def detect(self, frame: np.ndarray) -> list:
        """
        Detects AprilTags in a frame, and performs further detection centered on each tag found.
        Ensures that each tag is only re-centered once and avoids duplicates.
        
        Parameters:
        -----------
        frame: np.ndarray
            The frame in which the AprilTags are to be detected.
        
        Returns:
        --------
        list
            A list of detected AprilTags in the frame, with all detections adjusted
            to their original positions within the frame, without duplicates.
        """
        # Initial detection on the full frame
        tags = self.detector.detect(frame) #, estimate_tag_pose=True, 
                                    #camera_params=[self.fx, self.fy, self.cx, self.cy], 
                                    #tag_size=0.05)  # Assuming 5cm tag size

        detected_tags = []  # This will hold tags with original coordinates
        unique_tag_ids = set()  # Set to track unique tags in detected_tags
        for tag in tags:
            if tag.tag_id not in self.tag_ids:  
                # Append to the tags and tag_ids arrays if not already present
                self.tags = np.append(self.tags, tag)
                self.tag_ids = np.append(self.tag_ids, int(tag.tag_id))
            
            # Check if this tag has already been used as a re-centered detection
            if tag.tag_id in self.processed_tags:
                continue  # Skip if we've already re-centered on this tag
            
            # Mark the tag as processed to prevent re-centering on it again
            self.processed_tags.add(tag.tag_id)
            
            # Record the original center of the tag
            original_center_x, original_center_y = int(tag.center[0]), int(tag.center[1])
            
            # Define region around the tag (centered ±100 pixels)
            px_pm = 150  # Pixels per margin
            x_start = max(original_center_x - px_pm, 0)
            x_end = min(original_center_x + px_pm, frame.shape[1])
            y_start = max(original_center_y - px_pm, 0)
            y_end = min(original_center_y + px_pm, frame.shape[0])
            
            # Crop frame to this region
            cropped_frame = frame[y_start:y_end, x_start:x_end]

            # Perform a second detection within the cropped frame
            refined_tags = self.detector.detect(cropped_frame) #, estimate_tag_pose=True,
                                                # camera_params=[self.fx, self.fy, self.cx, self.cy],
                                                # tag_size=0.05)                
            # Adjust each detected tag in the cropped region to original frame coordinates
            for refined_tag in refined_tags:
                # Adjust center coordinates to match original frame
                refined_tag_center_x = refined_tag.center[0] + x_start
                refined_tag_center_y = refined_tag.center[1] + y_start
                refined_tag.center = (refined_tag_center_x, refined_tag_center_y)
                refined_tag.corners[:,0] += x_start
                refined_tag.corners[:,1] += y_start

                if refined_tag.tag_id not in self.tag_ids:  
                    # Append to the tags and tag_ids arrays if not already present
                    self.tags = np.append(self.tags, refined_tag)
                    self.tag_ids = np.append(self.tag_ids, int(refined_tag.tag_id))
                
                # Only add unique tags to detected_tags
                if refined_tag.tag_id not in unique_tag_ids:
                    unique_tag_ids.add(refined_tag.tag_id)
                    detected_tags.append(refined_tag)
        self.check_for_changes(detected_tags)
        self.processed_tags.clear()  # Clear the set of processed tags

    def check_for_changes(self, tags: list) -> None:
        """
        Check if the camera has moved and the chute tags have changed position. If so, reset the tags and tag_ids arrays.

        Params:
        -------
        tags: List
            A list of detected AprilTags in the frame
        
        Returns:
        --------
        None
            Nothing is returned, only if the camera has moved and the chute tags have changed position, the tags and tag_ids
            arrays are reset.
        """
        tag_ids = np.array([int(tag.tag_id) for tag in tags])
        accumulated_error = 0
        for i, t in enumerate(self.tag_ids):
            mask = np.where(tag_ids == t)[0]
            if mask.size == 0:
                continue
            t1 = tags[mask[0]]
            t2 = self.tags[i]
            # calculate absolute distance between the two tags
            accumulated_error += np.linalg.norm(np.array(t1.center) - np.array(t2.center))

        # If the camera moves and the chute tags change position, reset the tags
        if accumulated_error/len(self.tag_ids) > 0.1:
            self.tags = []
            self.tag_ids = []

    def order_corners(self, points, centroid):
        """
        Orders the points of a bounding box (tensor) as: top-right, bottom-right, bottom-left, top-left.
        The points are temporarily rotated to identify their roles, but the output is the original coordinates.
        
        Args:
            points (torch.Tensor): A (4, 2) tensor where each row is (x, y).
        
        Returns:
            torch.Tensor: Reordered points (4, 2) in their original coordinates.
        """
        # make sure the data is torch tensors
        if not torch.is_tensor(points):
            points = torch.tensor(points)
        if not torch.is_tensor(centroid):
            centroid = torch.tensor(centroid)
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

        return ordered_points.numpy()

    def rotate_line(self, point1: Tuple, point2: Tuple, angle: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Rotate the line that goes from point1 to point2 by angle degrees around point1

        Params:
        -------
        point1: Tuple
            The first point 
        point2: Tuple
            The second point
        angle: float
            The angle by which the line is to be rotated in radians
        
        Returns:
        --------
        Tuple
            The rotated line represented by the two points

        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        # Translate the points so that point1 is the origin
        p2 = p2 - p1
        # Rotate the point
        p2 = np.array([p2[0] * np.cos(angle) - p2[1] * np.sin(angle), p2[0] * np.sin(angle) + p2[1] * np.cos(angle)])
        # Translate the points back
        p2 = p2 + p1

        return point1, tuple(p2.astype(int))

    def draw(self, frame: np.ndarray, tags: list, make_cutout: bool = False, straw_level: float = 25, use_cutout=False) -> np.ndarray:
        """
        Draws the detected AprilTags on the frame. The number tags are drawn in yellow and the chute tags are drawn in blue.
        The function also draws a line from the right side of the number tag to the right side of the chute tag closest to it on the y-axis,
        representing the predicted straw level.

        Params:
        -------
        frame: np.ndarray
            The frame on which the tags are to be drawn
        tags: List
            A list of detected AprilTags in the frame
        
        Returns:
        --------
        np.ndarray
            The frame with the detected AprilTags drawn on it
        """
        original_image = frame.copy()
        number_tags = []
        chute_tags = []
        if len(tags) == 0:
            return frame, None
        for t in tags:
            if t.tag_id in self.wanted_tags.values():
                corners = self.order_corners(t.corners, t.center)
                if t.tag_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    number_tags.append(t)
                    cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
                else:
                    chute_tags.append(t)
                    cv2.polylines(frame, [corners.astype(np.int32)], True, (255, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f"{int(t.tag_id)}", tuple(corners[0].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        try:
            chute_right = []
            chute_left = []
            for tag in chute_tags:               
                # find the largest x value
                if tag.tag_id in [11, 12, 13, 14, 19, 20, 21, 22]: 
                    chute_left += [(tag.center[0], tag.center[1], tag.tag_id)]
                else:
                    chute_right += [(tag.center[0], tag.center[1], tag.tag_id)]

            x = np.array(chute_left)[:, 0]
            y = np.array(chute_left)[:, 1]
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)
            slope = model.coef_[0]

            perp_slope = -1 / (slope + 1e-6) # Avoid division by zero
            tag_angle = np.arctan(perp_slope)

            # The logic is as follows:
            # 1. For each number tag, find the center of the chute tag that is closest to it on the y-axis in let and right
            # 2. Draw a line going from the right site of the number tag going horizontaly to x being the x value of the right chute tag plus the difference between the x value of the number tag and the x value of the left chute tag
            for tag in number_tags:
                corners = self.order_corners(tag.corners, tag.center)
                # we first define the top right and bottom right corners based on the coordinates
                top_right = corners[1]
                bottom_right = corners[2]
                level_center_right = (top_right + bottom_right) / 2
                # get the angle of the tag wrt the x-axis for rotation purposes

                # print(tag.tag_id, top_left, top_right, np.rad2deg(tag_angle))
                min_distance_right = float('inf')
                min_distance_left = float('inf')
                closest_left_chute = None
                closest_right_chute = None

                # find the closest tag on the RIGHT side of the chute
                for chute in chute_right:
                    distance = abs(chute[1] - level_center_right[1])
                    if distance < min_distance_right:
                        min_distance_right = distance
                        closest_right_chute = chute

                # find the closest tag on the LEFT side of the chute
                for chute in chute_left:
                    distance = abs(chute[1] - level_center_right[1])
                    if distance < min_distance_left:
                        min_distance_left = distance
                        closest_left_chute = chute

                if closest_right_chute and closest_left_chute:
                    line_begin = (int(level_center_right[0]), int(level_center_right[1]))
                    line_end = (int(closest_right_chute[0] + np.abs(closest_left_chute[0] - level_center_right[0])), int(level_center_right[1]))
                    line_begin, line_end = self.rotate_line(line_begin, line_end, -tag_angle)
                    cv2.line(frame, tuple(line_begin), tuple(line_end), (0, 255, 0), 2)
                    cv2.putText(frame, f"{int(tag.tag_id) * 10}%", (int(closest_right_chute[0] + (closest_left_chute[0] - level_center_right[0]))+35, int(level_center_right[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            if make_cutout:
                # combine the left and right chutes
                chutes = chute_left + chute_right
                tag_graph = TagGraphWithPositionsCV(self.tag_connections, chutes)
                tag_graph.account_for_missing_tags()
                frame = tag_graph.draw_overlay(frame)
                cutout = tag_graph.crop_to_size(original_image)

                if use_cutout:
                    return frame, cutout
                else:
                    return frame, None
        except Exception as e:
            print("ERROR", e)
        return frame, None

    def prepare_for_inference(self, frame, results=None):
        if results is None:
            frame_data = frame
        else:
            # rotate and crop the frame to the chute bbox
            bbox_chute = results[1][0].flatten().cpu().detach().numpy() # x1, y1, x2, y2, x3, y3, x4, y4
            # check that the bbox has 8 values
            if len(bbox_chute) != 8:
                frame_data = frame
            else:
                frame_data, bbox_chute = cc.rotate_and_crop_to_bbox(frame, bbox_chute)
        # get edge features
        if self.edges:
            try:
                edges = cv2.Canny(frame_data, 100, 200)
                edges = edges.reshape(1, edges.shape[0], edges.shape[1])
                edges = torch.from_numpy(edges)/255
                # Visualise the image
                edge_vis = edges.permute(1, 2, 0).numpy()
                edge_vis = cv2.resize(edge_vis, (0,0), fx=0.6, fy=0.6)
                cv2.imshow("edges", edge_vis)
                cv2.waitKey(1)
            except Exception as e:
                print("Error in edge detection", e)
                return None
        # normalise with stats saved
        frame_data = self.transform(torch.from_numpy(frame_data).permute(2, 0, 1).float())

        # Visualise the image
        # vis_frame = self.img_unnormalize(frame_data).permute(1, 2, 0).numpy()
        vis_frame = cv2.cvtColor(frame_data.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
        vis_frame = cv2.resize(vis_frame, (0,0), fx=0.6, fy=0.6)
        cv2.imshow("frame", np.clip(vis_frame, 0, 1))
        cv2.waitKey(1)

        # stack the images together
        cutout_image = np.concatenate((frame_data, edges), axis=0)

        # reshape to 4, 384, 384
        cutout_image = self.resize(torch.from_numpy(cutout_image))

        # cv2.imshow("resized_frame", cutout_image[:3].permute(1,2,0).numpy())
        # cv2.waitKey(1)
        # cv2.imshow("resized_edge", cutout_image[3].numpy())
        # cv2.waitKey(1)
        
        cutout_image = cutout_image.unsqueeze(0)
        return cutout_image

class RTSPStream(AprilDetector):
    """
    Wrapper class to detect AprilTags in a real-time video stream. The class inherits from the AprilDetector class and
    provides methods to receive frames from the video stream, detect AprilTags in the frames, draw the detected tags on
    the frame, and display the frame with the detected tags. The class also provides a method to detect AprilTags in a
    single frame.

    NOTE Threading is necessary here because we are dealing with an RTSP stream.
    """
    def __init__(self, detector, ids, credentials_path, od_model_name=None, object_detect=True, yolo_threshold=0.2, device="cuda", window=True, rtsp=True, make_cutout=False, use_cutout=False, detect_april=False, with_predictor: bool = False, model_load_path: str = "models/vit_regressor/", regressor: bool = True, predictor_model: str = "vit", edges=True, heatmap=False) -> None:
        super().__init__(detector, ids, window, od_model_name, object_detect, yolo_threshold, device, with_predictor=with_predictor, model_load_path=model_load_path, regressor=regressor, predictor_model=predictor_model, edges=edges, heatmap=heatmap)
        self.rtsp = rtsp
        self.wanted_tags = ids
        if rtsp:
            self.cap = self.create_capture(credentials_path)
        else:
            self.cap = None
        self.make_cutout = make_cutout
        self.use_cutout = use_cutout
        self.detect_april = detect_april
        self.regressor = regressor
        self.predictor_model = predictor_model
        self.frame = None
        self.should_abort_immediately = False
        # Theadlock to prevent multiple threads from accessing the queue at the same time
        self.lock = threading.Lock()
        self.threads = []
        self.information = {
            "FPS":              {"text": "", "font_scale": 1,   "font_thicknesss": 2, "position": (10, 50)},
            "straw_level":      {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 100)},
            "undistort_time":   {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 125)},
            "april":            {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 150)},
            "od":               {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 175)},
            "prep":             {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 200)},
            "model":            {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 225)},
            "frame_time":       {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 250)},
            "GPU":              {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 275)},
        }
                            
                # texts = [f"Undistort Time: {undistort_time:.2f} s"]
                # font_scales = [0.5]
                # font_thicknesss = [1]
                # "positions" = [(10, 125)]
        
    def create_capture(self, credentials_path: str) -> cv2.VideoCapture:
        """
        Create a video capture object to receive frames from the RTSP stream.

        Params:
        -------
        params: str
            The path to the file containing the RTSP stream credentials

        Returns:
        --------
        cv2.VideoCapture
            The video capture object to receive frames from the RTSP stream
        """
        with open(credentials_path, 'r') as f:
            credentials = f.read().splitlines()
            username = credentials[0]
            password = credentials[1]
            ip = credentials[2]
            rtsp_port = credentials[3]
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
        return cap
    
    @staticmethod
    def time_function(func, *args, **kwargs):
        """
        Wrapper function to time the execution of any function.
        
        Parameters:
        -----------
        func : function
            The function to be timed.
        *args : tuple
            The positional arguments to pass to the function.
        **kwargs : dict
            The keyword arguments to pass to the function.
        
        Returns:
        --------
        result : any
            The result of the function call.
        elapsed_time : float
            The time taken to execute the function in seconds.
        """
        if isinstance(func, list):
            start_time = time.time()
            # TODO: Only works for non-cnn models right now
            result = func[0].forward_features(*args, **kwargs)
            # if result.shape[0] == 1: result = result.flatten()
            # else: result = result.flatten(1)
            result = func[1](result)
            elapsed_time = time.time() - start_time
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
        return result, elapsed_time

    def receive_frame(self, cap: cv2.VideoCapture) -> None:
        ret, frame = cap.read()
        self.q.put(frame)
        while not self.should_abort_immediately:
            try:
                ret, frame = cap.read()
                if ret:
                    self.q.put(frame)
            except Exception as e:
                print("Error in receiving frame", e)
                break

    def find_tags(self) -> None:
        while not self.should_abort_immediately:
            start = time.time()
            self.lock.acquire()
            try:
                if self.frame is None:
                    continue
                frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            finally:
                self.lock.release()
            self.detect(frame)
            end = time.time()
            self.information["april"]["text"] = f'(T3) AprilTag Time: {end - start:.2f} s' 

    def display_frame(self) -> None:
        """
        Display the frames with the detected AprilTags.
        """
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)  # Blue color for text
        box_color = (255, 255, 255)  # White color for box
        
        # Define variables for FPS calculation
        frame_count = 0
        start_time = time.time()

        while not self.should_abort_immediately:
            frame_time = time.time()
            # empty information
            for key in self.information.keys():
                if key == "april":
                    continue
                self.information[key]["text"] = ""
            if not self.q.empty():
                frame = self.q.get() # Get the frame from the queue
                # Update the frame in the class instance to be used in other methods and ensure thread safety
                if frame is None: # check if the frame is none. If it is, skip the iteration
                    continue
                self.lock.acquire()
                try:
                    self.frame = frame
                finally:
                    self.lock.release()

                if self.rtsp:
                    self.q.queue.clear() # Clear the queue to account for any lag and prevent the queue from getting too large
                
                # # Fix the frame by undistorting it
                # frame, undistort_time = self.time_function(self.fix_frame, frame) # NOTE this cant be used since undistort crops the top and bottom of the chute too much
                # self.information["undistort_time"]["text"] = f'Undistort Time: {undistort_time:.2f} s'

                if self.detect_april and (self.tags is not None):
                    # # Draw the detected AprilTags on the frame and get the cutout from the frame if make_cutout is True
                    frame_drawn, cutout = self.draw(frame=frame.copy(), tags=self.tags, make_cutout=self.make_cutout, use_cutout=self.use_cutout)
                else:
                    frame_drawn = frame
                    cutout = None
                # We initialise results to None to avoid errors when the model is not used -> only when OD is used do we need
                # the results to crop the bbox from the frame. However, with the apriltrags from self.draw, we simply make the 
                # cutout from the frame and do not need the results.
                if cutout is not None:
                    frame = cutout
                    results = None
                elif cutout is None and self.object_detect:
                    results, OD_time = self.time_function(self.OD.score_frame, frame) # This takes a lot of time if ran on CPU
                    if len(results[0]) == 0:
                        results = "NA"
                    self.information["od"]["text"] = f'(T2) OD Time: {OD_time:.2f} s'
                else:
                    raise ValueError("The cutout image is None and the object detection is not used.")

                if not results == "NA":
                    if self.with_predictor:
                        cutout_image, prep_time = self.time_function(self.prepare_for_inference, frame, results)
                        if cutout_image is not None:
                            if self.regressor:
                                if self.predictor_model != 'cnn':
                                    output, inference_time = self.time_function([self.model, self.regressor_model], cutout_image.to(self.device))
                                else:
                                    output, inference_time = self.time_function(self.model, cutout_image.to(self.device))
                                # detach the output from the device and get the predicted value
                                output = output.detach().cpu()
                                straw_level = output[0].item()*100
                            else:
                                output, inference_time = self.time_function(self.model, cutout_image.to(self.device)) 
                                # detach the output from the device and get the predicted value
                                output = output.detach().cpu()
                                _, predicted = torch.max(output, 1)
                                straw_level = predicted[0]*10

                            # Add the time taken for inference to the text
                            self.information["straw_level"]["text"] = f'(T2) Straw Level: {straw_level:.2f} %'
                            self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
                            self.information["prep"]["text"] = f'(T2) Image Prep. Time: {prep_time:.2f} s'

                    if self.object_detect:
                        frame_drawn = self.plot_boxes(results, frame_drawn)
                else:
                    frame_drawn = cv2.resize(frame_drawn, (0, 0), fx=0.6, fy=0.6) # Resize the frame for display
                    cv2.imshow('Video', frame_drawn) # Display the frame
                    continue
                frame_drawn = cv2.resize(frame_drawn, (0, 0), fx=0.6, fy=0.6) # Resize the frame for display

                frame_count += 1 # Increment frame count
                
                # Calculate FPS
                e = time.time()
                elapsed_time = e - start_time
                fps = frame_count / elapsed_time
                
                # Display the FPS on the frame
                self.information["FPS"]["text"] = f'(T2) FPS: {fps:.2f}'
                self.information["frame_time"]["text"] = f'(T2) Total Frame Time: {e - frame_time:.2f} s'
                self.information["GPU"]["text"] = f'(TM) GPU Usage: {f"Total RAM Usage (GB): {np.round(psutil.virtual_memory().used / 1e9, 2)}"}'
                # Draw the text on the frame
                for i, (key, val) in enumerate(self.information.items()):
                    # Get the text size
                    if val["text"] == "":
                        continue
                    font_scale = val["font_scale"]
                    font_thickness = val["font_thicknesss"]
                    (text_width, text_height), baseline = cv2.getTextSize(val["text"], font, font_scale, font_thickness)
                    pos = val["position"]
                    box_coords = ((pos[0], pos[1] - text_height - baseline), (pos[0] + text_width, pos[1] + baseline)) # Calculate the box coordinates
                    cv2.rectangle(frame_drawn, box_coords[0], box_coords[1], box_color, cv2.FILLED) # Draw the white box                    
                    cv2.putText(frame_drawn, val["text"], pos, font, font_scale, color, font_thickness, cv2.LINE_AA) # Draw the text over the box
                
                cv2.imshow('Video', frame_drawn) # Display the frame

                # flush everything from memory to prevent memory leak
                frame = None
                results = None
                cutout_image = None
                output = None
                torch.cuda.empty_cache()

    def display_frame_from_videofile(self) -> None:
        """
        Display the frames with the detected AprilTags.
        """
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)  # Blue color for text
        box_color = (255, 255, 255)  # White color for box
        
        # Define variables for FPS calculation
        frame_count = 0
        start_time = time.time()

        while True:
            frame_time = time.time()
            success, frame = self.cap.read() # Get the frame from the queue

            if not success: # check if the frame is none. If it is, skip the iteration
                continue
            self.lock.acquire()
            try:
                self.frame = frame
            finally:
                self.lock.release()
            # # Fix the frame by undistorting it
            # frame, undistort_time = self.time_function(self.fix_frame, frame) # NOTE this cant be used since undistort crops the top and bottom of the chute too much
            # self.information["undistort_time"]["text"] = f'Undistort Time: {undistort_time:.2f} s'

            if self.detect_april and (self.tags is not None):
                # # Draw the detected AprilTags on the frame and get the cutout from the frame if make_cutout is True
                frame_drawn, cutout = self.draw(frame.copy(), self.tags, self.make_cutout, self.use_cutout)
            else:
                frame_drawn = frame
                cutout = None
            # We initialise results to None to avoid errors when the model is not used -> only when OD is used do we need
            # the results to crop the bbox from the frame. However, with the apriltrags from self.draw, we simply make the 
            # cutout from the frame and do not need the results.
            results = None
            if cutout is not None:
                frame = cutout
            elif self.object_detect:
                results, OD_time = self.time_function(self.OD.score_frame, frame) # This takes a lot of time if ran on CPU
                if len(results[0]) == 0:
                    results = "NA"
                self.information["od"]["text"] = f'(T2) OD Time: {OD_time:.2f} s'
            else:
                raise ValueError("The cutout image is None and the object detection is not used.")

            if not results == "NA":
                if self.with_predictor:
                    cutout_image, prep_time = self.time_function(self.prepare_for_inference, frame, results)
                    if cutout_image is not None:
                        if self.regressor:
                            if self.predictor_model != 'cnn':
                                output, inference_time = self.time_function([self.model, self.regressor_model], cutout_image.to(self.device))
                            else:
                                output, inference_time = self.time_function(self.model, cutout_image.to(self.device))
                            # detach the output from the device and get the predicted value
                            output = output.detach().cpu()
                            straw_level = output[0].item()*100
                        else:
                            output, inference_time = self.time_function(self.model, cutout_image.to(self.device)) 
                            # detach the output from the device and get the predicted value
                            output = output.detach().cpu()
                            _, predicted = torch.max(output, 1)
                            straw_level = predicted[0]*10

                        # Add the time taken for inference to the text
                        self.information["straw_level"]["text"] = f'(T2) Straw Level: {straw_level:.2f} %'
                        self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
                        self.information["prep"]["text"] = f'(T2) Image Prep. Time: {prep_time:.2f} s'

                if self.object_detect:
                    frame_drawn = self.plot_boxes(results, frame_drawn)
            else:
                frame_drawn = cv2.resize(frame_drawn, (0, 0), fx=0.6, fy=0.6) # Resize the frame for display
                cv2.imshow('Video', frame_drawn) # Display the frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.close_threads()
                    break
                continue
            frame_drawn = cv2.resize(frame_drawn, (0, 0), fx=0.6, fy=0.6) # Resize the frame for display

            frame_count += 1 # Increment frame count
            
            # Calculate FPS
            e = time.time()
            elapsed_time = e - start_time
            fps = frame_count / elapsed_time
            
            # Display the FPS on the frame
            self.information["FPS"]["text"] = f'(T2) FPS: {fps:.2f}'
            self.information["frame_time"]["text"] = f'(T2) Total Frame Time: {e - frame_time:.2f} s'
            self.information["GPU"]["text"] = f'(TM) GPU Usage: {f"Total RAM Usage (GB): {np.round(psutil.virtual_memory().used / 1e9, 2)}"}'
            # Draw the text on the frame
            for i, (key, val) in enumerate(self.information.items()):
                # Get the text size
                if val["text"] == "":
                    continue
                font_scale = val["font_scale"]
                font_thickness = val["font_thicknesss"]
                (text_width, text_height), baseline = cv2.getTextSize(val["text"], font, font_scale, font_thickness)
                pos = val["position"]
                box_coords = ((pos[0], pos[1] - text_height - baseline), (pos[0] + text_width, pos[1] + baseline)) # Calculate the box coordinates
                cv2.rectangle(frame_drawn, box_coords[0], box_coords[1], box_color, cv2.FILLED) # Draw the white box                    
                cv2.putText(frame_drawn, val["text"], pos, font, font_scale, color, font_thickness, cv2.LINE_AA) # Draw the text over the box
            
            cv2.imshow('Video', frame_drawn) # Display the frame
            # flush everything from memory to prevent memory leak
            frame = None
            gray = None
            results = None
            cutout_image = None
            output = None
            torch.cuda.empty_cache()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close_threads()
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def close_threads(self):
        print("END: Threads and resources...")
        self.lock.acquire()
        self.should_abort_immediately = True
        self.lock.release()
        for thread in self.threads:
            thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        print("EXIT: Stream has been terminated...")

    def __call__(self, frame: Optional[np.ndarray] = None, cap: Optional[cv2.VideoCapture] = None, video_path: str=None) -> None | list:
        """
        Upon calling the object, if self.window is True and frame is None, the frames are received from the video stream
        and displayed with the detected AprilTags. If frame is not None, the detected AprilTags are drawn on the frame.
        If a cap object is passed, the threads are redefined with the new cap.

        Params:
        -------
        frame: np.ndarray
            The frame in which the AprilTags are to be detected
        cap: cv2.VideoCapture
            The video capture object to receive frames from the RTSP stream

        Returns:
        --------
        None
            Nothing is returned, only the frames are displayed with the detected AprilTags
        """
        if frame is None:
            if video_path:
                self.cap = cv2.VideoCapture(video_path)
                print("START: Videofile loaded")
                self.display_frame_from_videofile()
            else:
                if cap is not None:
                    self.cap = cap
                print("START: Threads and resources...")
                self.thread1 = threading.Thread(target=self.receive_frame, args=(self.cap,))
                self.thread2 = threading.Thread(target=self.display_frame)
                self.thread1.start()
                self.thread2.start()
                self.threads += [self.thread1, self.thread2]
                if self.detect_april:
                    self.thread3 = threading.Thread(target=self.find_tags)
                    self.thread3.start()
                    self.threads.append(self.thread3)
                while True:
                    if keyboard.is_pressed('q'):
                        self.close_threads()
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.close_threads()
                        break
        elif frame is not None:
            # resize frame half the size
            frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # get the four corner corrdinates of the frame
            cutout_coord = np.array([0, 0, frame.shape[1], frame.shape[0]])
            tags = self.detect(gray, cutout_coord)
            if self.window:
                frame = self.draw(frame, tags)
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
            return tags

if __name__ == "__main__":
    from pupil_apriltags import Detector
    import json
    with open("fiducial_marker/april_config.json", "r") as file:
        config = json.load(file)
    detector = Detector(
        families=config["dict_type"],
        nthreads=config["nthreads"],
        quad_decimate=config["quad_decimate"],
        quad_sigma=config["quad_sigma"],
        refine_edges=config["refine_edges"],
        decode_sharpening=config["decode_sharpening"],
        debug=config["debug"]
    )

    # video_path = "data/raw/stream-2024-09-23-10h11m28s.mp4"

    # RTSPStream(detector, config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
    #            rtsp=False, # Only used when the stream is from an RTSP source
    #            make_cutout=False, object_detect=True, od_model_name="models/yolov11_obb_m8100btb_best.pt", yolo_threshold=0.2,
    #            detect_april=False,
    #            with_predictor=True, predictor_model='vit', model_load_path='models/vit_regressor/', regressor=True, edges=True, heatmap=False)(video_path=video_path)
    
    RTSPStream(detector, config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
            rtsp=True , # Only used when the stream is from an RTSP source
            make_cutout=True, use_cutout=True, object_detect=True, od_model_name="models/yolov11_obb_m8100btb_best.pt", yolo_threshold=0.2,
            detect_april=True,
            with_predictor=False, predictor_model='vit', model_load_path='models/vit_regressor/', regressor=True, edges=True, heatmap=False)()