from __init__ import *
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Any
import torch
from strawml.models.straw_classifier import chute_cropper as cc
import yaml
import threading
import asyncio
from asyncua import Client
import copy
import time
from strawml.data.image_utils import SpecialRotate


class AprilDetectorHelpers:
    def __init__(self, april_detector_instance):
        self.ADI = april_detector_instance  # Store the main class instance (AprilDetectorInstances - ADI)

    def _initialize_information_dict(self) -> dict:
        temp = {
            "FPS":              {"text": "", "font_scale": 1,   "font_thicknesss": 2, "position": (10, 40)},
            "scada_level":      {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 75)},
            "straw_level":      {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 100)},
            "undistort_time":   {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 125)},
            "april":            {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 150)},
            "od":               {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 175)},
            "prep":             {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 200)},
            "model":            {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 225)},
            "frame_time":       {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 250)},
            "RAM":              {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 275)},
            "CPU":              {"text": "", "font_scale": 0.5, "font_thicknesss": 1, "position": (10, 300)},

        }
        return temp

    def _order_corners(self, points, centroid):
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

    def _classify_tags(self, tags: list) -> Tuple[list, list]:
        """Classify detected tags into number tags and chute tags."""
        number_tags = []
        chute_tags = []
        for tag in tags.values():
            if tag.tag_id in self.ADI.wanted_tags.values():
                if tag.tag_id in range(11):  # IDs 0-10 are number tags
                    number_tags.append(tag)
                else:  # IDs 11+ are chute tags
                    chute_tags.append(tag)
        return number_tags, chute_tags

    def _draw_tags(self, frame: np.ndarray, number_tags: list, chute_tags: list):
        """Draws number and chute tags on the frame."""
        for tag in number_tags + chute_tags:
            corners = self._order_corners(tag.corners, tag.center)
            color = (0, 255, 255) if tag in number_tags else (255, 0, 0)
            cv2.polylines(frame, [corners.astype(np.int32)], True, color, 4, cv2.LINE_AA)
            cv2.putText(frame, str(tag.tag_id), tuple(corners[0].astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return frame
    
    def _get_tag_angle(self, chute_tags: list) -> float:
        # extract all the centers for the tags in chute_tags
        # first see if tags is a list of tuples or a list of objects
        if len(chute_tags) == 0:
            return 0
        if isinstance(chute_tags[0], tuple) or isinstance(chute_tags[0], np.ndarray):
            tags = np.array(chute_tags).reshape(-1, 2)
        else:
            tags = np.array([tag.center for tag in chute_tags]).reshape(-1, 2)
        x = tags[:, 0]
        y = tags[:, 1]
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        slope = model.coef_[0]
        perp_slope = -1 / (slope + 1e-6) # Avoid division by zero
        return np.arctan(perp_slope)

    def _draw_level_lines(self, frame: np.ndarray, number_tags: list, chute_tags: list, straw_level: float):
        """Draws lines between number tags and chute tags indicating straw levels."""
        chute_right = [chute for chute in chute_tags if chute.tag_id not in [11, 12, 13, 14, 19, 20, 21, 22]]
        chute_left = [chute for chute in chute_tags if chute.tag_id in [11, 12, 13, 14, 19, 20, 21, 22]]
        if len(chute_right) == 0 or len(chute_left) == 0:
            return frame
        tag_angle = self._get_tag_angle(chute_left) 

        for tag in number_tags:
            level_center = self._get_right_side_center(tag)
            closest_right_chute = self._find_closest_chute(level_center, chute_right)
            closest_left_chute = self._find_closest_chute(level_center, chute_left)

            if closest_right_chute and closest_left_chute:
                line_start = tuple(map(int, level_center))
                line_end = (int(closest_right_chute[0] + abs(closest_left_chute[0] - level_center[0])), 
                            int(level_center[1]))
                line_start, line_end = self._rotate_line(line_start, line_end, tag_angle)  # Adjust rotation
                cv2.line(frame, line_start, line_end, (0, 255, 0), 2)
                cv2.putText(frame, f"{int(tag.tag_id) * 10}%", 
                            (line_end[0] + 35, line_end[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return frame
    
    def _find_closest_chute(self, level_center: Tuple[int, int], chute_tags: list) -> Optional[Tuple[int, int]]:
        """Finds the closest chute tag to the given level center."""
        min_distance = float('inf')
        closest_tag = None
        for chute in chute_tags:
            distance = abs(chute.center[1] - level_center[1])
            if distance < min_distance:
                min_distance = distance
                closest_tag = chute.center
        return closest_tag

    def _get_right_side_center(self, tag: Any) -> Tuple[int, int]:
        """Get the center of the right side of the tag."""
        corners = self._order_corners(tag.corners, tag.center)
        top_right = corners[1]
        bottom_right = corners[2]
        return (top_right + bottom_right) / 2
    
    def _handle_cutouts(self, frame: np.ndarray, chute_tags: list, use_cutout: bool):
        """Handles creation of a cutout based on chute tags."""
        tag_graph = TagGraphWithPositionsCV(self.ADI.tag_connections, chute_tags, self)
        tag_graph.account_for_missing_tags()
        overlay_frame = tag_graph.draw_overlay(frame)
        cutout = tag_graph.crop_to_size(frame)
        return (overlay_frame, cutout) if use_cutout else (overlay_frame, None)

    def _crop_to_bbox(self, frame, results):
        """
        Crop the frame to the bounding box provided in results.

        Parameters:
        -----------
        frame: np.ndarray
            The input image/frame to crop.
        results: Optional
            Results containing bounding box data.

        Returns:
        --------
        np.ndarray
            The cropped (or original) frame.
        """
        if results is None:
            return frame

        bbox_chute = results[1][0].flatten().cpu().detach().numpy()
        if len(bbox_chute) != 8:  # Ensure bbox has 8 coordinates
            return frame

        return cc.rotate_and_crop_to_bbox(frame, bbox_chute)[0]
    
    def _detect_edges(self, frame_data):
        """
        Perform edge detection on the frame.

        Parameters:
        -----------
        frame_data: np.ndarray
            The preprocessed frame to detect edges on.

        Returns:
        --------
        torch.Tensor
            The edge-detected version of the frame as a tensor.
        """
        try:
            edges = cv2.Canny(frame_data, 100, 200)
            edges = torch.from_numpy(edges.reshape(1, edges.shape[0], edges.shape[1])) / 255
            return edges
        except Exception as e:
            print("Error in edge detection:", e)
            return None

    def _visualize_frame(self, frame_data, edges=None):
        """
        Visualize the normalized frame and edges for debugging.

        Parameters:
        -----------
        frame_data: torch.Tensor
            The normalized frame.
        edges: Optional[torch.Tensor]
            The edge-detected version of the frame.
        """
        # Convert frame for visualization
        vis_frame = cv2.cvtColor(frame_data.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
        vis_frame = cv2.resize(vis_frame, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("Frame", np.clip(vis_frame, 0, 1))

        # If edges are provided, visualize them as well
        if edges is not None:
            edge_vis = edges.permute(1, 2, 0).numpy()
            edge_vis = cv2.resize(edge_vis, (0, 0), fx=0.6, fy=0.6)
            cv2.imshow("Edges", edge_vis)

        cv2.waitKey(1)

    def _combine_with_edges(self, frame_data, edges):
        """
        Combine the frame with its edge-detected version if edges are available.

        Parameters:
        -----------
        frame_data: torch.Tensor
            The normalized frame.
        edges: Optional[torch.Tensor]
            The edge-detected version of the frame.

        Returns:
        --------
        torch.Tensor
            The combined image (frame + edges) or the frame alone.
        """
        if edges is not None:
            return torch.cat((frame_data, edges), dim=0)
        return frame_data
    

    def _account_for_missing_tags_in_chute_numbers(self):
        # first we sort based on the tag id
        temp = copy.deepcopy(self.ADI.chute_numbers)
        sorted_chute_numbers = {k: v for k, v in sorted(temp.items(), key=lambda item: item[0])}
        prev_tag_id = None

        # we run through the sorted chute numbers and check if there are any missing tags. All mising tags with a tag id between the two tags can be inferred.
        for tag_id, center in sorted_chute_numbers.items():
            if prev_tag_id is None:
                prev_tag_id = tag_id
                continue
            if tag_id - prev_tag_id == 2:
                print(f"Missing tag between {prev_tag_id} and {tag_id} ---- {tag_id - 1}")
                # Infer the position of the missing tag by taking the mean of the two neighboring tags
                inferred_position = (np.array(sorted_chute_numbers[prev_tag_id]) + np.array(center)) / 2
                self.ADI.lock.acquire()
                self.ADI.chute_numbers[tag_id-1] = inferred_position
                self.ADI.lock.release()
            prev_tag_id = tag_id
        
    def _rotate_line(self, point1: Tuple[int, int], point2: Tuple[int, int], angle: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Rotate a line by a given angle around point1."""
        p1 = np.array(point1)
        p2 = np.array(point2)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_p2 = rotation_matrix @ (p2 - p1) + p1
        return tuple(p1), tuple(rotated_p2.astype(int))


    def _get_tag_connections(self) -> list:
        """Return the list of tag connections."""
        return [
            (11, 12), (12, 13), (13, 14), (14, 19), (19, 20), (20, 21),
            (21, 22), (22, 26), (26, 25), (25, 24), (24, 23), (23, 18),
            (18, 17), (17, 16), (16, 15), (15, 11)
        ]
    
    def _load_normalisation_constants(self):
        # Loads the normalisation constants from data/processed/statistics.yaml
        with open("data/processed/statistics.yaml", 'r') as file:
            data = yaml.safe_load(file)
            mean = data['mean']
            std = data['std']
        return mean, std
        
    def _load_camera_params(self):
        # open npz file
        with np.load("fiducial_marker/calibration.npz") as data:
            cameraMatrix = data['cameraMatrix']
            distCoeffs = data['distCoeffs']
            rvecs = data['rvecs']
            tvecs = data['tvecs']
        # load an image to get shape 
        self.ADI.fx = cameraMatrix[0, 0]
        self.ADI.fy = cameraMatrix[1, 1]
        self.ADI.cx = cameraMatrix[0, 2]
        self.ADI.cy = cameraMatrix[1, 2]
        return {"cameraMatrix": cameraMatrix, "distCoeffs": distCoeffs, "rvecs": rvecs, "tvecs": tvecs}

    def _fix_frame(self, frame: np.ndarray, blur: bool = False, balance: int = 1) -> np.ndarray:
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
        
        K, D = self.ADI.camera_params["cameraMatrix"], self.ADI.camera_params["distCoeffs"]
        h,  w = frame.shape[:2]
        new_K = K.copy()
        new_K[0,0] *= balance  # Scale fx and fy
        new_K[1,1] *= balance  # Scale fy 
        return cv2.fisheye.undistortImage(frame, K, D, Knew=new_K,new_size=(w,h))

    def _check_for_changes(self, tags: list) -> None:
        """
        Check if camera movement or tag shifts necessitate resetting state.

        Args:
            tags: List of detected tags in the current frame.
        """
        tag_ids = np.array([int(tag.tag_id) for tag in tags])
        accumulated_error = 0

        for t in self.ADI.tag_ids:
            matching_tags = np.where(tag_ids == t)[0]
            if matching_tags.size == 0:
                continue

            detected_tag = tags[matching_tags[0]]
            prev_tag = self.ADI.tags[t]
            accumulated_error += np.linalg.norm(np.array(detected_tag.center) - np.array(prev_tag.center))
        
        # First we make sure that the tags are not empty
        if len(self.ADI.tag_ids) != 0:
            if accumulated_error / len(self.ADI.tag_ids) > 10: # Threshold for accumulated error is 10 pixels
                self._reset_tags()

    def _reset_tags(self) -> None:
        """
        Resets the tag-related attributes to their initial states.
        """
        self.ADI.tags = {}
        self.ADI.tag_ids = []
        self.ADI.chute_numbers = {}

    def _process_tags(self, tags: list, detected_tags: list, unique_tag_ids: set, offsets: Tuple[int, int] = (0, 0)) -> Tuple[list, set]:
        """
        Process detected tags, adjust their coordinates, and update relevant attributes.
        
        Parameters:
        -----------
        tags: list
            List of detected AprilTags to process.
        frame: np.ndarray (Optional)
            Frame in which the tags were detected, used for optional operations.
        detected_tags: list
            List to store tags with original coordinates.
        unique_tag_ids: set
            Set to track unique tags in detected_tags.
        offsets: tuple
            Tuple (x_offset, y_offset) to adjust tag coordinates if detected in a cropped frame.
        """
        x_offset, y_offset = offsets

        for tag in tags:
            tag.center = (tag.center[0] + x_offset, tag.center[1] + y_offset)
            tag.corners[:, 0] += x_offset
            tag.corners[:, 1] += y_offset

            if tag.tag_id not in self.ADI.tag_ids:
                if tag.tag_id in range(11):  # Chute numbers assumed to be tag IDs [0-10]
                    self.ADI.chute_numbers[tag.tag_id] = tag.center
                self.ADI.tags[tag.tag_id] = tag
                self.ADI.tag_ids = np.append(self.ADI.tag_ids, int(tag.tag_id))

            if tag.tag_id not in unique_tag_ids:
                unique_tag_ids.add(tag.tag_id)
                detected_tags.append(tag)

        return detected_tags, unique_tag_ids
    
    def _refine_detection(self, frame: np.ndarray, tag, margin: int = 150):
        """
        Performs refined detection around a tag's region.

        Args:
            frame: Frame in which the tag is detected.
            tag: Tag object with initial detection.
            margin: Margin around tag's center for refinement.

        Returns:
            List of tags detected in the refined region.
        """
        x_start, x_end = max(0, tag.center[0] - margin), min(frame.shape[1], tag.center[0] + margin)
        y_start, y_end = max(0, tag.center[1] - margin), min(frame.shape[0], tag.center[1] + margin)
        cropped_frame = frame[int(y_start):int(y_end), int(x_start):int(x_end)]
        cropped_frame = self._fix_frame(cropped_frame, blur=True)

        return self.ADI.detector.detect(cropped_frame), x_start, y_start
    
    def _prepare_for_inference(self, frame, results=None, visualize=False):
        """
        Prepare a frame for inference by cropping, normalizing, and optionally detecting edges.

        Parameters:
        -----------
        frame: np.ndarray
            The input image/frame to be processed.
        results: Optional
            The results containing bounding box data for cropping. Default is None.
        visualize: bool
            Whether to visualize intermediate results for debugging. Default is False.

        Returns:
        --------
        torch.Tensor
            The preprocessed frame ready for inference.
        """
        # Crop the frame based on results, if provided
        frame_data = self._crop_to_bbox(frame, results)
        
        # Detect edges if required
        edges = self._detect_edges(frame_data) if self.ADI.edges else None

        # Normalize the frame
        frame_data = self.ADI.transform(torch.from_numpy(frame_data).permute(2, 0, 1).float())

        # Visualize intermediate results if requested
        if visualize:
            self._visualize_frame(frame_data, edges)

        # Stack edges with the frame if required
        cutout_image = self._combine_with_edges(frame_data, edges)

        # Resize and add batch dimension
        cutout_image = self.ADI.resize(cutout_image).unsqueeze(0)

        return cutout_image
    
    def _get_pixel_to_straw_level(self, frame, straw_bbox):
        """ Finds the straw level based on the detected tags in the chute. """
        chute_numbers_ = self.ADI.chute_numbers.copy()
        if not len(chute_numbers_) >= 2:
            return "NA"
            
        _, straw_cord,_ , _ = straw_bbox
        straw_cord = straw_cord[0].flatten()
        
        angle = self._get_tag_angle(list(chute_numbers_.values()))
        # from radians to degrees
        angle = np.degrees(angle)
        # Rotate the frame to the angle of the chute and the bbox
        _, _, bbox_, affine_warp = SpecialRotate(image=frame, bbox=straw_bbox[1][0].cpu().numpy(), angle=angle, return_affine=True) # type: ignore
        c_nr = np.expand_dims(np.array(list(chute_numbers_.values())).reshape(-1, 2), 1)
        warped_chute_numbers = cv2.perspectiveTransform(c_nr, affine_warp).squeeze(1)
        # replace the old values in the dict. Remember that the order is the same
        chute_numbers = {}
        for i, (k,_) in enumerate(chute_numbers_.items()):
            chute_numbers[k] = tuple(warped_chute_numbers[i])

        # Extract the top of the straw bbox
        straw_top = (bbox_[1] + bbox_[-1])/2
        
        # Given the straw bbox, we need to calculate the straw level based on the center of each tag in the chute. We know that the id of each tag corresponds to the level of the chute, meaning 1 is 10%, 2 is 20% and so on. We need to find the two closest tags in the y-axis to the straw bbox and calculate the straw level based on the distance between the two tags.
        # We can do this by calculating the distance between the straw bbox and the center of each tag in the chute. We then sort the distances and find the two closest tags. We then calculate the distance between the straw bbox and the two closest tags and use this to calculate the straw level.
        distance_dict_under = {}
        distance_dict_above = {}
        for key, values in chute_numbers.items():
            distance = straw_top - values[1]
            if distance < 0:
                distance_dict_under[distance] = key
            else:
                distance_dict_above[distance] = key
         
        # sort the dictionary by key
        distance_dict_under = dict(sorted(distance_dict_under.items(), reverse=True))
        distance_dict_above = dict(sorted(distance_dict_above.items()))
        
        # get the two closest tags
        tag_under, tag_above = list(distance_dict_under.values())[0], list(distance_dict_above.values())[0]

        # there are three cases to consider, no detected tags under, no detected tags above, and detected tags both above and under
        # lets first make a check to i see if the closest tag under is 10. Meaning then we should clip it to 10        
        if len(distance_dict_under) == 0:
            if tag_above == 0:
                return 0.0
            return "NA"
        elif len(distance_dict_above) == 0:
            if tag_under == 10:
                return 100
            return "NA"
        
        # we get the difference between the two tags ids to see if we are missing tags inbetween
        tag_diff = tag_above - tag_under
        if tag_diff > 1:
            interpolated = True
        else:
            interpolated = False
        
        # If the tag_diff is greateer than one, then we need to perform a linear interpolation between the points to get the straw level
        y_under = chute_numbers[tag_under][1]
        y_over = chute_numbers[tag_above][1]
        x_mean = (chute_numbers_[tag_under][0] + chute_numbers_[tag_above][0]) / 2
        
        # given the two y-values, take the y-value for straw_top and calculate the percentage of the straw level
        straw_level = (tag_diff * (y_under-straw_top) / (y_under-y_over) + tag_under)*10
        
        if self.ADI.record and self.ADI.recording_req:
            self.ADI.prediction_dict["yolo"] = {straw_level: (x_mean, straw_top)}
            self.ADI.prediction_dict["attr."] = {interpolated: sorted(chute_numbers.keys())}
        
        return straw_level
    
    def _get_straw_to_pixel_level(self, straw_level):
        # We know that the self.chute_numbers are ordered from 0 to 10. We can use this to calculate the pixel value of the straw level
        # we know that each tag is 10% of the chute, meaning that the distance between each tag is 10% of the chute height. We can use 
        # this to calculate the pixel value of the straw level.
        # We can use the distance between the two closest tags to calculate the pixel value of the straw level.
        chute_numbers = self.ADI.chute_numbers.copy()
        # make sure there are chute numbers to work with, otherwise we return
        if not len(chute_numbers) >= 2:
            return "NA"

        # First we divide the straw level by 10 to get it on the same scale as the tag ids
        straw_level = straw_level / 10

        # We then get the two closest tags
        tag_under, tag_over = int(straw_level), int(straw_level) + 1
        
        # next we find the two closest tags in chute_numbers based on the tag ids
        # First we create a list for the values that are less or equal to the tag_under and greater than the tag_over
        tag_under_list = [key for key, _ in chute_numbers.items() if key <= tag_under]
        tag_over_list = [key for key, _ in chute_numbers.items() if key >= tag_over]

        # next we order them
        tag_under_list = sorted(tag_under_list, reverse=True)
        tag_over_list = sorted(tag_over_list)
        
        # Then we see if the tag_under_closest is above or below the straw level
        tag_under_closest = tag_under_list[0]
        tag_over_closest = tag_over_list[0]

        # calculate difference between tag ids
        tag_diff = tag_over_closest - tag_under_closest
        if tag_diff > 1:
            interpolated = True
        else:
            interpolated = False
        
        # get the distance between the two closest tags
        y_under = chute_numbers[tag_under_closest][1]
        y_over = chute_numbers[tag_over_closest][1]
        
        # get the pixel value of the straw level
        excess = straw_level - tag_under_closest
        pixel_straw_level_x = (chute_numbers[tag_under_closest][0] + chute_numbers[tag_over_closest][0]) / 2
        pixel_straw_level_y = y_under - (y_under - y_over) * excess/tag_diff
        
        return (pixel_straw_level_x, pixel_straw_level_y)

    def _smooth_level(self, level: float | None, id:str):
        """Smooth the straw level using a queue."""
        if id == 'scada':
            self.ADI.scada_smoothing_queue.append(level)
            filtered_data = [x for x in self.ADI.scada_smoothing_queue if x is not None]
            return np.mean(filtered_data)
        elif id == 'straw':
            self.ADI.straw_smoothing_queue.append(level)
            filtered_data = [x for x in self.ADI.straw_smoothing_queue if x is not None]
            return np.mean(filtered_data)
                
    
    def _grab_scada_url_n_id(self):
        # Read the url from the scada.txt file
        data_path = 'data/opcua_server.txt'
        txt_file = open(data_path, 'r')
        url = txt_file.readline().strip()
        print(f'Read url: {url} from: {data_path}')
        sensor_node_id = txt_file.readline().strip()
        return url, sensor_node_id

class TagGraphWithPositionsCV:
    """
    A class to represent a graph of connected tags with inferred positions for missing tags. This class is designed to be used
    in conjunction with the AprilDetector class to detect AprilTags in a real-time video stream. The class provides methods to
    interpolate the position of a missing tag based on its connected neighbors, account for missing tags, crop the image to the
    size of the chute system, and draw the detected and inferred tags on the image. 

    NOTE This class is highly customized to detect AprilTags in the chute system af meliora bio, and may not be suitable for other
    applications without modification as it has hard-coded tag ids and corner tags.
    """

    def __init__(self, connections, detected_tags, helpers):
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
        self.helpers = helpers
        self.connections = connections
        self.detected_tags = {tag.tag_id: tuple(map(int, tag.center)) for tag in detected_tags}  # Detected tag positions
        self.inferred_tags = {}  # Store inferred tag positions
        self.corner_tags = set([11, 15, 26, 22])  # Store corner tags
        self.left_tags = set([12, 13, 14, 19, 20, 21])  # Store left tags
        self.right_tags = set([16, 17, 18, 23, 24, 25])  # Store right tags
        
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
        
        detected_tag_ids = set(list(self.detected_tags.keys()))

        if is_corner:
            if tag_id == 11:
                try:
                    intersection = list(detected_tag_ids - (detected_tag_ids - self.left_tags))
                    p1 = intersection[0]
                    p2 = intersection[1]
                    x_value, y_value = self.intersection_point(self.detected_tags[p1][0], 
                                                               self.detected_tags[p1][1], 
                                                               self.detected_tags[p2][0], 
                                                               self.detected_tags[p2][1], 
                                                               self.detected_tags[15][0], 
                                                               self.detected_tags[15][1])
                except KeyError:
                    return None
            if tag_id == 15:
                try:
                    intersection = list(detected_tag_ids - (detected_tag_ids - self.right_tags))
                    p1 = intersection[0]
                    p2 = intersection[1]
                    x_value, y_value = self.intersection_point(self.detected_tags[p1][0], 
                                                               self.detected_tags[p1][1], 
                                                               self.detected_tags[p2][0], 
                                                               self.detected_tags[p2][1], 
                                                               self.detected_tags[11][0], 
                                                               self.detected_tags[11][1])
                except KeyError:
                    return None
            if tag_id == 22:
                try:
                    intersection = list(detected_tag_ids - (detected_tag_ids - self.left_tags))[::-1]
                    p1 = intersection[0]
                    p2 = intersection[1]
                    x_value, y_value = self.intersection_point(self.detected_tags[p1][0], 
                                                               self.detected_tags[p1][1], 
                                                               self.detected_tags[p2][0], 
                                                               self.detected_tags[p2][1], 
                                                               self.detected_tags[26][0], 
                                                               self.detected_tags[26][1])
                except KeyError:
                    return None
            if tag_id == 26:
                try:
                    intersection = list(detected_tag_ids - (detected_tag_ids - self.right_tags))[::-1]
                    p1 = intersection[0]
                    p2 = intersection[1]

                    x_value, y_value = self.intersection_point(self.detected_tags[p1][0], 
                                                               self.detected_tags[p1][1], 
                                                               self.detected_tags[p2][0], 
                                                               self.detected_tags[p2][1], 
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
        if self._is_valid_quadrilateral(pts_src) and self._is_valid_quadrilateral(pts_dst):
            # Compute the perspective transformation matrix
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            # Apply the perspective warp to create a square cutout
            cutout = cv2.warpPerspective(image, M, (width, height))
            return np.array(cutout)
        else:
            return None
    
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
    
    def _is_valid_quadrilateral(self, pts):
        """Check if the points form a valid quadrilateral."""
        if len(pts) != 4:
            return False
        # Check for collinearity
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            x3, y3 = pts[(i + 2) % 4]
            if (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1):
                return False
        return True


class AsyncStreamThread:
    def __init__(self, server_keys: str):
        self.grab_keys()
        self.server_keys = server_keys
        self.recent_value = 50  # Shared variable for the most recent value
        self.lock = threading.Lock()  # Lock for thread-safe access
        self.loop = asyncio.new_event_loop()  # Create a new event loop
        self.thread = threading.Thread(target=self._start_event_loop, daemon=True)  # Worker thread
        self.wait = True

    def _start_event_loop(self):
        """Start the event loop in the thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._stream_values())

    async def _stream_values(self):
        """Async function to open the stream and update the recent value."""
        async with Client(self.url) as client:
            print(f'Connected to {self.url}!')
            sensor_node = client.get_node(self.sensor_node_id)
            while self.wait:
                value = await sensor_node.get_value()
                # Update the recent value in a thread-safe manner
                with self.lock:
                    self.recent_value = value
                await asyncio.sleep(0.02)
    
    def grab_keys(self):
        data_path = 'data/opcua_server.txt'
        txt_file = open(data_path, 'r')
        self.url = txt_file.readline().strip()
        print(f'Read url: {self.url} from: {data_path}')
        self.sensor_node_id = txt_file.readline().strip()
        print(f'Read sensor node id: {self.sensor_node_id} from: {data_path}')

    def start(self):
        """Start the thread."""
        self.thread.start()

    def get_recent_value(self) -> Optional[float]:
        """Get the most recent value in a thread-safe manner."""
        with self.lock:
            return self.recent_value
        
    def print_pending_tasks(self):
        tasks = [t for t in asyncio.all_tasks(self.loop) if t is not asyncio.current_task(self.loop)]
        print(f"Pending tasks: {tasks}")

    def join(self):
        """Stop the event loop and thread."""
        self.wait = False
        self.thread.join()

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
