from calendar import c
import cv2
from cv2 import aruco, rectangle
import numpy as np
from typing import Tuple, Optional, Any
import os
import pupil_apriltags
import time
import threading
import queue

class ArucoDetector:
    def __init__(self, marker_dict, detector_params, window=False):
        self.marker_dict = marker_dict
        self.detector_params = detector_params
        self.window = window
        self.detector = self.create_detector()
        self.q = queue.Queue()

    def create_detector(self):
        self.detector_params.minMarkerPerimeterRate = 0.01
        return aruco.ArucoDetector(self.marker_dict, self.detector_params)

    def detectMarkers(self, frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.detector.detectMarkers(frame)

    def drawMarkers(self, frame, marker_corners, marker_IDs):
        if marker_corners:
            for ids, corners in zip(marker_IDs, marker_corners):
                cv2.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                )
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                # top_left = corners[1].ravel()
                # bottom_right = corners[2].ravel()
                # bottom_left = corners[3].ravel()
                cv2.putText(
                    frame,
                    f"id: {ids[0]}",
                    top_right,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.3,
                    (200, 100, 0),
                    2,
                    cv2.LINE_AA,
                )
        return frame

    def __call__(self, frame: Optional[np.ndarray] = None, cap = None):
        if self.window:
            if not cap:
                cap = cv2.VideoCapture(0)
            while True:
                success, frame = cap.read()
                if not success:
                    continue
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                marker_corners, marker_IDs, reject = self.detectMarkers(gray_frame)
                print(marker_IDs)
                frame = self.drawMarkers(frame, marker_corners, marker_IDs)
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            marker_corners, marker_IDs, reject = self.detectMarkers(frame)
            frame = self.drawMarkers(frame, marker_corners, marker_IDs)
            return frame, marker_corners, marker_IDs


class TagGraphWithPositionsCV:
    def __init__(self, connections, detected_tags):
        """
        connections: List of tuples defining the connected tag ids (e.g., (11, 12), (12, 13))
        detected_tags: List of tuples (center_x, center_y, tag_id) for detected tags
        image: The image to overlay on
        corner_tags: List of tag ids that are corner points
        """
        self.connections = connections
        self.detected_tags = {tag_id: (int(x), int(y)) for (x, y, tag_id) in detected_tags}  # Detected tag positions
        self.inferred_tags = {}  # Store inferred tag positions
        self.corner_tags = set([11, 15, 26, 22])  # Store corner tags
    
    def interpolate_position(self, tag_id, neighbor_ids, is_corner):
        """
        Interpolate the position of a missing tag based on its connected neighbors.
        For corners, we prioritize direct neighbors (no averaging all around).
        """
        neighbor_positions = [self.detected_tags.get(n) for n in neighbor_ids if n in self.detected_tags]
        # TODO Account for slanting in the image, e.g. the top right corner should have a 90 degree angle to the top left corner with
        # the line created by the right site tags.
        # left_line = [11, 12, 13, 14, 19, 20, 21, 22]
        # left_line_coordinates = [self.detected_tags[tag] for tag in left_line]
        
        # right_line = [15, 16, 17, 18, 23, 24, 25, 26]
        
        
        if not neighbor_positions:
            return None
        
        if is_corner:
            if tag_id == 11:
                try:
                    x_value = self.detected_tags[12][0]
                    y_value = self.detected_tags[15][1]
                except KeyError:
                    return None
            if tag_id == 15:
                try:
                    x_value = self.detected_tags[16][0]
                    y_value = self.detected_tags[11][1]
                except KeyError:
                    return None
            if tag_id == 22:
                try:
                    x_value = self.detected_tags[26][0]
                    y_value = self.detected_tags[23][1]
                except KeyError:
                    return None
            if tag_id == 26:
                try:
                    x_value = self.detected_tags[25][0]
                    y_value = self.detected_tags[22][1]
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
        from PIL import Image, ImageDraw
        import numpy as np
        import cv2

        # Convert the image to RGBA format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        
        # Create a mask with the same size as the image
        maskIm = Image.new('L', (image.shape[1], image.shape[0]), 0)
        
        # Create a list to store the ordered coordinates
        ordered_coords = []
        
        # Iterate through the connections and add the coordinates to the list
        for tag1, tag2 in self.connections:
            if tag1 in self.detected_tags and self.detected_tags[tag1] not in ordered_coords:
                ordered_coords.append(self.detected_tags[tag1])
            if tag2 in self.detected_tags and self.detected_tags[tag2] not in ordered_coords:
                ordered_coords.append(self.detected_tags[tag2])
        
        # Draw the polygon on the mask
        ImageDraw.Draw(maskIm).polygon(ordered_coords, outline=1, fill=1)
        
        # Convert the mask to a numpy array
        mask = np.array(maskIm)
        
        # Apply the mask to the image
        newImArray = np.empty(image.shape, dtype='uint8')
        newImArray[:, :, :3] = image[:, :, :3]
        newImArray[:, :, 3] = mask * 255
        
        # Convert the numpy array back to an image
        # NOTE When perfoming inference with CNN model, we should use this newIm as it makes a cutout that is non-square and more accurate.
        # The code under is just for visualisation as jpg, but that is not needed when running inference.
        newIm = Image.fromarray(newImArray, "RGBA")
        # cv2.imwrite("fiducial_marker/cutout.jpg", cv2.cvtColor(np.array(newIm), cv2.COLOR_RGBA2BGRA))
        # get bbox
        bbox = newIm.getbbox()
        newIm = newIm.crop(bbox)
        # Return the cropped image as a numpy array
        return np.array(newIm)
    
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
    def __init__(self, detector: pupil_apriltags.bindings.Detector, ids: dict, window: bool=False, frame_shape: tuple = (1440, 2560)) -> None:
        self.detector = detector
        self.window = window
        self.ids = ids
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

        
    def load_camera_params(self):
        # open npz file
        with np.load("fiducial_marker/calibration.npz") as data:
            cameraMatrix = data['cameraMatrix']
            distCoeffs = data['distCoeffs']
            rvecs = data['rvecs']
            tvecs = data['tvecs']
        # load an image to get shape 
        img = cv2.imread("data/calibration/images/frame_0.jpg")
        # chagne cx and cy to the center of the image
        # cameraMatrix[0, 2] = img.shape[1] // 2
        # cameraMatrix[1, 2] = img.shape[0] // 2
        self.fx = cameraMatrix[0, 0]
        self.fy = cameraMatrix[1, 1]
        self.cx = cameraMatrix[0, 2]
        self.cy = cameraMatrix[1, 2]
        return {"cameraMatrix": cameraMatrix, "distCoeffs": distCoeffs, "rvecs": rvecs, "tvecs": tvecs}
    
    def gaussian_filter(self, kernel_size,img,sigma=1, muu=0):
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                        np.linspace(-1, 1, kernel_size))
        dst = np.sqrt(x**2+y**2)
        normal = 1/(((2*np.pi)**0.5)*sigma)
        gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
        gauss = np.pad(gauss, [(0, img.shape[0] - gauss.shape[0]), (0, img.shape[1] - gauss.shape[1])], 'constant')
        return gauss

    def fft_deblur(self, img,kernel_size,kernel_sigma=5,factor='wiener',const=0.002):
        gauss = self.gaussian_filter(kernel_size,img,kernel_sigma)
        img_fft = np.fft.fft2(img)
        gauss_fft = np.fft.fft2(gauss)
        weiner_factor = 1 / (1+(const/np.abs(gauss_fft)**2))
        if factor!='wiener':
            weiner_factor = factor
        recon = img_fft/gauss_fft
        recon*=weiner_factor
        recon = np.abs(np.fft.ifft2(recon))
        return recon
       
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
        K, D = self.camera_params["cameraMatrix"], self.camera_params["distCoeffs"]
        h,  w = frame.shape[:2]
        # change c_x and c_y to the center of the image
        # mtx[0, 2] = w/2
        # mtx[1, 2] = h/2
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
        new_K = K.copy()
        new_K[0,0] *= balance  # Scale fx
        new_K[1,1] *= balance  # Scale fy
            
        undistorted_image = cv2.fisheye.undistortImage(frame, K, D, Knew=new_K,new_size=(w,h))



        # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.camera_params["cameraMatrix"], self.camera_params["distCoeffs"], (w,h), 1, (w,h))
        # dst = cv2.undistort(frame, self.camera_params["cameraMatrix"], self.camera_params["distCoeffs"], None, newcameramtx)
        
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        
        # frame = self.fft_deblur(dst, 5, 5, factor="wiener")
        # kernel = np.array([[0, -1, 0],
        #            [-1, 5,-1],
        #            [0, -1, 0]])
        # image_sharp = cv2.filter2D(src=dst, ddepth=-1, kernel=kernel)
        # frame = cv2.undistort(frame, self.camera_params["cameraMatrix"], self.camera_params["distCoeffs"])
        # K, d = self.camera_params["cameraMatrix"], self.camera_params["distCoeffs"]
        # final_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, d, (w, h), np.eye(3), balance=1.0)

        # map_1, map_2 = cv2.fisheye.initUndistortRectifyMap(K, d, np.eye(3), final_K, (w, h), cv2.CV_32FC1)

        # undistorted_image = cv2.remap(frame, map_1, map_2, interpola½tion=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # NOTE down here we apply image correction to the frame, meaning gamma correction, brightness and contrast
        # image = cv2.bilateralFilter(frame, 9, 75, 75)
        if blur:
            image = cv2.GaussianBlur(frame, (7, 7), 0.7) 
            return image.astype(np.uint8)
        return undistorted_image 
        
    def detect(self, frame: np.ndarray) -> np.ndarray[Any] | list:
        """
        Wrapper method to detect AprilTags in a frame using the pupil_apriltags library. While detecting it checks if the
        detected tags are already present in the tags array. If not, it appends the new tags to the tags array. Also, for 
        each time the function is called it checks if the camera has moved and the chute tags have changed position. If so,
        it resets the tags and tag_ids arrays.

        Params:
        -------
        frame: np.ndarray
            The frame in which the AprilTags are to be detected
        
        Returns:
        --------
        List
            A list of detected AprilTags in the frame
        """     
        # Detect tags on the frame
        # frame = self.fix_frame(frame)

        tags = self.detector.detect(frame, tag_size=0.05) #, estimate_tag_pose=True, camera_params=[self.fx, self.fy, self.cx, self.cy], tag_size=0.05) # 5cm
        for tag in tags:
            if tag.tag_id not in self.tag_ids:            
                self.tags = np.append(self.tags, tag)
                self.tag_ids = np.append(self.tag_ids, int(tag.tag_id))
        self.check_for_changes(tags)
        return self.tags
    
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
        if len(tags) == 0:
            self.tags = []
            self.tag_ids = []
            return
        tag_ids = np.array([int(tag.tag_id) for tag in tags])
        accumulated_error = 0
        for i, t in enumerate(self.tag_ids):
            mask = np.where(tag_ids == t)[0]
            if mask.size == 0:
                continue
            t1 = tags[mask[0]]
            t2 = self.tags[i]
            # calculate absolute distance between the two tags
            accumulated_error += np.linalg.norm(t1.center - t2.center)

        # If the camera moves and the chute tags change position, reset the tags
        if accumulated_error/len(self.tag_ids) > 0.1:
            self.tags = []
            self.tag_ids = []

    def inverse_linear_interpolation(self, x_values: list, y_values: list, y: float) -> int | None:
        """
        Perform inverse linear interpolation to find the interpolated x position for a given y value.
        
        Params:
        -------
        x_values: List
            List of known x positions (must be sorted in ascending order)
        y_values: List
            List of corresponding y values at each x position
        y: float
            The y value where the inverse interpolation is to be performed

        Returns:
        --------
        int | None
            Interpolated x position for the value y
        """
        # Ensure that x_values and y_values have the same length
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length")

        # Check if y is outside the known range
        if y < min(y_values) or y > max(y_values):
            raise ValueError("y is outside the interpolation range")
        
        # Locate the interval [y_i, y_{i+1}] such that y_i <= y <= y_{i+1}
        for i in range(len(y_values) - 1):
            if y_values[i] <= y <= y_values[i + 1]:
                # Perform inverse linear interpolation
                y_i, y_i1 = y_values[i], y_values[i + 1]
                x_i, x_i1 = x_values[i], x_values[i + 1]
                x = (y_i1 - y) / (y_i1 - y_i) * x_i + (y - y_i) / (y_i1 - y_i) * x_i1
                return int(x)

    def get_angle(self, point1: Tuple, point2: Tuple) -> float:
        """
        Get the angle between two points wrt the x-axis.

        Params:
        -------
        point1: Tuple
            The first point
        point2: Tuple
            The second point    
        
        Returns:
        --------
        float
            The angle between the two points wrt the x-axis in radians
        """
        from math import atan2
        return atan2(point2[1] - point1[1], point2[0] - point1[0])
    
    def order_corners(self, corners: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        The purpose of this is to have the corners in the following order:
        top_left, top_right, bottom_right, bottom_left

        and we do this based on the coordiantes, knowing that topleft of the image is (0, 0)
        and bottom right is (width, height).

        Params:
        -------
        corners: np.ndarray
            The corners of the detected AprilTag
        center: np.ndarray
            The center of tag
            
        Returns:
        --------
        np.ndarray
            The ordered corners of the detected AprilTag
        """
        c = {}
        for i, corner in enumerate(corners):
            x, y = corner
            if x < center[0] and y < center[1]:
                c[0] = corner  # top_left
            elif x > center[0] and y < center[1]:
                c[1] = corner  # top_right
            elif x > center[0] and y > center[1]:
                c[2] = corner  # bottom_right
            else:
                c[3] = corner  # bottom_left
        if len(c) != 4:
            raise ValueError("The corners must have 4 points with the center in the middle")
        return np.array(list(dict(sorted(c.items())).values()))

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

    def draw(self, frame: np.ndarray, tags: list, make_cutout: bool = False, straw_level: float = 25) -> np.ndarray:
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

        number_tags = []
        chute_tags = []
        if len(tags) == 0:
            return frame
        for t in tags:
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

            # The logic is as follows:
            # 1. For each number tag, find the center of the chute tag that is closest to it on the y-axis in let and right
            # 2. Draw a line going from the right site of the number tag going horizontaly to x being the x value of the right chute tag plus the difference between the x value of the number tag and the x value of the left chute tag
            for tag in number_tags:
                corners = self.order_corners(tag.corners, tag.center)
                # we first define the top right and bottom right corners based on the coordinates
                top_left = corners[0]
                top_right = corners[1]
                bottom_right = corners[2]
                level_center_right = (top_right + bottom_right) / 2
                # get the angle of the tag wrt the x-axis for rotation purposes

                tag_angle = -self.get_angle(top_left, top_right)
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
                cutout = tag_graph.crop_to_size(frame)
                frame = tag_graph.draw_overlay(frame)
                
                # # save cutout image to disk
                # cv2.imwrite("fiducial_marker/cutout.jpg", cutout)
                return frame, cutout
            
            center_chute = (chute_right[0][0] + chute_left[0][0]) // 2
            number_tags = sorted(number_tags, key=lambda x: x.tag_id)
            number_tag_centers = np.array([tag.center for tag in number_tags])
            # PIXEL_VALUE = self.inverse_linear_interpolation(number_tag_centers[:, 1], np.array(range(0, 11))*10, straw_level)
            # cv2.circle(frame, (int(center_chute),  PIXEL_VALUE), 10, (0, 255, 0), -1)
            # cv2.putText(frame, f"{straw_level:.2f}%", (int(center_chute) - 20, PIXEL_VALUE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception as e:
            print("ERROR", e)
            

        return frame

class RTSPStream(AprilDetector):
    """
    Wrapper class to detect AprilTags in a real-time video stream. The class inherits from the AprilDetector class and
    provides methods to receive frames from the video stream, detect AprilTags in the frames, draw the detected tags on
    the frame, and display the frame with the detected tags. The class also provides a method to detect AprilTags in a
    single frame.

    NOTE Threading is necessary here because we are dealing with an RTSP stream.
    """
    def __init__(self, detector, ids, credentials_path, window=True, rtsp=True, make_cutout=False):
        super().__init__(detector, ids, window)
        if rtsp:
            self.cap = self.create_capture(credentials_path)
        else:
            self.cap = cv2.VideoCapture(0)
        self.make_cutout = make_cutout
        
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

    def crop_chute(self, frame: np.ndarray, left_tags: list, right_tags: list) -> np.ndarray:
        """
        This function will open up an additional cv2 window where the image will be cropped based on the detected tags.
        Only the chute will be shown in the window.
        
        Params:
        -------
        frame: np.ndarray
            The frame in which the AprilTags are to be detected
        tags: List
            A list of detected AprilTags in the frame 
        """
        if len(left_tags) == 0 or len(right_tags) == 0:
            return frame
        # TODO Add logic to crop the chute if the tags are detected
        return frame
        
    def receive_frame(self, cap: cv2.VideoCapture) -> None:
        """
        Read frames from the video stream and store them in a queue.

        Params:
        -------
        cap: cv2.VideoCapture
            The video capture object to receive frames from the RTSP stream
        
        Returns:
        --------
        None
            Nothing is returned, only the frames are stored in a queue
        """
        ret, frame = cap.read()
        self.q.put(frame)
        while ret and cap.isOpened():
            ret, frame = cap.read()
            self.q.put(frame)        

    def display_frame(self, cap: cv2.VideoCapture) -> None:
        """
        Display the frames with the detected AprilTags.
        
        Params:
        -------
        cap: cv2.VideoCapture
            The video capture object to receive frames from the RTSP stream
        
        Returns:
        --------
        None
            Nothing is returned, only the frames are displayed with the detected AprilTags
        """
        frame_count = 0
        start_time = time.time()
        while True:
            if not self.q.empty():
                frame = self.q.get()
                frame = self.fix_frame(frame)
                if frame_count % 5 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    tags = self.detect(self.fix_frame(gray, blur=True))
                frame = self.draw(frame, tags, self.make_cutout)
                frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                # Increment frame count
                frame_count += 1
                # Calculate FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                # Display FPS on the frame
                cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                # save the frame
                cv2.imwrite(f"fiducial_marker/frame_with_cutout.png", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def __call__(self, frame: Optional[np.ndarray] = None, cap: Optional[cv2.VideoCapture] = None) -> None | list:
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
            if cap:
                self.cap = cap
            self.thread1 = threading.Thread(target=self.receive_frame, args=(self.cap,))
            self.thread2 = threading.Thread(target=self.display_frame, args=(self.cap,))
            self.thread1.start()
            self.thread2.start()
        elif frame is not None:
            # resize frame half the size
            frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.detect(gray)
            if self.window:
                frame = self.draw(frame, tags)
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
            return tags
