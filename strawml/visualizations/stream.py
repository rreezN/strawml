from __init__ import *
# Misc.
import cv2
import time
import yaml
import torch
import queue
import psutil
import keyboard
import threading
import numpy as np
from typing import Tuple, Optional, Any
from torchvision.transforms import v2 as transforms

# Model imports
import timm
import pupil_apriltags

## File imports
from strawml.models.chute_finder.yolo import ObjectDetect
import strawml.models.straw_classifier.cnn_classifier as cnn
import strawml.models.straw_classifier.feature_model as feature_model
from strawml.visualizations.utils_stream import AprilDetectorHelpers


class AprilDetector:
    """
    NOTE This class is highly customized to detect AprilTags in the chute system af meliora bio. 

    AprilTag detector class that uses the pupil_apriltags library to detect AprilTags in a frame. The class is designed
    to be used in conjunction with the RTSPStream class to detect AprilTags in a real-time video stream. The class
    provides methods to detect AprilTags in a frame, draw the detected tags on the frame, and given a predicted straw level
    value performs inverse linear interpolation to get the corresponding pixel value on the frame.
    """
    def __init__(self, detector: pupil_apriltags.bindings.Detector, ids: dict, window: bool=False, od_model_name=None, object_detect=True, yolo_threshold=0.5, device="cuda", frame_shape: tuple = (1440, 2560), yolo_straw=False, yolo_straw_model="models/yolov11-straw-detect-obb.pt", with_predictor: bool = False, model_load_path: str = "models/vit_regressor/", regressor: bool = True, predictor_model: str = "vit", edges=True, heatmap=False) -> None:
        self.helpers = AprilDetectorHelpers(self)  # Instantiate the helper class

        self.lock = threading.Lock()
        self.detector = detector
        self.ids = ids
        self.od_model_name = od_model_name
        self.window = window
        self.device = device
        self.yolo_threshold = yolo_threshold
        self.edges = edges
        self.heatmap = heatmap
        self.yolo_straw = yolo_straw
        self.with_predictor = with_predictor
        self.regressor = regressor
        self.model_load_path = model_load_path
        self.predictor_model = predictor_model
        self.frame_shape = frame_shape
        self.model = None
        self.q = queue.Queue()
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.camera_params = self.helpers._load_camera_params()
        self.tags = {}
        self.tag_ids = np.array([])
        self.chute_numbers = {}
        self.tag_connections = self.helpers._get_tag_connections()
        self.processed_tags = set()  # Track tags that have already been re-centered
        self.detected_tags = []
        self.object_detect = object_detect

        # Setup object detection if enabled
        if self.object_detect:
            self.setup_object_detection(od_model_name)
        # Setup prediction model if enabled YOLO is prioritized over the predictor
        if yolo_straw:
            self.setup_yolo_straw(yolo_straw_model)
        elif with_predictor:
            self.setup_predictor()

    def setup_object_detection(self, od_model_name: str) -> None:
        """Setup object detection model."""
        self.OD = ObjectDetect(od_model_name, yolo_threshold=self.yolo_threshold, device=self.device, verbose=False)

    def setup_yolo_straw(self, yolo_straw_model: str) -> None:
        """Setup YOLO straw model."""
        self.model = ObjectDetect(model_name=yolo_straw_model, yolo_threshold=self.yolo_threshold, device=self.device, verbose=False)

    def setup_predictor(self) -> None:
        """Setup the predictor model and related transformations."""
        self.mean, self.std = self.helpers._load_normalisation_constants()
        self.img_unnormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(self.mean, self.std)],
            std=[1 / s for s in self.std]
        )
        self.transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=False),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        num_classes = 1 if self.regressor else 11
        input_channels = 3 + int(self.edges) + int(self.heatmap) * 3

        # Model selection based on predictor_model
        self.model, image_size = self.load_predictor_model(input_channels, num_classes)
        self.resize = transforms.Resize(image_size)

        if self.regressor:
            self.setup_regressor(image_size, input_channels)
        else:
            self.regressor_model = None
            self.model.load_state_dict(torch.load(self.model_load_path, weights_only=True))
            self.model.to(self.device)

    def load_predictor_model(self, input_channels: int, num_classes: int) -> torch.nn.Module:
        """Load the appropriate model based on the predictor_model string."""
        model = None
        match self.predictor_model:
            case 'cnn':
                image_size = (384, 384)
                model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels, output_size=num_classes)
            case 'convnextv2':
                image_size = (224, 224)
                model = timm.create_model('convnext_small.in12k_ft_in1k_384', pretrained=False, in_chans=input_channels, num_classes=num_classes)
            case 'vit':
                image_size = (384, 384)
                model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', pretrained=False, in_chans=input_channels, num_classes=num_classes)
            case 'eva02':
                image_size = (448, 448)
                model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=False, in_chans=input_channels, num_classes=num_classes)
            case 'caformer':
                image_size = (384, 384)
                model = timm.create_model('caformer_m36.sail_in22k_ft_in1k_384', in_chans=input_channels, num_classes=num_classes, pretrained=False)
        
        model.load_state_dict(torch.load(f'{self.model_load_path}/{self.predictor_model}_feature_extractor_overall_best.pth', weights_only=True))
        model.to(self.device)
        return model, image_size

    def setup_regressor(self, image_size: tuple, input_channels: int) -> None:
        """Setup the regressor model."""
        if self.predictor_model != 'cnn':
            features = self.model.forward_features(torch.randn(1, input_channels, image_size[0], image_size[1]))
            feature_size = torch.flatten(features, 1).shape[1]
            self.regressor_model = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1)
            
            self.model.load_state_dict(torch.load(f'{self.model_load_path}/{self.predictor_model}_feature_extractor_overall_best.pth', weights_only=True))
            self.regressor_model.load_state_dict(torch.load(f'{self.model_load_path}/{self.predictor_model}_regressor_overall_best.pth', weights_only=True))
            self.regressor_model.to(self.device)
        else:
            self.model.load_state_dict(torch.load(self.model_load_path, weights_only=True))
        self.model.to(self.device)

    def plot_boxes(self, results, frame, straw=False, straw_lvl=None, model_type: str = 'obb'):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        # First check if the results are empty
        if not results:
            return frame
        if 'obb' in model_type:
            labels, cord, labels_conf, angle_rad = results
        else:
            labels, cord, labels_conf = results
        n = len(labels)

        for i in range(n):  
            # plot polygon around the object based on the coordinates cord
            if 'obb' in model_type:
                x1, y1, x2, y2, x3, y3, x4, y4 = cord[i].flatten()
                x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
                if straw:
                    # Only plot the upper vertical line
                    cv2.line(frame, (x1, y1), (x4, y4), (127,0,255), 2)
                    cv2.putText(frame, f"{straw_lvl:.2f} %", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 0, 255), 2, cv2.LINE_AA)
                else:                        
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

    def detect(self, frame: np.ndarray) -> list:
        """
        Detects AprilTags in a frame, performs refined detection for precision, and handles deduplication.
        """
        detected_tags = []  # This will hold tags with original coordinates
        unique_tag_ids = set()  # Set to track unique tags in detected_tags
        tags = self.detector.detect(frame)

        detected_tags, unique_tag_ids = self.helpers._process_tags(tags, detected_tags, unique_tag_ids)

        # Perform refined detection in parallel regions
        for tag in tags:
            if tag.tag_id in self.processed_tags:
                continue
            # Mark the tag as processed to prevent re-centering on it again
            self.processed_tags.add(tag.tag_id)
            # Define crop region
            refined_tags, x_start, y_start = self.helpers._refine_detection(frame, tag, margin=150)
            # Process the refined tags and update state
            detected_tags, unique_tag_ids = self.helpers._process_tags(refined_tags, detected_tags, unique_tag_ids, offsets=(x_start, y_start))
        
        # Infer missing tags and update state
        self.helpers._account_for_missing_tags_in_chute_numbers()
        # Look for changes in the chute numbers and reset if necessary
        self.helpers._check_for_changes(detected_tags)
        # Clear the processed tags set for the next frame
        self.processed_tags.clear()


    def draw(self, frame: np.ndarray, tags: list, make_cutout: bool = False, straw_level: float = 25, use_cutout=False) -> np.ndarray:
        """
        Draws detected AprilTags on the frame with visual enhancements like lines indicating straw levels.
        Optionally creates a cutout of the processed frame.
        
        Parameters:
        -----------
        frame : np.ndarray
            The input frame to be processed.
        tags : list
            A list of detected AprilTags with their properties.
        make_cutout : bool, optional
            Whether to create a cutout of the frame (default is False).
        straw_level : float, optional
            The straw level percentage (default is 25).
        use_cutout : bool, optional
            Whether to return the cutout or the full frame (default is False).
        
        Returns:
        --------
        np.ndarray
            The frame with annotations, and optionally the cutout if requested.
        """
        try:
            # Step 1: Classify tags into number tags and chute tags
            number_tags, chute_tags = self.helpers._classify_tags(tags)
            if not number_tags or not chute_tags:
                return frame, None
            
            # Step 2: Draw detected tags on the frame
            frame = self.helpers._draw_tags(frame, number_tags, chute_tags)
            
            # Step 3: Draw straw level lines between number tags and chute tags
            frame = self.helpers._draw_level_lines(frame, number_tags, chute_tags, straw_level)
            
            # Step 4: Optionally create and return the cutout
            if make_cutout:
                return self.helpers._handle_cutouts(frame, chute_tags, use_cutout)

            return frame, None        
        except Exception as e:
            print(f"ERROR in draw: {e}")
            return frame, None

class RTSPStream(AprilDetector):
    """
    Wrapper class to detect AprilTags in a real-time video stream. The class inherits from the AprilDetector class and
    provides methods to receive frames from the video stream, detect AprilTags in the frames, draw the detected tags on
    the frame, and display the frame with the detected tags. The class also provides a method to detect AprilTags in a
    single frame.

    NOTE Threading is necessary here because we are dealing with an RTSP stream.
    """
    def __init__(self, detector, ids, credentials_path, od_model_name=None, object_detect=True, yolo_threshold=0.2, device="cpu", window=True, rtsp=True, make_cutout=False, use_cutout=False, detect_april=False, yolo_straw=False, yolo_straw_model="", with_predictor: bool = False, model_load_path: str = "models/vit_regressor/", regressor: bool = True, predictor_model: str = "vit", edges=True, heatmap=False) -> None:
        super().__init__(detector, ids, window, od_model_name, object_detect, yolo_threshold, device, yolo_straw=yolo_straw, yolo_straw_model=yolo_straw_model, with_predictor=with_predictor, model_load_path=model_load_path, regressor=regressor, predictor_model=predictor_model, edges=edges, heatmap=heatmap)
        self.rtsp = rtsp
        self.yolo_straw = yolo_straw
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
        self.threads = []
        self.information = self.helpers._initialize_information_dict()
        
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
    
    def get_pixel_to_straw_level(self, frame, straw_bbox):
        
        # make sure that all the tags are detected
        if len(self.chute_numbers) != 11:
            return frame, 0
        _, straw_cord,_ , _ = straw_bbox
        straw_cord = straw_cord[0].flatten()
        
        # Get angle of self.chute_numbers
        angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
        
        # Taking image height and width 
        height, width = frame.shape[:2]
        # get the image centers
        image_center = (width/2, height/2)

        rotation_arr = cv2.getRotationMatrix2D(image_center, float(angle), 1)

        abs_cos = abs(rotation_arr[0,0])
        abs_sin = abs(rotation_arr[0,1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        rotation_arr[0, 2] += bound_w/2 - image_center[0]
        rotation_arr[1, 2] += bound_h/2 - image_center[1]

        rotated_frame = cv2.warpAffine(frame, rotation_arr, (bound_w, bound_h))
        affine_warp = np.vstack((rotation_arr, np.array([0,0,1])))
        bbox_ = np.expand_dims(straw_cord.reshape(-1, 2), 1)
        bbox_ = cv2.perspectiveTransform(bbox_, affine_warp)
        straw_top = (bbox_[0][0][1] + bbox_[3][0][1])/2
        
        # Given the straw bbox, we need to calculate the straw level based on the center of each tag in the chute. We know that the id of each tag corresponds to the level of the chute, meaning 1 is 10%, 2 is 20% and so on. We need to find the two closest tags in the y-axis to the straw bbox and calculate the straw level based on the distance between the two tags.
        # We can do this by calculating the distance between the straw bbox and the center of each tag in the chute. We then sort the distances and find the two closest tags. We then calculate the distance between the straw bbox and the two closest tags and use this to calculate the straw level.
        distance_dict = {}
        for key, values in self.chute_numbers.items():
            distance = abs(straw_top - values[1])
            distance_dict[distance] = key
        
        # sort the dictionary by key
        distance_dict = dict(sorted(distance_dict.items()))
        # get the two closest tags
        tag0, tag1 = list(distance_dict.values())[:2]
        # Wee know that the tag with the lower y-value has to be the current level.
        first_closest_tag_id, second_closest_tag_id = min(tag0, tag1), max(tag0, tag1)
        # get the distance between the two closest tags
        y_first = self.chute_numbers[first_closest_tag_id][1]
        y_second = self.chute_numbers[second_closest_tag_id][1]
        # given the two y-values, take the y-value for straw_top and calculate the percentage of the straw level
        straw_level = ((y_first-straw_top) / (y_first-y_second) + first_closest_tag_id)*10
        return rotated_frame, straw_level
    
    def get_straw_to_pixel_level(self, straw_level):
        # We know that the self.chute_numbers are ordered from 0 to 10. We can use this to calculate the pixel value of the straw level
        # we know that each tag is 10% of the chute, meaning that the distance between each tag is 10% of the chute height. We can use 
        # this to calculate the pixel value of the straw level.
        # We can use the distance between the two closest tags to calculate the pixel value of the straw level.
        if len(self.chute_numbers) != 11:
            return 0,0
        # First we divide the straw level by 10 to get it on the same scale as the tag ids
        straw_level = straw_level / 10
        # We then get the two closest tags
        first_closest_tag_id, second_closest_tag_id = int(straw_level), int(straw_level) + 1
        # get the distance between the two closest tags
        y_first = self.chute_numbers[first_closest_tag_id][1]
        y_second = self.chute_numbers[second_closest_tag_id][1]
        # get the pixel value of the straw level
        excess = straw_level - int(straw_level)
        pixel_straw_level_y = y_first - (y_first - y_second) * excess
        pixel_straw_level_x = (self.chute_numbers[first_closest_tag_id][0] + self.chute_numbers[second_closest_tag_id][0]) / 2
        return pixel_straw_level_y, pixel_straw_level_x
    
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
                    print("Frame is None. Skipping...")
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
                    frame_drawn, cutout = self.draw(frame=frame.copy(), tags=self.tags.copy(), make_cutout=self.make_cutout, use_cutout=self.use_cutout)

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
                    results = "NA"
                # else:
                #     print("The cutout image is None and the object detection is not used.")
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
                            y_pixel, x_pixel = self.get_straw_to_pixel_level(straw_level)
                            # draw line on the frame with straw_level as the text
                            cv2.line(frame_drawn, (int(x_pixel), int(y_pixel)), (int(x_pixel)+100, int(y_pixel)), (0, 0, 255), 2)
                            cv2.putText(frame_drawn, f"{straw_level:.2f}%", (int(x_pixel)+110, int(y_pixel)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)                     
                            # Add the time taken for inference to the text
                            self.information["straw_level"]["text"] = f'(T2) Straw Level: {straw_level:.2f} %'
                            self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
                            self.information["prep"]["text"] = f'(T2) Image Prep. Time: {prep_time:.2f} s'                           
                    if self.object_detect:
                        frame_drawn = self.plot_boxes(results, frame_drawn, straw=False, straw_lvl=None, model_type="obb")
                elif self.yolo_straw:
                    output, inference_time = self.time_function(self.model.score_frame, frame)
                    if len(output[0]) != 0:
                        frame_drawn, straw_level = self.get_pixel_to_straw_level(frame_drawn, output)
                        frame_drawn = self.plot_boxes(output, frame_drawn, straw=True, straw_lvl=straw_level, model_type="obb")
                        self.information["straw_level"]["text"] = f'(T2) Straw Level: {straw_level:.2f} %'
                    self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
                else:
                    frame_drawn = cv2.resize(frame_drawn, (0, 0), fx=0.6, fy=0.6) # Resize the frame for display
                    cv2.imshow('Video', frame_drawn) # Display the frame
                    cv2.waitKey(1)
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
                cv2.waitKey(1)
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
                frame_drawn, cutout = self.draw(frame=frame.copy(), tags=self.tags.copy(), make_cutout=self.make_cutout, use_cutout=self.use_cutout)

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
                results = "NA"
            # else:
            #     print("The cutout image is None and the object detection is not used.")
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
                        y_pixel, x_pixel = self.get_straw_to_pixel_level(straw_level)
                        # draw line on the frame with straw_level as the text
                        cv2.line(frame_drawn, (int(x_pixel), int(y_pixel)), (int(x_pixel)+100, int(y_pixel)), (0, 0, 255), 2)
                        cv2.putText(frame_drawn, f"{straw_level:.2f}%", (int(x_pixel)+110, int(y_pixel)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)                     
                        # Add the time taken for inference to the text
                        self.information["straw_level"]["text"] = f'(T2) Straw Level: {straw_level:.2f} %'
                        self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
                        self.information["prep"]["text"] = f'(T2) Image Prep. Time: {prep_time:.2f} s'                           
                if self.object_detect:
                    frame_drawn = self.plot_boxes(results, frame_drawn)
            elif self.yolo_straw:
                output, inference_time = self.time_function(self.model.score_frame, frame)
                if len(output[0]) != 0:
                    frame_drawn, straw_level = self.get_pixel_to_straw_level(frame_drawn, output)
                    frame_drawn = self.plot_boxes(output, frame_drawn, straw=True, straw_lvl=straw_level, model_type="obb")
                    self.information["straw_level"]["text"] = f'(T2) Straw Level: {straw_level:.2f} %'
                self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
            else:
                frame_drawn = cv2.resize(frame_drawn, (0, 0), fx=0.6, fy=0.6) # Resize the frame for display
                cv2.imshow('Video', frame_drawn) # Display the frame
                cv2.waitKey(1)
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
            cv2.waitKey(1)
            # flush everything from memory to prevent memory leak
            frame = None
            results = None
            cutout_image = None
            output = None
            torch.cuda.empty_cache()
            
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
                import os
                # make sure the video path exists
                if not os.path.exists(video_path):
                    print("The video path does not exist.")
                    return
                self.cap = cv2.VideoCapture(video_path)
                print("START: Videofile loaded")
                if self.detect_april:
                    self.thread3 = threading.Thread(target=self.find_tags)
                    self.thread3.start()
                    self.threads.append(self.thread3)
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

    # video_path = "data/raw/Pin drum Chute 2_HKVision_HKVision_20241102105959_20241102112224_1532587042.mp4"
    video_path = "D:/HCAI/msc/strawml/data/special/Pin drum Chute 2_HKVision_HKVision_20241102112224_20241102113000_1532606664.mp4"
    # RTSPStream(detector, config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
    #            rtsp=False, # Only used when the stream is from an RTSP source
    #            make_cutout=False, object_detect=True, od_model_name="models/yolov11_obb_m8100btb_best.pt", yolo_threshold=0.2,
    #            detect_april=False,
    #            with_predictor=True, predictor_model='vit', model_load_path='models/vit_regressor/', regressor=True, edges=True, heatmap=False)(video_path=video_path)
    
    # ### YOLO PREDICTOR (VIDEOFILE)
    # RTSPStream(detector, config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
    #         rtsp=False , # Only used when the stream is from an RTSP source
    #         make_cutout=True, use_cutout=False, object_detect=False, od_model_name="models/yolov11-chute-detect-obb.pt", yolo_threshold=0.2,
    #         detect_april=True, yolo_straw=True, yolo_straw_model="models/yolov11-straw-detect-obb.pt",
    #         with_predictor=False , predictor_model='vit', model_load_path='models/vit_regressor/', regressor=True, edges=True, heatmap=False)(video_path=video_path)
    
    # CONVNEXTV2 PREDICTOR
    # RTSPStream(detector, config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
    #         rtsp=True , # Only used when the stream is from an RTSP source
    #         make_cutout=False, use_cutout=False, object_detect=True, od_model_name="models/yolov11-chute-detect-obb.pt", yolo_threshold=0.2,
    #         detect_april=True, yolo_straw=False, yolo_straw_model="models/yolov11-straw-detect-obb.pt",
    #         with_predictor=True , predictor_model='convnextv2', model_load_path='models/convnext_regressor/', regressor=True, edges=False, heatmap=False)()
    
    # YOLO PREDICTOR
    RTSPStream(detector, config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
        rtsp=True , # Only used when the stream is from an RTSP source
        make_cutout=True, use_cutout=False, object_detect=False, od_model_name="models/yolov11-chute-detect-obb.pt", yolo_threshold=0.2,
        detect_april=True, yolo_straw=True, yolo_straw_model="models/yolov11-straw-detect-obb.pt",
        with_predictor=False , predictor_model='convnextv2', model_load_path='models/convnext_regressor/', regressor=True, edges=False, heatmap=False)()


# TODO: look into why the tags are found, but not plottet for the chute-numbers.