from __init__ import *
# Misc.
import os
import cv2
import time
import h5py
import torch
import queue
import psutil
import keyboard
import threading
import numpy as np
from collections import deque
from typing import Tuple, Optional, Any
from torchvision.transforms import v2 as transforms
from scipy import ndimage

# Model imports
import timm
import pupil_apriltags

## File imports
from strawml.models.chute_finder.yolo import ObjectDetect
import strawml.models.straw_classifier.cnn_classifier as cnn
import strawml.models.straw_classifier.feature_model as feature_model
from strawml.visualizations.utils_stream import AprilDetectorHelpers, AsyncStreamThread, time_function

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
        self.q = queue.LifoQueue()
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

    def plot_straw_level(self, frame_drawn, line_start, line_end, straw_level, color=(127, 0, 255)) -> np.ndarray:
        cv2.line(frame_drawn, line_start, line_end, color, 2)
        if type(straw_level) == str:
            cv2.putText(frame_drawn, f"{straw_level}", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_drawn, f"{straw_level:.2f}%", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return frame_drawn
    
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
                    cv2.line(frame, (x1, y1), (x4, y4), (127,0,255), 2)
                    if type(straw_lvl) == str:
                        cv2.putText(frame, f"{straw_lvl}", (x1+10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 0, 255), 2, cv2.LINE_AA)
                    else:                        
                        cv2.putText(frame, f"{straw_lvl:.2f} %", (x1+10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 0, 255), 2, cv2.LINE_AA)
                else:                        
                    # draw lines between the cornersq
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
            if len(number_tags) == 0 or len(chute_tags) == 0:
                return frame, None
            
            # Step 2: Draw detected tags on the frame
            frame_drawn = self.helpers._draw_tags(frame, number_tags, chute_tags)
            
            # Step 3: Draw straw level lines between number tags and chute tags
            frame_drawn = self.helpers._draw_level_lines(frame_drawn, number_tags, chute_tags, straw_level)
            
            # Step 4: Optionally create and return the cutout
            if make_cutout:
                return self.helpers._handle_cutouts(frame_drawn, chute_tags, use_cutout)

            return frame_drawn, None        
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
    def __init__(self, record, record_threshold, detector, ids, credentials_path, od_model_name=None, object_detect=True, yolo_threshold=0.2, device="cuda", window=True, rtsp=True, make_cutout=False, use_cutout=False, detect_april=False, yolo_straw=False, yolo_straw_model="", with_predictor: bool = False, model_load_path: str = "models/vit_regressor/", regressor: bool = True, predictor_model: str = "vit", edges=True, heatmap=False, smoothing:bool=True) -> None:
        super().__init__(detector, ids, window, od_model_name, object_detect, yolo_threshold, device, yolo_straw=yolo_straw, yolo_straw_model=yolo_straw_model, with_predictor=with_predictor, model_load_path=model_load_path, regressor=regressor, predictor_model=predictor_model, edges=edges, heatmap=heatmap)
        self.record = record
        self.recording_req = False
        self.record_threshold = record_threshold
        if record:
            self.recording_queue = queue.Queue()
            self.scada_smoothing_queue = deque(maxlen=5)
        self.straw_smoothing_queue = deque(maxlen=5)
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
        self.smoothing = smoothing
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
            username, password, ip, rtsp_port = f.read().splitlines()
        rtsp_url = f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101"
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_FPS, 25)
        # cap.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
        return cap
  
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
        self._setup_display()
        self._process_frames_from_queue()

    def display_frame_from_videofile(self) -> None:
        """
        Display the frames with the detected AprilTags.
        """
        self._setup_display()
        self._process_frames_from_videofile()

    def _setup_display(self) -> None:
        """Set up display settings and initialize variables for both modes."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_color = (255, 0, 0)  # Blue color for text
        self.box_color = (255, 255, 255)  # White color for box
        self.frame_count = 0
        self.start_time = time.time()
        if self.record:
            self.since_last_save = time.time()

    def _process_frames_from_queue(self) -> None:
        """Process frames from the queue for display."""
        while not self.should_abort_immediately:
            self._reset_information()
            if not self.q.empty():
                frame = self.q.get() 
                self._process_frame(frame, from_videofile=False)

    def _process_frames_from_videofile(self) -> None:
        """Process frames from a video file."""
        while True:
            self._reset_information()
            success, frame = self.cap.read()
            if not success:
                break

            # rotate the frame by 30 degrees
            # frame = ndimage.rotate(frame, 30, reshape=True)
            self._process_frame(frame, from_videofile=True)

    def _process_frame(self, frame: np.ndarray, from_videofile: bool) -> None:
        """Process a single frame for display."""
        if frame is None:
            print("Frame is None. Skipping...")
            return
        # Clear the queue in RTSP mode to avoid lag
        if not from_videofile and self.rtsp:
            self.q.queue.clear()
        frame_time = time.time()
        
        display_scada_line = False
        if self.record:
            try:
                # Get scada
                sensor_scada_data = self.scada_thread.get_recent_value()
                sensor_scada_data = self.helpers._smooth_level(sensor_scada_data, 'scada')
                self.information["scada_smooth"]["text"] = f'(T2) Smoothed Scada Level: {sensor_scada_data:.2f}%'
                # Get pixel values for scada
                scada_pixel_values = self.helpers._get_straw_to_pixel_level(sensor_scada_data)
                # Record sensor data if enabled
                self.recording_req =  (frame_time - self.since_last_save >= self.record_threshold)
                if self.recording_req:
                    self.prediction_dict = {}
                    scada_pixel_values_ = (scada_pixel_values[0], scada_pixel_values[1])
                    # Get angle of self.chute_numbers
                    angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
                    line_start = (int(scada_pixel_values_[0]), int(scada_pixel_values_[1]))
                    line_end = (int(scada_pixel_values_[0])+300, int(scada_pixel_values_[1]))
                    line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
                    self.prediction_dict["scada"] = {sensor_scada_data: [line_start, line_end]}
                    # Reset the time
                    self.since_last_save = time.time()
                display_scada_line = True
            except Exception as e:
                print(f'Error while trying to get scada data: {e}')
                    
        # Lock and set the frame for thread safety
        with self.lock:
            self.frame = frame

        # Process frame (AprilTags, Object Detection, Predictor, etc.)
        frame_drawn = self._process_frame_content(frame)

        # Draw sensor_scada_data on frame based on scada_pixel_values
        if display_scada_line and self.record:
            if type(scada_pixel_values) != str:
                scada_pixel_values = (scada_pixel_values[0], scada_pixel_values[1])
                
                # Get angle of self.chute_numbers
                angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
                # print("angle: ", np.degrees(angle))

                line_start = (int(scada_pixel_values[0]), int(scada_pixel_values[1]))
                line_end = (int(scada_pixel_values[0])+300, int(scada_pixel_values[1]))
                
                line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
                
                cv2.line(frame_drawn, line_start, line_end, (32,165,218), 2)
                cv2.putText(frame_drawn, f"{sensor_scada_data:.2f}%", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (32,165,218), 2, cv2.LINE_AA)

        # Update FPS and resource usage information
        self._update_information(frame_time)

        # Display frame and overlay text
        self._display_frame(frame_drawn)
    
        if self.recording_req:
            self._save_frame

        # Clean up resources
        torch.cuda.empty_cache()
    
    def _save_frame(self) -> None:
        path = "data/recording/"
        if not os.path.exists(path):
            os.makedirs(path)
        path += 'recording.hdf5'
        if os.path.exists(path):
            file_stats = os.stat(path)
            if file_stats.st_size / 1e9 > 100:
                return
        timestamp = time.time()
        encoded_frame = cv2.imencode('.jpg', self.frame)[1]
        encoded_frame = np.asarray(encoded_frame)
        with h5py.File(path, 'a') as hf:
            time_group = hf.create_group(f'{timestamp}')
            time_group.create_dataset('image', data=encoded_frame)
            for key, value in self.prediction_dict.items():
                predict_group = time_group.create_group(key)
                t1, t2 = list(value.items())[0]
                if key == 'attr.':
                    t1_name = 'interpolated'
                    t2_name = 'tags'
                else:
                    t1_name = 'percent'
                    t2_name = 'pixel'
                predict_group.create_dataset(t1_name, data=t1)
                predict_group.create_dataset(t2_name, data=t2)
            hf.close()

    def _reset_information(self) -> None:
        """Reset the information dictionary for each frame."""
        for key in self.information.keys():
            if key == "april":
                continue
            self.information[key]["text"] = ""

    def _process_frame_content(self, frame: np.ndarray) -> np.ndarray:
        """Handle specific processing like AprilTags, Object Detection, etc."""
        # Apriltags
        if self.detect_april and self.tags is not None:
            frame_drawn, cutout = self.draw(frame=frame.copy(), tags=self.tags.copy(), make_cutout=self.make_cutout, use_cutout=self.use_cutout)
        else:
            frame_drawn, cutout = frame, None

        # Object Detection
        if cutout is not None:
            frame = cutout
            results = None
        elif self.object_detect:
            results, OD_time = time_function(self.OD.score_frame, frame)
            # Make sure the results are not empty
            if len(results[0]) == 0:
                results = "NA"
            self.information["od"]["text"] = f'(T2) OD Time: {OD_time:.2f} s' if results else "NA"
        else:
            results = "NA"

        # Predictor of the straw level
        if results != "NA":
            if self.with_predictor:
                frame_drawn = self._process_predictions(frame, results, frame_drawn)
            if self.object_detect:
                frame_drawn = self.plot_boxes(results, frame_drawn, straw=False, straw_lvl=None, model_type="obb")
        elif self.yolo_straw:
            output, inference_time = time_function(self.model.score_frame, frame)
            # If the output is not empty, we can plot the boxes and get the straw level
            if len(output[0]) != 0:
                # Extract the straw level from the pixel values of the bbox
                straw_level = self.helpers._get_pixel_to_straw_level(frame_drawn, output)
                if self.smoothing:
                    if straw_level == "NA": # If the straw level is not detected, we add None to the previous straw level smoothing predictions
                        straw_level = self.helpers._smooth_level(None, 'straw')
                    else: # otherwise we add the detected straw level to the smoothing predictions and get the smoothed straw level
                        straw_level = self.helpers._smooth_level(straw_level, 'straw')
                
                # Since the new straw level might be a smoothed value, we need to update the pixel values of the straw level. We do this everytime to ensure that the overlay is based on the same pixel values all the time. Otherwise the overlay would shift from being based on the bbox pixel values vs. based on the tags.
                x_pixel, y_pixel = self.helpers._get_straw_to_pixel_level(straw_level)
                line_start = (int(x_pixel), int(y_pixel))
                line_end = (int(x_pixel) + 300, int(y_pixel))
                # Plot the boxes and straw level on the frame
                frame_drawn = self.plot_straw_level(frame_drawn, line_start, line_end, straw_level)
                
                if type(straw_level) == str:
                    self.information["straw_smooth"]["text"] = f'(T2) Smoothed Straw Level: {straw_level}'
                else:
                    self.information["straw_smooth"]["text"] = f'(T2) Smoothed Straw Level: {straw_level:.2f} %'
                self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
            else:
                if self.smoothing:
                    straw_level = self.helpers._smooth_level(0, 'straw')
                else:
                    straw_level = 0
                if self.recording_req:
                    # if no bbox is detected, we add 0 to the previous straw level smoothing predictions
                    angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
                    self.helpers._save_tag_0(angle)
                self.information["straw_smooth"]["text"] = f'(T2) Smoothed Straw Level: {straw_level:.2f} %'
                self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
        return frame_drawn

    def _process_predictions(self, frame, results, frame_drawn):
        """Run model predictions and update overlay."""
        cutout_image, prep_time = time_function(self.helpers._prepare_for_inference, frame, results)
        if cutout_image is not None:
            if self.regressor:
                if self.predictor_model != 'cnn':
                    output, inference_time = time_function([self.model, self.regressor_model], cutout_image.to(self.device))
                else:
                    output, inference_time = time_function(self.model, cutout_image.to(self.device))
                # detach the output from the device and get the predicted value
                output = output.detach().cpu()
                straw_level = output[0].item()*100
            else:
                output, inference_time = time_function(self.model, cutout_image.to(self.device)) 
                # detach the output from the device and get the predicted value
                output = output.detach().cpu()
                _, predicted = torch.max(output, 1)
                straw_level = predicted[0]*10
            
            # We smooth the straw level
            straw_level = self.helpers._smooth_level(straw_level, 'straw')
            
            x_pixel, y_pixel = self.helpers._get_straw_to_pixel_level(straw_level)      
            if self.recording_req:
                self.prediction_dict["predicted"] = {straw_level: (x_pixel, y_pixel)}
            # Draw line and text for straw level
            cv2.line(frame_drawn, (int(x_pixel), int(y_pixel)), (int(x_pixel) + 300, int(y_pixel)), (92, 92, 205), 2)
            cv2.putText(frame_drawn, f"{straw_level:.2f}%", (int(x_pixel) + 110, int(y_pixel)), cv2.FONT_HERSHEY_SIMPLEX, 1,  (92, 92, 205), 2, cv2.LINE_AA)
            self.information["straw_smooth"]["text"] = f'(T2) Smoothed Straw Level: {straw_level:.2f} %'
            self.information["model"]["text"] = f'(T2) Inference Time: {inference_time:.2f} s'
            self.information["prep"]["text"] = f'(T2) Image Prep. Time: {prep_time:.2f} s'
        return frame_drawn

    def _update_information(self, frame_time):
        """Update frame processing information."""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        self.frame_count += 1
        self.information["FPS"]["text"] = f'(T2) FPS: {fps:.2f}'
        self.information["frame_time"]["text"] = f'(T2) Total Frame Time: {time.time() - frame_time:.2f} s'
        self.information["RAM"]["text"] = f'(TM) Total RAM Usage: {np.round(psutil.virtual_memory().used / 1e9, 2)} GB'
        self.information["CPU"]["text"] = f'(TM) Total CPU Usage: {psutil.cpu_percent()} %'

    def _display_frame(self, frame_drawn, with_text=True, fx=0.6, fy=0.6):
        """Resize and display the processed frame."""
        frame_drawn = cv2.resize(frame_drawn, (0, 0), fx=fx, fy=fy)

        if with_text:
            # Draw the text on the frame
            for i, (key, val) in enumerate(self.information.items()):
                # Get the text size
                if val["text"] == "":
                    continue
                font_scale = val["font_scale"]
                font_thickness = val["font_thicknesss"]
                (text_width, text_height), baseline = cv2.getTextSize(val["text"], self.font, font_scale, font_thickness)
                pos = val["position"]
                box_coords = ((pos[0], pos[1] - text_height - baseline), (pos[0] + text_width, pos[1] + baseline)) # Calculate the box coordinates
                cv2.rectangle(frame_drawn, box_coords[0], box_coords[1], self.box_color, cv2.FILLED) # Draw the white box                    
                cv2.putText(frame_drawn, val["text"], pos, self.font, font_scale, self.text_color, font_thickness, cv2.LINE_AA) # Draw the text over the box

        cv2.imshow('Video', frame_drawn)
        cv2.waitKey(1)

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
                    self.thread1 = threading.Thread(target=self.find_tags)
                    self.thread1.start()
                    self.threads.append(self.thread1)
                if self.record:
                    self.scada_thread = AsyncStreamThread(server_keys='data/opcua_server.txt')
                    self.scada_thread.start()
                    self.threads.append(self.scada_thread)
                self.display_frame_from_videofile()
            else:
                if cap is not None:
                    self.cap = cap
                print("START: Threads and resources...")
                if self.record:
                    self.scada_thread = AsyncStreamThread(server_keys='data/opcua_server.txt')
                    self.scada_thread.start()
                    self.threads.append(self.scada_thread)
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

    # video_path = "data/raw/videos/2024-11-21-16h02m03s.mp4"
    # video_path = "C:/Users/ikaos/OneDrive/Desktop/strawml/data/raw/stream-2024-09-23-10h11m28s.mp4"
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
    
    ### CONVNEXTV2 PREDICTOR
    # RTSPStream(record=False, record_threshold=5, detector=detector, ids=config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
    #         rtsp=True , # Only used when the stream is from an RTSP source
    #         make_cutout=False, use_cutout=False, object_detect=True, od_model_name="models/yolov11-chute-detect-obb.pt", yolo_threshold=0.2,
    #         detect_april=True, yolo_straw=False, yolo_straw_model="models/yolov11-straw-detect-obb.pt",
    #         with_predictor=True , predictor_model='convnextv2', model_load_path='models/convnext_regressor/', regressor=True, edges=False, heatmap=False)()
    
    # ### YOLO PREDICTOR
    RTSPStream(record=False, record_threshold=5, detector=detector, ids=config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
        rtsp=True , # Only used when the stream is from an RTSP source
        make_cutout=True, use_cutout=False, object_detect=False, od_model_name="models/yolov11-chute-detect-obb.pt", yolo_threshold=0.2,
        detect_april=True, yolo_straw=True, yolo_straw_model="models/obb_best.pt",
        with_predictor=False , predictor_model='convnextv2', model_load_path='models/convnext_regressor/', regressor=True, edges=False, heatmap=False,
        device='cuda')()