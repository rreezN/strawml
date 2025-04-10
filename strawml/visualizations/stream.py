from __init__ import *
# Misc.
import os
import cv2
import csv
import time
import h5py
import json
import timm
import torch
import queue
import psutil
import pickle
import keyboard
import threading
import numpy as np
from tqdm import tqdm
import pupil_apriltags
from collections import deque
from datetime import datetime
from pupil_apriltags import Detector
from typing import Tuple, Optional, Any
from argparse import ArgumentParser, Namespace
from torchvision.transforms import v2 as transforms

## File imports
from strawml.models.chute_finder.yolo import ObjectDetect
from strawml.data.make_dataset import decode_binary_image
import strawml.models.straw_classifier.cnn_classifier as cnn
import strawml.models.straw_classifier.feature_model as feature_model
from strawml.visualizations.utils_stream import AprilDetectorHelpers, AsyncStreamThread, time_function, TagGraphWithPositionsCV

torch.use_deterministic_algorithms(True)

class AprilDetector:
    """
    NOTE This class is highly customized to detect AprilTags in the chute system af meliora bio. 

    AprilTag detector class that uses the pupil_apriltags library to detect AprilTags in a frame. The class is designed
    to be used in conjunction with the RTSPStream class to detect AprilTags in a real-time video stream. The class
    provides methods to detect AprilTags in a frame, draw the detected tags on the frame, and given a predicted straw level
    value performs inverse linear interpolation to get the corresponding pixel value on the frame.
    """
    def __init__(self, detector: pupil_apriltags.bindings.Detector, ids: dict, window: bool=False, od_model_name=None, object_detect=True, yolo_threshold=0.5, device="cuda", frame_shape: tuple = (1440, 2560), yolo_straw=False, yolo_straw_model="models/yolov11-straw-detect-obb.pt", with_predictor: bool = False, model_load_path: str = "models/vit_regressor/", regressor: bool = True, predictor_model: str = "vit", edges=True) -> None:
        self.helpers = AprilDetectorHelpers(self)  # Instantiate the helper class
        self.lock = threading.Lock()
        self.detector = detector
        self.ids = ids
        self.od_model_name = od_model_name
        self.window = window
        self.device = device
        self.yolo_threshold = yolo_threshold
        self.edges = edges
        self.yolo_straw = yolo_straw
        self.with_predictor = with_predictor
        self.regressor = regressor
        self.model_load_path = model_load_path
        self.predictor_model = predictor_model
        self.frame_shape = frame_shape
        self.yolo_model = None
        self.model = None 
        self.q = queue.LifoQueue()
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.camera_params = self.helpers._load_camera_params()
        self.tags = {}
        self.chute_numbers = {}
        self.tag_connections = self.helpers._get_tag_connections()
        self.processed_tags = set()  # Track tags that have already been re-centered
        self.inferred_tags = set()
        self.detected_tags = []
        self.object_detect = object_detect
        self.tag_graph = TagGraphWithPositionsCV(self.tag_connections, self.helpers)

        # Setup object detection if enabled
        if self.object_detect:
            self.setup_object_detection(od_model_name)
        # Setup prediction model if enabled YOLO is prioritized over the predictor
        if yolo_straw:
            self.setup_yolo_straw(yolo_straw_model)
        if with_predictor:
            self.setup_predictor()

    def setup_object_detection(self, od_model_name: str) -> None:
        """Setup object detection model."""
        self.OD = ObjectDetect(od_model_name, yolo_threshold=self.yolo_threshold, device=self.device, verbose=False)

    def setup_yolo_straw(self, yolo_straw_model: str) -> None:
        """Setup YOLO straw model."""
        self.yolo_model = ObjectDetect(model_name=yolo_straw_model, yolo_threshold=self.yolo_threshold, device=self.device, verbose=False)

    def setup_predictor(self) -> None:
        """Setup the predictor model and related transformations."""
        self.mean, self.std = self.helpers._load_normalisation_constants()
        self.img_unnormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(self.mean, self.std)],
            std=[1 / s for s in self.std]
        )
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])

        self.normalise = transforms.Normalize(mean=self.mean, std=self.std)
        
        num_classes = 1 if self.regressor else 11
        input_channels = 3 + int(self.edges)

        # Model selection based on predictor_model
        self.model, image_size = self.load_predictor_model(input_channels, num_classes)
        self.resize = transforms.Resize(image_size)

        if self.regressor:
            self.setup_regressor(image_size, input_channels)
        else:
            self.regressor_model = None
            self.model.load_state_dict(torch.load(self.model_load_path, weights_only=True))
            self.model.to(self.device)
            self.model.eval()

    def load_predictor_model(self, input_channels: int, num_classes: int, use_sigmoid: bool = False, image_size: tuple = (672, 208)) -> torch.nn.Module:
        """Load the appropriate model based on the predictor_model string."""
        model = None
        match self.predictor_model:
            case 'cnn':
                model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels, output_size=num_classes, use_sigmoid=use_sigmoid)
            case 'convnextv2':
                model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', in_chans=input_channels, num_classes=num_classes)
            case 'convnext':
                model = timm.create_model('convnext_small.in12k_ft_in1k_384', in_chans=input_channels, num_classes=num_classes)
            case 'vit':
                model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', in_chans=input_channels, num_classes=num_classes, img_size=image_size)
            case 'eva02':
                model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', in_chans=input_channels, num_classes=num_classes, img_size=image_size)
            case 'caformer':
                model = timm.create_model('caformer_m36.sail_in22k_ft_in1k_384', in_chans=input_channels, num_classes=num_classes, img_size=image_size)
        
        
        model.load_state_dict(torch.load(f'{self.model_load_path}/{self.predictor_model}_feature_extractor.pth', weights_only=True))
        return model, image_size

    def setup_regressor(self, image_size: tuple, input_channels: int, use_sigmoid: bool = False, num_hidden_layers: int = 0, num_neurons: int = 512) -> None:
        """Setup the regressor model."""
        if self.predictor_model != 'cnn':
            features = self.model.forward_features(torch.randn(1, input_channels, image_size[0], image_size[1]))
            feature_size = torch.flatten(features, 1).shape[1]
            self.regressor_model = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1, use_sigmoid=use_sigmoid, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons)
            
            self.model.load_state_dict(torch.load(f'{self.model_load_path}/{self.predictor_model}_feature_extractor.pth', weights_only=True))
            self.regressor_model.load_state_dict(torch.load(f'{self.model_load_path}/{self.predictor_model}_regressor.pth', weights_only=True))
            self.regressor_model.to(self.device)
            self.regressor_model.eval()
        else:
            self.model.load_state_dict(torch.load(self.model_load_path, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def plot_straw_level(self, frame_drawn, line_start, line_end, straw_level, color=(225, 105, 65)) -> np.ndarray:
        # fist check if the straw level is None
        if straw_level is None or line_start[0] is np.nan:
            return frame_drawn
        cv2.line(frame_drawn, line_start, line_end, color, 2)
        if straw_level is None:
            cv2.putText(frame_drawn, "NA", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_drawn, f"{straw_level:.2f}%", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return frame_drawn
    
    def plot_boxes(self, results, frame, model_type: str = 'obb', label=None):
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

        # Account for the case where there are no detections and if there are more than 1 detection
        if n > 1:
            n=1

        for i in range(n):  
            # plot polygon around the object based on the coordinates cord
            if 'obb' in model_type:
                x1, y1, x2, y2, x3, y3, x4, y4 = cord[i].flatten()
                x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
                      
                # draw lines between the cornersq
                cv2.line(frame, (x1, y1), (x2, y2), (138,43,226), 2)
                cv2.line(frame, (x2, y2), (x3, y3), (138,43,226), 2)
                cv2.line(frame, (x3, y3), (x4, y4), (138,43,226), 2)
                cv2.line(frame, (x4, y4), (x1, y1), (138,43,226), 2)
                if label is None:
                    label = self.class_to_label(labels[i])
                # plot label on the object
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                x1, y1, x2, y2 = cord[i].flatten()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # plot label on the object
                if label is None:
                    label = self.class_to_label(labels[i])
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.OD.classes[int(x)]

    def detect(self, frame: np.ndarray, testing: bool = False) -> list:
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
        # update the tag graph
        self.tag_graph.update_init(self.tags, self.inferred_tags)
        # account for missing numbers in the tag graph
        self.tag_graph.account_for_missing_tags()
        # extract the tags from the tag graph
        self.tags, self.inferred_tags = self.tag_graph.get_tags()
        if testing:
            return detected_tags, self.tags, self.inferred_tags

    def draw(self, frame: np.ndarray, tags: list, make_cutout: bool = False, use_cutout=False) -> np.ndarray:
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
            frame_ = frame.copy()
            
            # Step 2: Draw detected tags on the frame
            frame_drawn = self.helpers._draw_tags(frame, number_tags, chute_tags)

            # Step 3: Draw straw level lines between number tags and chute tags
            frame_drawn = self.helpers._draw_level_lines(frame_drawn, number_tags, chute_tags)

            # Step 4: Optionally create and return the cutout
            if make_cutout:
                return self.helpers._handle_cutouts(frame_drawn, frame_, use_cutout)
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
    def __init__(self, detector, ids, credentials_path, od_model_name=None, object_detect=True, yolo_threshold=0.2, device="cuda", window=True, make_cutout=False, use_cutout=False, detect_april=False, yolo_straw=False, yolo_straw_model="", with_predictor: bool = False, model_load_path: str = "models/vit_regressor/", regressor: bool = True, predictor_model: str = "vit", edges=True, smoothing:bool=True, mode: str | None = None, pull_opcua: bool = False, write_opcua: bool = False, write_csv: bool = False) -> None:
        super().__init__(detector, ids, window, od_model_name, object_detect, yolo_threshold, device, yolo_straw=yolo_straw, yolo_straw_model=yolo_straw_model, with_predictor=with_predictor, model_load_path=model_load_path, regressor=regressor, predictor_model=predictor_model, edges=edges)
        self.scada_smoothing_queue = deque(maxlen=5)
        self.yolo_smoothing_queue = deque(maxlen=5)
        self.predictor_smoothing_queue = deque(maxlen=5)
        self.yolo_straw = yolo_straw
        self.wanted_tags = ids
        self.cap = self.create_capture(credentials_path)
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
        self.last_save_timestamp = 0
        self.mode = mode
        self.pull_opcua = pull_opcua
        self.write_opcua = write_opcua
        self.write_csv = write_csv
        self.prediction_dict = {}
        
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

    def _warm_up_for_apriltags(self, path, idx, nr:int=50) -> np.ndarray:
        self.helpers._reset_tags()
        with h5py.File(path, 'r') as hf:
            timestamps = list(hf.keys())

            if "frame" == timestamps[0].split("_")[0]:
                timestamps = sorted(timestamps, key=lambda x: int(x.split('_')[1]))
            else:
                timestamps = sorted(timestamps, key=lambda x: float(x))

            for timestamp in timestamps[idx:idx+nr]:
                frame = hf[timestamp]['image'][()]
                frame = decode_binary_image(frame)
                # change from bgr to rgb
                self.frame = frame

    def _setup_display(self) -> None:
        """Set up display settings and initialize variables for both modes."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_color = (255, 0, 0)  # Blue color for text
        self.box_color = (255, 255, 255)  # White color for box
        self.frame_count = 0
        self.start_time = time.time()
        self.yolo_color = (225, 105, 65)
        self.scada_color = (32, 165, 218)
        self.predictor_color = (80, 73, 149)

        self.information["scada_level"]["color"] = self.scada_color
        self.information["scada_smooth"]["color"] = self.scada_color
        self.information["yolo_level"]["color"] = self.yolo_color
        self.information["yolo_smooth"]["color"] = self.yolo_color
        self.information["predictor_level"]["color"] = self.predictor_color
        self.information["predictor_smooth"]["color"] = self.predictor_color

    def _process_frames_from_queue(self) -> None:
        """Process frames from the queue for display."""
        while not self.should_abort_immediately:
            self._reset_information()
            if not self.q.empty():
                frame = self.q.get() 
                self._process_frame(frame, from_videofile=False)

    def _process_frame(self, frame: np.ndarray, from_videofile: bool) -> None:
        """Process a single frame for display."""
        if frame is None:
            print("Frame is None. Skipping...")
            return
        # Lock and set the frame for thread safety
        with self.lock:
            self.frame = frame

        # Clear the queue in RTSP mode to avoid lag
        self.q.queue.clear()

        frame_time = time.time()

        # Find Apriltags
        if self.detect_april and self.tags is not None:
            frame_drawn, cutout = self.draw(frame=frame.copy(), tags=self.tags.copy(), make_cutout=self.make_cutout, use_cutout=self.use_cutout)
        else:
            frame_drawn, cutout = frame, None
        
        # If not tags have been found we go to the next frame
        if len(self.tags) == 0:
            self._display_frame(frame_drawn)
            return
        
        # Get scada data
        display_scada_line = self.pull_opcua
        if self.pull_opcua and self.scada_thread is not None:
            try:
                sensor_scada_data = self.scada_thread.get_recent_value()
                line_start, line_end, sensor_scada_data, scada_pixel_values = self._retrieve_scada_data(sensor_scada_data)
                self.prediction_dict = {}
                self.prediction_dict["scada"] = {sensor_scada_data: [line_start, line_end]}
                display_scada_line = True
            except Exception as e:
                print(f'Error while trying to get scada data: {e}')
                    
        # Process frame (Object Detection, Predictor, etc.)
        frame_drawn = self._process_frame_content(frame, frame_drawn, cutout)

        # Draw sensor_scada_data on frame based on scada_pixel_values
        if display_scada_line and scada_pixel_values[0] is not None:
            cv2.line(frame_drawn, line_start, line_end, (32,165,218), 2)
            cv2.putText(frame_drawn, f"{sensor_scada_data:.2f}%", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (32,165,218), 2, cv2.LINE_AA)

        # Write to OPCUA if enabled
        if self.write_opcua and self.scada_thread is not None:
            try:
                if self.yolo_straw:
                    if self.smoothing:
                        predicted_value = list(self.prediction_dict["yolo_smooth"].keys())[0]
                    else:
                        predicted_value = list(self.prediction_dict["yolo"].keys())[0]
                elif self.with_predictor:
                    if self.smoothing:
                        predicted_value = list(self.prediction_dict["convnext_smooth"].keys())[0]
                    else:
                        predicted_value = list(self.prediction_dict["convnext"].keys())[0]
                self.scada_thread.write_scada_data(predicted_value)
            except Exception as e:
                print(f'Error while trying to write scada data: {e}')
        
        # Write to CSV if enabled
        if self.write_csv:
            self.log_prediction()
        
        # Update FPS and resource usage information
        self._update_information(frame_time)

        # Display frame and overlay text
        self._display_frame(frame_drawn)

        # Clean up resources
        torch.cuda.empty_cache()

    def log_prediction(self, csv_path: str = "data/logs/predictions.csv") -> None:
        """Log predictions to a CSV file."""
        file_exists = os.path.isfile(csv_path)
        
        timestamp = datetime.now().isoformat()
        model = "yolo" if self.yolo_straw else "convnext"
        if model == "yolo":
            if self.smoothing:
                prediction = list(self.prediction_dict["yolo_smooth"].keys())[0]
            else:
                prediction = list(self.prediction_dict["yolo"].keys())[0]
        else:
            if self.smoothing:
                prediction = list(self.prediction_dict["convnext_smooth"].keys())[0]
            else:
                prediction = list(self.prediction_dict["convnext"].keys())[0]
        
        if self.pull_opcua and self.scada_thread is not None:
            scada_reading = list(self.prediction_dict["scada"].keys())[0]
        else:
            scada_reading = None
        
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                # Write header if file does not exist
                writer.writerow(['timestamp', 'model', 'prediction', 'scada_reading'])
            writer.writerow([timestamp, model, prediction, scada_reading])
    
    def _retrieve_scada_data(self, sensor_scada_data, time_stamp = None) -> None:
        # Get scada
        if self.smoothing:
            sensor_scada_data = self.helpers._smooth_level(sensor_scada_data, 'scada', time_stamp = time_stamp)
            self.information["scada_smooth"]["text"] = f'(T2) Smoothed Scada Level: {sensor_scada_data:.2f}%'
        # Get pixel values for scada
        scada_pixel_values = self.helpers._get_straw_to_pixel_level(sensor_scada_data)
        if scada_pixel_values[0] is not np.nan:
            # Record sensor data if enabled
            scada_pixel_values_ = (scada_pixel_values[0], scada_pixel_values[1])
            # Get angle of self.chute_numbers
            angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
            line_start = (int(scada_pixel_values_[0]), int(scada_pixel_values_[1]))
            line_end = (int(scada_pixel_values_[0])+300, int(scada_pixel_values_[1]))
            line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
        else:
            line_start, line_end = (np.nan, np.nan), (np.nan, np.nan)
        return line_start, line_end, sensor_scada_data, scada_pixel_values

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

    def _process_frame_content(self, frame: np.ndarray, frame_drawn: np.ndarray, cutout, results=None, time_stamp = None) -> np.ndarray:
        """Handle specific processing like AprilTags, Object Detection, etc."""
        with torch.no_grad():
            # Object Detection
            if cutout is not None:
                frame = cutout
                results = results 
            elif self.object_detect:
                # _frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results, OD_time = time_function(self.OD.score_frame, frame.copy())
                # Make sure the results are not empty
                if len(results[0]) == 0:
                    results = "NA"
                else:
                    self.prediction_dict["yolo_cutout"] = results[1][0].flatten().cpu().detach().numpy()
                self.information["od"]["text"] = f'(T2) OD Time: {OD_time:.2f} s'
            else:
                results = results
            # Predictor of the straw level
            if self.with_predictor:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return self._predictor_model(frame, frame_drawn, results, time_stamp, cutout=cutout)
            elif self.yolo_straw:
                return self._yolo_model(frame, frame_drawn, cutout, time_stamp)
                
    def _predictor_model(self, frame: np.ndarray, frame_drawn: np.ndarray, results: list, time_stamp=None, cutout=None) -> None:
        if results != "NA":
            frame_drawn = self._process_predictions(frame, results, frame_drawn, time_stamp = time_stamp, cutout=cutout)
            if self.object_detect:
                frame_drawn = self.plot_boxes(results, frame_drawn, model_type="obb")
            return frame_drawn
        else:
            self.prediction_dict["convnext"] = {np.nan: [(np.nan, np.nan), (np.nan, np.nan)]}
            if self.smoothing:
                self.prediction_dict["convnext_smooth"] = {np.nan: [(np.nan, np.nan), (np.nan, np.nan)]}
            return frame_drawn

    def _extract_straw_level(self, straw_level: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        x_pixel, y_pixel = self.helpers._get_straw_to_pixel_level(straw_level)
        if x_pixel is not np.nan:
            line_start = (int(x_pixel), int(y_pixel))
            line_end = (int(x_pixel) + 300, int(y_pixel))
            angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
            line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
        else:
            line_start, line_end = (np.nan, np.nan), (np.nan, np.nan)
        return line_start, line_end

    def _yolo_model(self, frame: np.ndarray, frame_drawn: np.ndarray, cutout, time_stamp=None) -> None:
        # bgr to rgb
        output, inference_time = time_function(self.yolo_model.score_frame, frame.copy())
        # plot the box on the frame
        # frame_drawn = self.plot_boxes(output, frame_drawn, model_type="obb", label="straw")	
        smoothed_straw_level = 0
        # If the output is not empty, we can plot the boxes and get the straw level
        if len(output[0]) != 0:
            if cutout is not None:
                straw_level, interpolated, chute_nrs = self.helpers._get_pixel_to_straw_level_cutout(frame, output)
            else:
                straw_level, interpolated, chute_nrs = self.helpers._get_pixel_to_straw_level(frame_drawn, output)

            if straw_level is None:
                self.information["yolo_level"]["text"] = f'(T2) YOLO Level: NA'
            else:
                self.information["yolo_level"]["text"] = f'(T2) YOLO Level: {straw_level:.2f} %'
            # Smooth the data
            if self.smoothing:
                smoothed_straw_level = self.helpers._smooth_level(straw_level, 'yolo', time_stamp = time_stamp)
                if smoothed_straw_level is None:
                    self.information["yolo_smooth"]["text"] = f'(T2) Smoothed Straw Level: NA'
                else:
                    self.information["yolo_smooth"]["text"] = f'(T2) Smoothed Straw Level: {smoothed_straw_level:.2f} %'
            
            if straw_level is not None:
                line_start, line_end = self._extract_straw_level(straw_level)
            else:
                straw_level = np.nan
                line_start, line_end = (np.nan, np.nan), (np.nan, np.nan)

            if self.smoothing:
                smoothed_line_start, smoothed_line_end = self._extract_straw_level(smoothed_straw_level)
                frame_drawn = self.plot_straw_level(frame_drawn, smoothed_line_start, smoothed_line_end, smoothed_straw_level, self.yolo_color)
            else:
                frame_drawn = self.plot_straw_level(frame_drawn, line_start, line_end, straw_level, self.yolo_color)
                    
            # Get coordiantes for the original data
            self.prediction_dict["yolo"] = {straw_level: [line_start, line_end]}
            self.prediction_dict["attr."] = {interpolated: chute_nrs}
            if self.smoothing:
                self.prediction_dict["yolo_smooth"] = {smoothed_straw_level: [smoothed_line_start, smoothed_line_end]}

            self.information["yolo_model"]["text"] = f'(T2) YOLO Time: {inference_time:.2f} s'
        else:
            straw_level = 0          
            line_start, line_end = self._extract_straw_level(straw_level)

            if self.smoothing:
                smoothed_straw_level = self.helpers._smooth_level(straw_level, 'yolo', time_stamp = time_stamp)
                self.information["yolo_smooth"]["text"] = f'(T2) Smoothed Straw Level: {smoothed_straw_level:.2f} %'
                smoothed_line_start, smoothed_line_end = self._extract_straw_level(smoothed_straw_level)
                frame_drawn = self.plot_straw_level(frame_drawn, line_start, line_end, smoothed_straw_level, self.yolo_color)
            else:
                frame_drawn = self.plot_straw_level(frame_drawn, line_start, line_end, straw_level, self.yolo_color)

            self.information["yolo_model"]["text"] = f'(T2) YOLO Time: {inference_time:.2f} s'
            
            # Get coordiantes for the original data
            self.prediction_dict["yolo"] = {straw_level: [line_start, line_end]}
            self.prediction_dict["attr."] = {False: np.nan}
            if self.smoothing:
                if smoothed_straw_level is None:
                    smoothed_straw_level = 0
                self.prediction_dict["yolo_smooth"] = {smoothed_straw_level: [smoothed_line_start, smoothed_line_end]}

        return frame_drawn
        
    def _process_predictions(self, frame, results, frame_drawn, time_stamp=None, cutout=None) -> np.ndarray:
        """Run model predictions and update overlay."""
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cutout_image, prep_time = time_function(self.helpers._prepare_for_inference, frame, results, cutout)
        
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

            line_start, line_end = self._extract_straw_level(straw_level)            
            self.information["predictor_level"]["text"] = f'(T2) Predictor Level: {straw_level:.2f} %'

            self.information["predictor_model"]["text"] = f'(T2) Predictor Time: {inference_time:.2f} s'
            self.information["prep"]["text"] = f'(T2) Image Prep. Time: {prep_time:.2f} s'
            
            # We smooth the straw level
            if self.smoothing:
                smoothed_straw_level = self.helpers._smooth_level(straw_level, 'predictor', time_stamp = time_stamp)
                self.information["predictor_smooth"]["text"] = f'(T2) Smoothed Straw Level: {smoothed_straw_level:.2f} %'
                smoothed_line_start, smoothed_line_end = self._extract_straw_level(smoothed_straw_level)
                frame_drawn = self.plot_straw_level(frame_drawn, smoothed_line_start, smoothed_line_end, smoothed_straw_level, self.predictor_color)
            else:
                frame_drawn = self.plot_straw_level(frame_drawn, line_start, line_end, straw_level, self.predictor_color)

            if straw_level is None:
                straw_level = np.nan
                line_start, line_end = (np.nan, np.nan), (np.nan, np.nan)    
            self.prediction_dict["convnext"] = {straw_level: [line_start, line_end]}
            if self.smoothing:
                if smoothed_straw_level is None:
                    smoothed_straw_level = np.nan
                    smoothed_line_start, smoothed_line_end = (np.nan, np.nan), (np.nan, np.nan)
                self.prediction_dict["convnext_smooth"] = {smoothed_straw_level: [smoothed_line_start, smoothed_line_end]}

        else:
            self.information["predictor_level"]["text"] = f'(T2) Predictor Level: NA'
            self.information["predictor_smooth"]["text"] = f'(T2) Smoothed Straw Level: NA'
            self.information["predictor_model"]["text"] = f'(T2) Predictor Time: NA'
            self.information["prep"]["text"] = f'(T2) Image Prep. Time: NA'
            self.prediction_dict["convnext"] = {np.nan: [(np.nan, np.nan), (np.nan, np.nan)]}
            if self.smoothing:
                self.prediction_dict["convnext_smooth"] = {np.nan: [(np.nan, np.nan), (np.nan, np.nan)]}
            print("YOLO: No cutout image found.")
        return frame_drawn

    def _update_information(self, frame_time):
        """Update frame processing information."""
        total_time = time.time() - frame_time
        self.information["frame_time"]["text"] = f'(T2) Total Frame Time: {total_time:.2f} s'

        fps = 1 / total_time
        self.information["FPS"]["text"] = f'(T2) FPS: {fps:.2f}'

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
                cv2.putText(frame_drawn, val["text"], pos, self.font, font_scale, val["color"], font_thickness, cv2.LINE_AA) # Draw the text over the box

        cv2.imshow('Video', frame_drawn)
        cv2.waitKey(1)

    def close_threads(self):
        print("END: Threads and resources...")
        self.lock.acquire()
        self.should_abort_immediately = True
        self.lock.release()
        for thread in self.threads:
            thread.join()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("EXIT: Stream has been terminated...")

    def __call__(self) -> None | list:
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
        print("START: Threads and resources...")
        # start opcua server pulling data
        if self.pull_opcua:
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

def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('--window', action='store_true', help='Display the frames in a window.')
    parser.add_argument('--make_cutout', action='store_true', help='Make a cutout of the detected AprilTags.')
    parser.add_argument('--use_cutout', action='store_true', help='Use the cutout for predictions.')
    parser.add_argument('--object_detect', action='store_true', help='Use object detection for predictions.')
    parser.add_argument('--yolo_threshold', type=float, default=0.2, help='The threshold for object detection.')
    parser.add_argument('--detect_april', action='store_true', help='Detect AprilTags in the frames.')
    parser.add_argument('--yolo_straw', action='store_true', help='Use YOLO for straw level detection.')
    parser.add_argument('--with_predictor', action='store_true', help='Use the predictor for straw level detection.')
    parser.add_argument('--regressor', action='store_true', help='Use the regressor for straw level detection.')
    parser.add_argument('--edges', action='store_true', help='Use edge detection for predictions.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='The device to use for predictions.')
    parser.add_argument('--smoothing', action='store_true', help='Smooth the predictions.')
    parser.add_argument('--pull_opcua', action='store_true', help='Pull data from OPC UA server.')
    parser.add_argument('--write_opcua', action='store_true', help='Write data to OPC UA server.')
    parser.add_argument('--write_csv', action='store_true', help='Write data to CSV file.')
    return parser.parse_args()


def main(args: Namespace) -> None:
    """
    The main function that runs the script based on the arguments provided.

    Parameters
    ----------
    args    :   Namespace
        The arguments parsed by the ArgumentParser.
    """
    if args.yolo_straw == False and args.with_predictor == False:
        raise ValueError("One of the following must be True: yolo_straw, with_predictor")
    args.window = True
    args.make_cutout = True
    args.yolo_threshold = 0.2
    args.detect_april = True
    if args.with_predictor:
        args.object_detect = True
        args.regressor = True
    print(f"Running with smoothing: {args.smoothing}")
        
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

    # ### YOLO PREDICTOR
    RTSPStream(detector=detector, 
               ids=config["ids"], 
               window=args.window, 
               credentials_path='data/hkvision_credentials.txt', 
               make_cutout=args.make_cutout, 
               use_cutout=args.use_cutout, 
               object_detect=args.object_detect, 
               od_model_name="models/obb_cutout_best.pt", 
               yolo_threshold=args.yolo_threshold,
               detect_april=args.detect_april, 
               yolo_straw=args.yolo_straw, 
               yolo_straw_model="models/obb_best_dazzling.pt",
               with_predictor=args.with_predictor, 
               predictor_model='convnext', 
               model_load_path='models/convnext_regressor/', 
               regressor=args.regressor, 
               edges=args.edges, 
               device=args.device,
               smoothing=args.smoothing,
               mode='stream', # Only used when a single model is used for predictions
               pull_opcua=args.pull_opcua,
               write_opcua=args.write_opcua,
               write_csv=args.write_csv,
            )()


if __name__ == "__main__":
    args = get_args()
    main(args)

