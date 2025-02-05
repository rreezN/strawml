from __init__ import *
# Misc.
import os
import cv2
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
        cv2.line(frame_drawn, line_start, line_end, color, 2)
        if straw_level is None:
            cv2.putText(frame_drawn, "NA", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_drawn, f"{straw_level:.2f}%", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return frame_drawn
    
    def plot_boxes(self, results, frame, model_type: str = 'obb'):
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
    def __init__(self, record, record_threshold, detector, ids, credentials_path, od_model_name=None, object_detect=True, yolo_threshold=0.2, device="cuda", window=True, rtsp=True, make_cutout=False, use_cutout=False, detect_april=False, yolo_straw=False, yolo_straw_model="", with_predictor: bool = False, model_load_path: str = "models/vit_regressor/", regressor: bool = True, predictor_model: str = "vit", edges=True, heatmap=False, smoothing:bool=True, save_as_new_hdf5: bool=True, process_like_recording:bool=True, with_annotations:bool=False, fps_test:bool=False, hdf5_model_save_name:str | None = None, mode: str | None = None, carry_over: bool = True) -> None:
        super().__init__(detector, ids, window, od_model_name, object_detect, yolo_threshold, device, yolo_straw=yolo_straw, yolo_straw_model=yolo_straw_model, with_predictor=with_predictor, model_load_path=model_load_path, regressor=regressor, predictor_model=predictor_model, edges=edges, heatmap=heatmap)
        self.record = record
        self.save_as_new_hdf5 = save_as_new_hdf5
        self.with_annotations = with_annotations
        self.process_like_recording = process_like_recording
        self.recording_req = False
        self.record_threshold = record_threshold
        self.fps_test = fps_test
        self.hdf5_model_save_name = hdf5_model_save_name
        if record:
            self.recording_queue = queue.Queue()
        self.scada_smoothing_queue = deque(maxlen=5)
        self.yolo_smoothing_queue = deque(maxlen=5)
        self.predictor_smoothing_queue = deque(maxlen=5)
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
        self.last_save_timestamp = 0
        self.mode = mode
        self.carry_over = carry_over
        
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

    def test_april_detector(self, video_path: list[str]) -> None:
        for path in video_path:
            self._setup_display()
            if path.endswith(".mp4"):
                ...
            elif path.endswith(".hdf5"):
                with h5py.File(path, 'r') as hf:
                    timestamps = list(hf.keys())

                    if "frame" == timestamps[0].split("_")[0]:
                        timestamps = sorted(timestamps, key=lambda x: int(x.split('_')[1]))
                    else:
                        timestamps = sorted(timestamps, key=lambda x: float(x))
                    i = 0
                    for timestamp in tqdm(timestamps):
                        frame = hf[timestamp]['image'][()]
                        bbox_label = hf[timestamp]['annotations']['bbox_chute'][...]
                        frame = decode_binary_image(frame)
                        # change from bgr to rgb
                        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        results, duration = time_function(self.detect, frame_, True)
                        found_tags, existing_tags, inferred_tags = results
                        self.information["april"]["text"] = f'(T3) AprilTag Time: {duration:.2f} s' 
                        frame_drawn, cutout = self.draw(frame=frame.copy(), tags=self.tags.copy(), make_cutout=self.make_cutout, use_cutout=self.use_cutout)
                        self._display_frame(frame_drawn)
                        remain_set = set(range(27)) - set(sorted([tag.tag_id for tag in found_tags] + list(inferred_tags) + list(existing_tags.keys())))
                        # save the cutout as a png to show as an example
                        if self.make_cutout and self.use_cutout:
                            cv2.imwrite(f"data/cutouts/{'carry_over' if self.carry_over else 'no_carry_over'}/{timestamp}_full.png", frame_drawn)
                            cv2.imwrite(f"data/cutouts/{'carry_over' if self.carry_over else 'no_carry_over'}/{timestamp}_cutout.png", cutout)
                        # print("----------------------------------------------------")
                        save_dict = {timestamp : {"carry_over": self.carry_over,"found_tags": sorted([tag.tag_id for tag in found_tags]), "existing_tags": sorted(list(existing_tags.keys())), "inferred_tags": sorted(list(inferred_tags)), "not_found_tags": list(remain_set)}}
                        # write save_dict to a new line in a file
                        # print(list(remain_set))
                        file_name = f"data/{'carry_over' if self.carry_over else 'no_carry_over'}_april_tag_testing.json"
                        with open(file_name, "a") as file:
                            file.write(json.dumps(save_dict) + "\n")
    
                        if not self.carry_over:
                            self.helpers._reset_tags()
                            self.tag_graph.update_init(self.tags, self.inferred_tags)

        
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

    def display_frame_from_hdf5(self, paths: str) -> None:
        """
        Display the frames with the detected AprilTags.
        """
        for path in paths:
            # Start by running through the first 50 frames to get the tags
            self.helpers._reset_tags()
            self._warm_up_for_apriltags(path, 0, 50)
            # pause to allow for the apriltag detect thread to catch up
            print("3 seconds pause to allow for apriltag detection to catch up...")
            time.sleep(3)
            self._setup_display()
            self._process_frames_from_hdf5(path)

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
            self._process_frame(frame, from_videofile=True)

    def _process_frames_from_hdf5(self, path: str) -> None:
        """Process frames from an HDF5 file."""
        file_name = path.split("/")[-1].split(".")[0]
        if self.fps_test:
            # NOTE if this is true, then we only need to load the image, run inference and then continue. We do not need to save anything.
            # Then we also wish to create a dictionary that explains the different durations of loading image, processing image and then total image time
            # we must ensure that only one model is running at a time otherwise the results will be skewed and not accurate
            if self.yolo_straw:
                self.fps_test_results = {"fps": [], "load_time": [], "inference_time": [], "postprocess_time": [], "total_time": []}
            else:
                self.fps_test_results = {"fps": [], "load_time": [], "cutout_time": [], "inference_time": [], "postprocess_time": [], "total_time": []}
        with h5py.File(path, 'r+') as hf:
            timestamps = list(hf.keys())
            # sort timestamps
            if "frame" == timestamps[0].split("_")[0]:
                timestamps = sorted(timestamps, key=lambda x: int(x.split('_')[1]))
            else:
                timestamps = sorted(timestamps, key=lambda x: float(x))
            # Check if hf[timestamp]['label'] exists for the first timestamp. 
            # If it does we do not need to get labels again
            self.recording_req = True
            self.last_save_timestamp = timestamps[0]
            self.save_counter = 0
            pbar = tqdm(timestamps)
            for i, timestamp in enumerate(pbar):
                if 'type' in hf[timestamp].attrs.keys():
                    if i == 0:
                        self.previous_frame_type = hf[timestamp].attrs['type']
                    pbar.set_description(f"{file_name}, #Saved frames: {self.save_counter}, {self.previous_frame_type}")
                else:
                    pbar.set_description(f"{file_name}, #Saved frames: {self.save_counter}")
                frame_time = time.time()
                self.prediction_dict = {}
                frame = hf[timestamp]['image'][()]
                frame = decode_binary_image(frame) # process the frame
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if 'type' in hf[timestamp].attrs.keys():
                    if hf[timestamp].attrs['type'] != self.previous_frame_type:
                        self._warm_up_for_apriltags(path, i, 50)
                        self.previous_frame_type = hf[timestamp].attrs['type']

                if self.fps_test:
                    self.fps_test_results["load_time"].append(time.time() - frame_time)
                    self._process_fps_test(frame, frame_time)
                else:
                    if self.with_annotations:
                        self._process_sensor_hdf5_frame(frame, hf, path, timestamp, frame_time)
                    else:
                        self._process_hdf5_frame(frame, hf, path, timestamp, frame_time)
        if self.fps_test:
            print(f"Results: {self.fps_test_results}")
            # save the results to a file
            with open("fps_test_results.pkl", "wb") as f:
                pickle.dump(self.fps_test_results, f)
        # self.close_threads()

    def _process_fps_test(self, frame, frame_time) -> None:
        if self.yolo_straw:
            return self._process_fps_test_yolo(frame, frame_time)
        if self.with_predictor:
            return self._process_fps_test_predictor(frame, frame_time)
        
    def _process_fps_test_yolo(self, frame, frame_time) -> None:
        # Get yolo results
        frame_drawn = self._yolo_model(frame, frame, None)
        self._update_information(frame_time)
        self._display_frame(frame_drawn)

    def _process_fps_test_predictor(self, frame, frame_time) -> None:
        # Get predictor results
        results, OD_time = time_function(self.OD.score_frame, frame.copy())
        # Make sure the results are not empty
        if len(results[0]) == 0:
            results = "NA"
        self.information["od"]["text"] = f'(T2) OD Time: {OD_time:.2f} s'
        frame_drawn = self._predictor_model(frame, frame, results)
        self._update_information(frame_time)
        self._display_frame(frame_drawn)

    def _process_sensor_hdf5_frame(self, frame: np.ndarray, hf, path, timestamp, frame_time) -> None:
        """ 
        Special function for just processing sensor data. 
        It needs to do a series of operations on the existing sensor file.
        1. Retrieve existing annotated bbox and calculate the straw level
        2. Save the straw level as label to the existing group
        3. run the normal _process_hdf5_frame 
        """
        try:
            # Load and calculate the straw level based on the bbox and the fullness score
            straw_bbox = hf[timestamp]['annotations']['bbox_straw'][...]
            straw_level_bbox = self.helpers._get_pixel_to_straw_level(frame, straw_bbox, object=False)[0]
            straw_level_bbox_line = self.helpers._get_straw_to_pixel_level(straw_level_bbox)
            
            straw_level_fullness =  hf[timestamp]['annotations']['fullness'][...] * 100
            straw_level_fullness_line = self.helpers._get_straw_to_pixel_level(straw_level_fullness)

            # prepare the straw level for saving by checking if the group already exists and
            # if they do we delete them and replace them with the new values
            if "straw_percent_bbox" in hf[timestamp].keys():
                del hf[timestamp]["straw_percent_bbox"]
            if "straw_percent_fullness" in hf[timestamp].keys():
                del hf[timestamp]["straw_percent_fullness"]
            angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))

            line_start = (int(straw_level_bbox_line[0]), int(straw_level_bbox_line[1]))
            line_end = (int(straw_level_bbox_line[0])+300, int(straw_level_bbox_line[1]))
            line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
            straw_percent_bbox_group = hf[timestamp].create_group('straw_percent_bbox')
            straw_percent_bbox_group.create_dataset('percent', data=straw_level_bbox)
            straw_percent_bbox_group.create_dataset('pixel', data=[line_start, line_end])
            
            line_start = (int(straw_level_fullness_line[0]), int(straw_level_fullness_line[1]))
            line_end = (int(straw_level_fullness_line[0])+300, int(straw_level_fullness_line[1]))
            line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
            straw_percent_fullness_group = hf[timestamp].create_group('straw_percent_fullness')
            straw_percent_fullness_group.create_dataset('percent', data=straw_level_fullness)
            straw_percent_fullness_group.create_dataset('pixel', data=[line_start, line_end])

        except Exception as e:
            t1 = "straw_percent_bbox" in hf[timestamp].keys()
            t2 = "straw_percent_fullness" in hf[timestamp].keys()
            if t1:
                del hf[timestamp]["straw_percent_bbox"]
            if t2:
                del hf[timestamp]["straw_percent_fullness"]
            straw_level = 0
            straw_level_line = self.helpers._get_straw_to_pixel_level(straw_level)
            print(self.chute_numbers.values())
            angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
            line_start = (int(straw_level_line[0]), int(straw_level_line[1]))
            line_end = (int(straw_level_line[0])+300, int(straw_level_line[1]))
            line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)

            straw_percent_bbox_group = hf[timestamp].create_group('straw_percent_bbox')
            straw_percent_bbox_group.create_dataset('percent', data=straw_level)
            straw_percent_bbox_group.create_dataset('pixel', data=[line_start, line_end])

            straw_percent_fullness_group = hf[timestamp].create_group('straw_percent_fullness')
            straw_percent_fullness_group.create_dataset('percent', data=straw_level)
            straw_percent_fullness_group.create_dataset('pixel', data=[line_start, line_end])
            print(f"{timestamp}: {e}, replaced: {t1, t2}")
        self._process_hdf5_frame(frame, hf, path, timestamp, frame_time)

    def _process_hdf5_frame(self, frame, hf, path, timestamp, frame_time) -> None:

        results = None 
        if frame is None:
            print("Frame is None. Skipping...")
            self._update_information(frame_time)
            return  
        with self.lock:
            self.frame = frame

        # Find Apriltags
        if self.detect_april and self.tags is not None:
            frame_drawn, cutout = self.draw(frame=frame.copy(), tags=self.tags.copy(), make_cutout=self.make_cutout, use_cutout=self.use_cutout)
            # print shape of cutout
        else:
            frame_drawn, cutout = frame, None

        if self.yolo_straw and self.with_predictor:
            return self._process_hdf5_frame_with_yolo_and_predictor(frame, frame_drawn, cutout, hf, path, timestamp, frame_time)
        
        if not self.use_cutout and not self.object_detect:
            try:
                bbox = torch.from_numpy(hf[timestamp]["annotations"]["bbox_chute"][()]).to(self.device)
                results = [0,[bbox]]
            except Exception as e:
                print(f"Error in getting bbox: {e}")
                return
        
        # If not tags have been found we go to the next frame
        if len(self.tags) == 0:
            self._display_frame(frame_drawn)
            return
        
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_drawn = self._process_frame_content(frame, frame_drawn, cutout, results, time_stamp=timestamp)

        # Display frame and overlay text
        self._update_information(frame_time)
        self._display_frame(frame_drawn)
        # # Now we save the frame
        if self.recording_req:
            self.save_counter += 1
            if self.save_as_new_hdf5:
                self._save_frame_existing_hdf5(hf=None, timestamp=timestamp, path=path)
            else:
                self._save_frame_existing_hdf5(hf, timestamp, path)

    def _process_hdf5_frame_with_yolo_and_predictor(self, frame, frame_drawn, cutout, hf, path, timestamp, frame_time) -> None:
        # If not tags have been found we go to the next frame
        if len(self.tags) == 0:
            self._display_frame(frame_drawn)
            return

        # Process scada data        
        scada_level = hf[timestamp]["scada"]["percent"][()]
        line_start, line_end, sensor_scada_data, scada_pixel_values = self._retrieve_scada_data(scada_level)
        if self.smoothing:
            self.recording_req =  (float(timestamp) - float(self.last_save_timestamp) >= self.record_threshold)
        else:
            self.recording_req = True
        if self.recording_req:
            if scada_pixel_values[0] is not None:
                self.prediction_dict = {}
                self.prediction_dict["scada"] = {sensor_scada_data: [line_start, line_end]} # NOTE FILLING SCADA DATA HERE
                self.last_save_timestamp = timestamp

        cv2.line(frame_drawn, line_start, line_end, self.scada_color , 2)
        cv2.putText(frame_drawn, f"{sensor_scada_data:.2f}%", (int(line_end[0])+10, int(line_end[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, self.scada_color, 2, cv2.LINE_AA)
        # Get yolo results
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_drawn = self._yolo_model(frame, frame_drawn, None) # NOTE FILLING YOLO DATA HERE

        # Get predictor results
        results, OD_time = time_function(self.OD.score_frame, frame.copy())
        # Make sure the results are not empty
        if len(results[0]) == 0:
            results = "NA"
        self.information["od"]["text"] = f'(T2) OD Time: {OD_time:.2f} s'
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_drawn = self._predictor_model(frame, frame_drawn, results) # NOTE FILLING CONVNEXTV2 DATA HERE

        # Display frame and overlay text
        self._update_information(frame_time)
        self._display_frame(frame_drawn)

        # save results from frame if recording requirement is met
        if self.recording_req:
            self.save_counter += 1
            if self.save_as_new_hdf5:
                self._save_frame_existing_hdf5(hf=None, timestamp=timestamp, path=path)
            else:
                self._save_frame_existing_hdf5(hf, timestamp, path)

    def _save_frame_existing_hdf5(self, hf, timestamp, path) -> None:
        """
        Function to save the frame to an existing HDF5 file.

        Parameters:
        -----------
        hf : h5py.File
            The HDF5 file object.
        timestamp : str
            The timestamp of the frame
        """
        def _save_results(hf, timestamp, hdf5_model_save_name = None):
            # check if the timestamp already exists in the file
            if not timestamp in hf.keys():
                group = hf.create_group(timestamp)
            else:
                group = hf[timestamp]

            # Save the image to the group if it does not exist
            if not "image" in group.keys():
                img = cv2.imencode('.jpg', self.frame)[1]
                group.create_dataset('image', data=img)

            if hdf5_model_save_name is not None:
                try:
                    # remove the existing entry:
                    if hdf5_model_save_name in group.keys():
                        del group[hdf5_model_save_name]

                    if self.yolo_straw:
                        t = self.prediction_dict["yolo"]
                    elif self.with_predictor:
                        t = self.prediction_dict["convnextv2"]
                        t_ = self.prediction_dict["yolo_cutout"]
                        if 'yolo_cutout' in group.keys():
                            del group['yolo_cutout']
                        group.create_dataset('yolo_cutout', data=t_)
                    
                    t1_name = 'percent'
                    t2_name = 'pixel'
                    t1, t2 = list(t.keys())[0], list(t.values())[0]
                    # then we create the group and add the datasets
                    pred = group.create_group(hdf5_model_save_name)
                    pred.create_dataset(t1_name, data=t1)
                    pred.create_dataset(t2_name, data=t2)


                except Exception as e:
                    if hdf5_model_save_name in group.keys():
                        del group[hdf5_model_save_name]
                    if 'yolo_cutout' in group.keys():
                        del group['yolo_cutout']
                    print("###################")
                    print(f"!ERROR! {timestamp}: {hdf5_model_save_name},\n{t1_name}: {t1},\n{t2_name}: {t2}\n--- {e}")
                    print("###################")

            else:
                try:
                    for key, value in self.prediction_dict.items():
                        if key in group.keys():
                            del group[key]
                        predict_group = group.create_group(key)
                        t1, t2 = list(value.items())[0]
                        if key == 'attr.':
                            t1_name = 'interpolated'
                            t2_name = 'tags'
                        else:
                            t1_name = 'percent'
                            t2_name = 'pixel'
                        predict_group.create_dataset(t1_name, data=t1)
                        predict_group.create_dataset(t2_name, data=t2)
                except Exception as e:
                    print("###################")
                    print(f"!ERROR! {timestamp}: {key},\n{t1_name}: {t1},\n{t2_name}: {t2}\n--- {e}")
                    print("###################")
        if hf is not None:
            _save_results(hf, timestamp, hdf5_model_save_name=self.hdf5_model_save_name)
        else:
            with h5py.File(path.split(".")[0] + "_processed.hdf5", 'a') as hf:
                # print(f"Saving frame to {path.split('.')[0] + '_processed.hdf5'}")
                _save_results(hf, timestamp)

    def _process_frame(self, frame: np.ndarray, from_videofile: bool) -> None:
        """Process a single frame for display."""
        if frame is None:
            print("Frame is None. Skipping...")
            return
        # Lock and set the frame for thread safety
        with self.lock:
            self.frame = frame

        # Clear the queue in RTSP mode to avoid lag
        if not from_videofile and self.rtsp:
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
        display_scada_line = False
        if self.record:
            try:
                sensor_scada_data = self.scada_thread.get_recent_value()
                line_start, line_end, sensor_scada_data, scada_pixel_values = self._retrieve_scada_data(sensor_scada_data)
                self.recording_req =  (frame_time - self.since_last_save >= self.record_threshold)
                if self.recording_req and scada_pixel_values[0] is not None:
                    self.prediction_dict = {}
                    self.prediction_dict["scada"] = {sensor_scada_data: [line_start, line_end]}
                    # Reset the time
                    self.since_last_save = time.time()
                display_scada_line = True
            except Exception as e:
                print(f'Error while trying to get scada data: {e}')
                
        # Process frame (Object Detection, Predictor, etc.)
        frame_drawn = self._process_frame_content(frame, frame_drawn, cutout)

        # Draw sensor_scada_data on frame based on scada_pixel_values
        if display_scada_line and scada_pixel_values[0] is not None:
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

    def _retrieve_scada_data(self, sensor_scada_data, time_stamp = None) -> None:
        # Get scada
        if self.smoothing:
            sensor_scada_data = self.helpers._smooth_level(sensor_scada_data, 'scada', time_stamp = time_stamp)
            self.information["scada_smooth"]["text"] = f'(T2) Smoothed Scada Level: {sensor_scada_data:.2f}%'
        # Get pixel values for scada
        scada_pixel_values = self.helpers._get_straw_to_pixel_level(sensor_scada_data)
        if scada_pixel_values[0] is not None:
            # Record sensor data if enabled
            scada_pixel_values_ = (scada_pixel_values[0], scada_pixel_values[1])
            # Get angle of self.chute_numbers
            angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
            line_start = (int(scada_pixel_values_[0]), int(scada_pixel_values_[1]))
            line_end = (int(scada_pixel_values_[0])+300, int(scada_pixel_values_[1]))
            line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
        else:
            line_start, line_end = (None,None), (None,None)
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
                results, OD_time = time_function(self.OD.score_frame, frame.copy())
                self.prediction_dict["yolo_cutout"] = results[1][0].flatten().cpu().detach().numpy()
                # Make sure the results are not empty
                if len(results[0]) == 0:
                    results = "NA"
                self.information["od"]["text"] = f'(T2) OD Time: {OD_time:.2f} s'
            else:
                results = results
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Predictor of the straw level
            if self.with_predictor:
                return self._predictor_model(frame, frame_drawn, results, time_stamp)
            elif self.yolo_straw:
                return self._yolo_model(frame, frame_drawn, cutout, time_stamp)
                
    def _predictor_model(self, frame: np.ndarray, frame_drawn: np.ndarray, results: list, time_stamp=None) -> None:
        if results != "NA":
            frame_drawn = self._process_predictions(frame, results, frame_drawn, time_stamp = time_stamp)
            if self.object_detect:
                frame_drawn = self.plot_boxes(results, frame_drawn, model_type="obb")
            return frame_drawn
        else:
            return frame_drawn

    def _yolo_model(self, frame: np.ndarray, frame_drawn: np.ndarray, cutout, time_stamp=None) -> None:
        # bgr to rgb
        output, inference_time = time_function(self.yolo_model.score_frame, frame.copy())
        # If the output is not empty, we can plot the boxes and get the straw level
        if self.fps_test:
            start = time.time()
        if len(output[0]) != 0:
            x_pixel, y_pixel = output[1][0][-1].cpu().numpy()

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
                straw_level = self.helpers._smooth_level(straw_level, 'yolo', time_stamp = time_stamp)
                if straw_level is None:
                    self.information["yolo_smooth"]["text"] = f'(T2) Smoothed Straw Level: NA'
                else:
                    self.information["yolo_smooth"]["text"] = f'(T2) Smoothed Straw Level: {straw_level:.2f} %'

            if straw_level is not None:
                # Since the new straw level might be a smoothed value, we need to update the pixel values of the straw level. We do this everytime to ensure that the overlay is based on the same pixel values all the time. Otherwise the overlay would shift from being based on the bbox pixel values vs. based on the tags.
                x_pixel, y_pixel = self.helpers._get_straw_to_pixel_level(straw_level)
                if x_pixel is None:
                    return frame_drawn
                
                # Define the overlay lines and orientation
                line_start = (int(x_pixel), int(y_pixel))
                line_end = (int(x_pixel) + 300, int(y_pixel))
                angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
                line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
                
                # Plot the line on the frame
                frame_drawn = self.plot_straw_level(frame_drawn, line_start, line_end, straw_level, self.yolo_color)

            if self.recording_req:
                # Get coordiantes for the original data
                self.prediction_dict["yolo"] = {straw_level: [line_start, line_end]}
                self.prediction_dict["attr."] = {interpolated: chute_nrs}

            if self.fps_test:
                self.fps_test_results["inference_time"].append(inference_time)
                self.fps_test_results["postprocess_time"].append(time.time() - start)
            
            self.information["yolo_model"]["text"] = f'(T2) YOLO Time: {inference_time:.2f} s'
        else:
            if self.smoothing:
                straw_level = self.helpers._smooth_level(0, 'yolo', time_stamp = time_stamp)
                self.information["yolo_smooth"]["text"] = f'(T2) Smoothed Straw Level: {straw_level:.2f} %'
            else:
                straw_level = 0
            x_pixel, y_pixel = self.helpers._get_straw_to_pixel_level(straw_level)
            if x_pixel is None:
                return frame_drawn
            line_start = (int(x_pixel), int(y_pixel))
            line_end = (int(x_pixel) + 300, int(y_pixel))
            angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
            line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
            # Plot the boxes and straw level on the frame
            frame_drawn = self.plot_straw_level(frame_drawn, line_start, line_end, straw_level, self.yolo_color)
            self.information["yolo_model"]["text"] = f'(T2) YOLO Time: {inference_time:.2f} s'
            if self.recording_req:
                # Get coordiantes for the original data
                self.prediction_dict["yolo"] = {0: [line_start, line_end]}
                self.prediction_dict["attr."] = {False: np.nan}
                # if no bbox is detected, we add 0 to the previous straw level smoothing predictions
                angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
                self.helpers._save_tag_0(angle)
            if self.fps_test:
                self.fps_test_results["inference_time"].append(np.nan)
                self.fps_test_results["postprocess_time"].append(time.time() - start)
        return frame_drawn
        
    def _process_predictions(self, frame, results, frame_drawn, time_stamp=None) -> np.ndarray:
        """Run model predictions and update overlay."""
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            if self.fps_test:
                start = time.time()
            self.information["predictor_level"]["text"] = f'(T2) Predictor Level: {straw_level:.2f} %'
            # We smooth the straw level
            if self.smoothing:
                straw_level = self.helpers._smooth_level(straw_level, 'predictor', time_stamp = time_stamp)
                self.information["predictor_smooth"]["text"] = f'(T2) Smoothed Straw Level: {straw_level:.2f} %'

            x_pixel, y_pixel = self.helpers._get_straw_to_pixel_level(straw_level)

            if x_pixel is not None:
                angle = self.helpers._get_tag_angle(list(self.chute_numbers.values()))
                line_start = (int(x_pixel), int(y_pixel))
                line_end = (int(x_pixel)+300, int(y_pixel))
                line_start, line_end = self.helpers._rotate_line(line_start, line_end, angle=angle)
            else:
                line_start, line_end = (None,None), (None,None)

            if self.recording_req:
                if straw_level is None:
                    print(f"Straw level is None: {time_stamp}")
                self.prediction_dict["convnextv2"] = {straw_level: [line_start, line_end]}
            # Draw line and text for straw level
            if straw_level is not None:
                if line_start[0] is not None:
                    cv2.line(frame_drawn, line_start, line_end, self.predictor_color, 2)
                    cv2.putText(frame_drawn, f"{straw_level:.2f}%", (line_end[0] + 10, line_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,  self.predictor_color, 2, cv2.LINE_AA)
                self.information["predictor_model"]["text"] = f'(T2) Predictor Time: {inference_time:.2f} s'
                self.information["prep"]["text"] = f'(T2) Image Prep. Time: {prep_time:.2f} s'

            if self.fps_test:
                self.fps_test_results["cutout_time"].append(prep_time)
                self.fps_test_results["inference_time"].append(inference_time)
                self.fps_test_results["postprocess_time"].append(time.time() - start)
        else:
            self.information["predictor_level"]["text"] = f'(T2) Predictor Level: NA'
            self.information["predictor_smooth"]["text"] = f'(T2) Smoothed Straw Level: NA'
            self.information["predictor_model"]["text"] = f'(T2) Predictor Time: NA'
            self.information["prep"]["text"] = f'(T2) Image Prep. Time: NA'
            self.prediction_dict["convnextv2"] = {0: [None, None]}
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

        if self.fps_test:
            self.fps_test_results["fps"].append(fps)
            self.fps_test_results["total_time"].append(total_time)

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
            if video_path is not None:
                import os
                # make sure the video paths in video path list exists
                for vp in video_path:
                    if not os.path.exists(vp):
                        print(f"The video path '{vp}' does not exist.")
                        return
                        
                if self.mode == 'april_testing':
                    threading.Thread(target=self.test_april_detector, args=(video_path,)).start()
                # Check if the file mp4 or hdf5
                elif video_path[0].endswith('.mp4'):
                    self.cap = cv2.VideoCapture(video_path[0])
                    print("START: Videofile loaded")
                    if self.detect_april:
                        self.thread1 = threading.Thread(target=self.find_tags)
                        self.thread1.start()
                        self.threads.append(self.thread1)
                    if self.record:
                        self.scada_thread = AsyncStreamThread(server_keys='data/opcua_server.txt')
                        self.scada_thread.start()
                        self.threads.append(self.scada_thread)
                    threading.Thread(target=self.display_frame_from_videofile).start()
                    # self.display_frame_from_videofile()
                elif video_path[0].endswith('.hdf5'):
                    if self.detect_april:
                        self.thread1 = threading.Thread(target=self.find_tags)
                        self.thread1.start()
                        self.threads.append(self.thread1)
                    threading.Thread(target=self.display_frame_from_hdf5, args=(video_path,)).start()
                    # self.display_frame_from_hdf5(video_path)
                else:
                    raise ValueError("The file type is not supported. Please provide a .mp4 or .hdf5 file.")
                while True:
                    if keyboard.is_pressed('q'):
                        self.close_threads()
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.close_threads()
                        break
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

def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('mode', type=str, choices=['manual', 'stream', 'fps_test', 'file_predict', 'record', 'april_testing'], help='Mode to run the script in (extracts images from videos and saves them to an hdf5 file, validate shows the difference between the original and extracted images, and tree prints the tree structure of the hdf5 file).')
    parser.add_argument('--record', action='store_true', help='Record the stream to an hdf5 file.')
    parser.add_argument('--record_threshold', type=int, default=5, help='The time in seconds to record the stream.')
    parser.add_argument('--window', action='store_true', help='Display the frames in a window.')
    parser.add_argument('--rtsp', action='store_true', help='Use an RTSP stream.')
    parser.add_argument('--make_cutout', action='store_true', help='Make a cutout of the detected AprilTags.')
    parser.add_argument('--use_cutout', action='store_true', help='Use the cutout for predictions.')
    parser.add_argument('--object_detect', action='store_true', help='Use object detection for predictions.')
    parser.add_argument('--yolo_threshold', type=float, default=0.2, help='The threshold for object detection.')
    parser.add_argument('--detect_april', action='store_true', help='Detect AprilTags in the frames.')
    parser.add_argument('--yolo_straw', action='store_true', help='Use YOLO for straw level detection.')
    parser.add_argument('--with_predictor', action='store_true', help='Use the predictor for straw level detection.')
    parser.add_argument('--regressor', action='store_true', help='Use the regressor for straw level detection.')
    parser.add_argument('--edges', action='store_true', help='Use edge detection for predictions.')
    parser.add_argument('--heatmap', action='store_true', help='Use a heatmap for predictions.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='The device to use for predictions.')
    parser.add_argument('--smoothing', action='store_true', help='Smooth the predictions.')
    parser.add_argument('--save_as_new_hdf5', action='store_true', help='Save the frames to a new hdf5 file.')
    parser.add_argument('--process_like_recording', action='store_true', help='Process the frames like a recording.')
    parser.add_argument('--with_annotations', action='store_true', help='Use the annotations for predictions.')
    parser.add_argument('--fps_test', action='store_true', help='Run the FPS test.')
    parser.add_argument('--hdf5_model_save_name', type=str, default=None, help='The name of the model to save the predictions.')
    parser.add_argument('-vp', '--video_path', nargs='+', default=[], help='The path to the video file(s).')
    parser.add_argument('--carry_over', action='store_true', help='Carry over AprilTag Detection to the next frame')
    return parser.parse_args()
    # python .\strawml\visualizations\stream.py file_predict --smoothing --save_as_new_hdf5 --process_like_recording --video_path data/predictions/recording_rotated_all_frames.hdf5
def main(args: Namespace) -> None:
    """
    The main function that runs the script based on the arguments provided.

    Parameters
    ----------
    args    :   Namespace
        The arguments parsed by the ArgumentParser.
    """

    if args.mode == 'manual':
        pass
    elif args.mode == 'stream':
        if args.yolo_straw == False and args.with_predictor == False:
            raise ValueError("One of the following must be True: yolo_straw, with_predictor")
        args.record = False
        args.record_threshold = 5
        args.window = True
        args.rtsp = True
        args.make_cutout = True
        args.yolo_threshold = 0.2
        args.detect_april = True
        if args.with_predictor:
            args.object_detect = True
            args.regressor = True
        print(f"Running with smoothing: {args.smoothing}")
    elif args.mode == 'fps_test':
        if args.yolo_straw == False and args.with_predictor == False:
            raise ValueError("One of the following must be True: yolo_straw, with_predictor")
        args.fps_test = True
        args.window = True
        args.make_cutout = True
        args.yolo_threshold = 0.2
        args.detect_april = True
        if args.with_predictor:
            args.object_detect = True
            args.regressor = True
        if args.video_path is None:
            raise ValueError("The video_path must be provided for the FPS test.")
        if args.yolo_straw and args.with_predictor:
            raise ValueError(f"Cannot run both YOLO and Predictor at the same time for mode: {args.mode}. Please choose **one**.")
    elif args.mode == 'file_predict':
        if args.yolo_straw == False and args.with_predictor == False:
            raise ValueError("One of the following must be True: yolo_straw, with_predictor")
        # args.yolo_straw = True
        # args.with_predictor = True
        args.make_cutout = True
        args.window = True
        args.yolo_threshold = 0.2
        args.detect_april = True
        args.regressor = True
        args.object_detect = True
        if args.video_path is None:
            raise ValueError("The video_path must be provided for the file predict.")
        print(f"NOTE. **hdf5_model_save_name** is only used when the video_path leads to an hdf5 file.")
    elif args.mode == 'record':
        if args.yolo_straw == False and args.with_predictor == False:
            raise ValueError("One of the following must be True: yolo_straw, with_predictor")
        args.record = True
        args.window = True
        args.rtsp = True
        args.make_cutout = True
        args.yolo_threshold = 0.2
        args.detect_april = True
        if args.yolo_straw and args.with_predictor:
            raise ValueError(f"Cannot run both YOLO and Predictor at the same time for mode: {args.mode}. Please choose **one**.")
        if args.with_predictor:
            args.object_detect = True
            args.regressor = True
    elif args.mode == 'april_testing':
        args.detect_april = True
        args.window = True
        args.make_cutout = True
        if args.video_path is None:
            raise ValueError("The video_path must be provided for the AprilTag Detector test.")
    
    if len(args.video_path) == 0:
        args.video_path = None
        
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
    RTSPStream(record=args.record, 
               record_threshold=args.record_threshold, 
               detector=detector, 
               ids=config["ids"], 
               window=args.window, 
               credentials_path='data/hkvision_credentials.txt', 
               rtsp=args.rtsp , # Only used when the stream is from an RTSP source
               make_cutout=args.make_cutout, 
               use_cutout=args.use_cutout, 
               object_detect=args.object_detect, 
               od_model_name="models/obb_cutout_best.pt", 
               yolo_threshold=args.yolo_threshold,
               detect_april=args.detect_april, 
               yolo_straw=args.yolo_straw, 
               yolo_straw_model="models/obb_best_swift.pt",
               with_predictor=args.with_predictor, 
               predictor_model='convnext', 
               model_load_path='models/convnext_regressor/', 
               regressor=args.regressor, 
               edges=args.edges, 
               heatmap=args.heatmap,
               device=args.device,
               smoothing=args.smoothing,
               save_as_new_hdf5=args.save_as_new_hdf5, 
               process_like_recording=args.process_like_recording, 
               with_annotations=args.with_annotations, 
               fps_test=args.fps_test, 
               hdf5_model_save_name = args.hdf5_model_save_name,
               mode=args.mode, # Only used when a single model is used for predictions
               carry_over=args.carry_over, # Only used when the carry over during april testing i needed
            )(video_path=args.video_path)

    # # ### YOLO PREDICTOR STREAM
    # RTSPStream(record=True, record_threshold=5, detector=detector, ids=config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', 
    #     rtsp=True , # Only used when the stream is from an RTSP source
    #     make_cutout=True, use_cutout=False, object_detect=False, od_model_name="models/yolov11-chute-detect-obb.pt", yolo_threshold=0.2,
    #     detect_april=True, yolo_straw=True, yolo_straw_model="models/obb_best.pt",
    #     with_predictor=False , predictor_model='convnextv2', model_load_path='models/convnext_regressor/', regressor=True, edges=False, heatmap=False,
    #     device='cuda')()


if __name__ == "__main__":
    args = get_args()
    main(args)

