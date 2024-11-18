from __init__ import *
import cv2
import json
from pupil_apriltags import Detector
from strawml.visualizations.stream import RTSPStream

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
RTSPStream(detector, config["ids"], window=True, credentials_path='data/hkvision_credentials.txt', rtsp=True, 
            make_cutout=True, object_detect=True, od_model_name="models/yolov11_obb_m8100btb_best.pt", yolo_threshold=0.2,
            detect_april=False,
            with_predictor=True, predictor_model='vit', model_load_path='models/vit_regressor/', regressor=True, edges=True, heatmap=False)()