from __init__ import *
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union
import json
import cv2.aruco as aruco
import argparse

from fiducial_marker.make_marker import ARUCO_DICT, aruco_make, april_make
from fiducial_marker.detect_marker import ArucoDetector, RTSPStream
from fiducial_marker.calibrate_marker import calibrate_camera
# from aruco.p_aruco.aruco_calibrate import aruco_calibrate

def main(args):

    if args.tag_type == "aruco":
        # load dictionary from json config file
        with open("fiducial_marker/aruco_config.json", "r") as file:
            config = json.load(file)

        # Define the marker_dict
        marker_dict = aruco.getPredefinedDictionary(ARUCO_DICT[config["dict_type"]])
        detector_params = aruco.DetectorParameters()
        marker_size = config["marker_size"]
        ids = config["ids"]

        if args.mode == "make":
            aruco_make(marker_dict, marker_size, ids)
        elif args.mode == "calibrate":
            # aruco_calibrate()
            ...
        elif args.mode == "detect":
            ArucoDetector(marker_dict, detector_params, window=args.windowed)()

    if args.tag_type == "april":
        from pupil_apriltags import Detector

        with open("fiducial_marker/april_config.json", "r") as file:
            config = json.load(file)

        if args.mode == 'make':
            april_make(config["dict_type"], config["ids"])
        elif args.mode == 'calibrate':
            calibrate_camera(config["calibrate_dict"], min_acceptable_images=args.min_acceptable_images, rtsp=args.rtsp)
        elif args.mode == 'detect':
            detector = Detector(
                families=config["dict_type"],
                nthreads=config["nthreads"],
                quad_decimate=config["quad_decimate"],
                quad_sigma=config["quad_sigma"],
                refine_edges=config["refine_edges"],
                decode_sharpening=config["decode_sharpening"],
                debug=config["debug"]
            )
            RTSPStream(detector, config["ids"], window=args.windowed, credentials_path='data/hkvision_credentials.txt', rtsp=args.rtsp)()
            
            ## or load file
            # AD = RTSPStream(detector, config["ids"], window=args.windowed, credentials_path='data/hkvision_credentials.txt')
            # frame = cv2.imread("fiducial_marker/chute.png")
            # AD(frame)
            
def get_args() -> argparse.Namespace:
    """
    Get the arguments for the data augmentation script.

    Returns
    -------
    argparse.Namespace
        The arguments passed to the script
    """
    parser = argparse.ArgumentParser(description='Augment the chute data.')
    parser.add_argument('mode', type=str, choices=['make', 'calibrate', 'detect'], help='Mode to run the script in.')
    parser.add_argument('--tag_type', type=str, default='april', choices=['april', 'aruco'], help='Which tag type to use.')
    parser.add_argument('--min_acceptable_images', type=int, default=60, help='Minimum number of images to calibrate the camera.')
    parser.add_argument('--rtsp', action='store_true', help='Whether to use rtsp stream.')
    parser.add_argument('--windowed', action='store_true', help='Whether to run the script in windowed mode.')
    return parser.parse_args()

if __name__ == '__main__':
    # TODO Write the functionality to calibrate the camera
    args = get_args()
    main(args)

