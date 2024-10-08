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
        detector = Detector(
            families=config["dict_type"],
            nthreads=config["nthreads"],
            quad_decimate=config["quad_decimate"],
            quad_sigma=config["quad_sigma"],
            refine_edges=config["refine_edges"],
            decode_sharpening=config["decode_sharpening"],
            debug=config["debug"]
        )

        if args.mode == 'make':
            april_make(config["dict_type"], config["ids"])
        elif args.mode == 'calibrate':
            ...
        elif args.mode == 'detect':
            # load file
            # AD = RTSPStream(detector, config["ids"], window=args.windowed, credentials_path='data/hkvision_credentials.txt')
            # frame = cv2.imread("fiducial_marker/chute.png")
            # AD(frame)
            # AD(cap=cap)
            RTSPStream(detector, config["ids"], window=args.windowed, credentials_path='data/hkvision_credentials.txt')()
            
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

    parser.add_argument('--windowed', type=bool, default=True, help='Whether to run the script in windowed mode.')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    main(args)


