from __init__ import *
import cv2 as cv
from cv2 import aruco
from tqdm import tqdm
import os
from moms_apriltag import TagGenerator3


ARUCO_DICT = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_5X5_1000": aruco.DICT_5X5_1000,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_6X6_1000": aruco.DICT_6X6_1000,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11,
}


def aruco_make(marker_dict, marker_size, id_dict):
    marker_path = "fiducial_marker/aruco_tags/"
    # remove all markers from the folder
    if not os.path.exists(marker_path):
        os.makedirs(marker_path)
    else:
        for file in os.listdir(marker_path):
            os.remove(marker_path + file)

        # generating unique IDs using for loop
    for name, id in tqdm(id_dict.items()):  # genereting 20 markers
        # using funtion to draw a marker
        marker_image = aruco.generateImageMarker(marker_dict, id, marker_size)
        cv.imwrite(f"{marker_path}{name}.png", marker_image)

def april_make(families, id_dict):
    marker_path = "fiducial_marker/april_tags/"
    if not os.path.exists(marker_path):
        os.makedirs(marker_path)
    else:
        for file in os.listdir(marker_path):
            os.remove(marker_path + file)

    tagGen = TagGenerator3(families)    
    for name, id in tqdm(id_dict.items()):
        tag = tagGen.generate(id, scale=100)
        cv.imwrite(f"{marker_path}{name}.png", tag)

    