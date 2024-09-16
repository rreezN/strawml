from __init__ import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import h5py
import random
# from skimage.transform import rotate, resize
from scipy.ndimage.interpolation import rotate


from make_dataset import decode_binary_image


def is_continuous(numbers):
    # Sort the list of numbers
    sorted_numbers = sorted(numbers)
    # Check if the difference between consecutive numbers is 1
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i + 1] - sorted_numbers[i] != 1:
            print(f"Missing frame: {sorted_numbers[i] + 1}")
            return False
    return True


def rotate_image(image, image_diff, bbox, angle):
    """
    https://medium.com/@coding-otter/image-and-bounding-box-rotation-using-opencv-python-2def6c39453
    """
    # image = rotate(image, angle).astype(np.uint8)
    # image_diff = rotate(image_diff, angle).astype(np.uint8)
    
    h, w = image.shape[:2] 
    cx, cy = (int(w / 2), int(h / 2))
    M = cv2.getRotationMatrix2D((cx, cy), -angle, scale=1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    newW = int((h * sin) + (w * cos))
    newH = int((h * cos) + (w * sin))
    M[0, 2] += (newW / 2) - cx
    M[1, 2] += (newH / 2) - cy
    
    
    # Rotate image and bounding box
    image = cv2.warpAffine(image, M, (newW, newH))
    image_diff = cv2.warpAffine(image_diff, M, (newW, newH))
    
    bbox_tuple = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[3]),
        (bbox[4], bbox[5]),
        (bbox[6], bbox[7]),
    ] 
    
    rotated_bbox = []

    for i, coord in enumerate(bbox_tuple):
        v = [coord[0], coord[1], 1]
        adjusted_coord = np.dot(M, v)
        rotated_bbox.insert(i, (adjusted_coord[0], adjusted_coord[1]))

    new_bbox = [int(x) for t in rotated_bbox for x in t]
    # show the image and the bounding box
    x1, y1, x2, y2, x3, y3, x4, y4 = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], new_bbox[4], new_bbox[5], new_bbox[6], new_bbox[7]
    # draw lines between the corners
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(image, (x2, y2), (x3, y3), (0, 255, 0), 2)
    cv2.line(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
    cv2.line(image, (x4, y4), (x1, y1), (0, 255, 0), 2)
    cv2.imshow("Rotated Image", image)
    cv2.waitKey(0)
    return image, image_diff, new_bbox
    
def translate(image, image_diff, bbox, x, y):
    ...

def scale(image, image_diff, bbox, scale):
    ...
    
def crop(image, image_diff, bbox, x, y, w, h):
    ...
    
    
def augment_chute_data(args):
    # Copy file from args.data to args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    os.system(f"cp {args.data} {args.output_dir}/")
    # Open the file    
    hf = h5py.File(args.output_dir + "/" + args.data.split("/")[-1], 'r')   
    # Based on hf.keys() get the largest number of frames and add 1 to get the next frame number
    frame_nr = max([int(frame.split('_')[1]) for frame in hf.keys()]) + 1
    
    for frame in hf.keys():
        image = decode_binary_image(hf[frame]['image'][...])
        image_diff = decode_binary_image(hf[frame]['image_diff'][...])
        bbox_chute = hf[frame]['annotations']['bbox_chute'][...]
        
        image, image_diff, bbox_chute = rotate_image(image, image_diff, bbox_chute, -90)
        # if random.random() < args.fraction:
            # for i in range(args.num):
                # do augment
                # pass




def get_args() -> argparse.Namespace:
    """Get the arguments for the data augmentation script.
    """
    parser = argparse.ArgumentParser(description='Augment the chute data.')
    parser.add_argument('--data', type=str, default='data/interim/chute_detection.hdf5', help='Directory containing the chute data')
    parser.add_argument('--output_dir', type=str, default='data/processed/augmented', help='Directory to save the augmented data')
    parser.add_argument('--num', type=int, default=3, help='Number of augmentations to create per image')
    parser.add_argument('--fraction', type=float, default=0.75, help='Fraction of images to augment')
    parser.add_argument('--type', type=str, nargs='+', default='rotation translation scaling flipping', help='Type of augmentation to apply. Options: rotation, translation, scaling, flipping')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    augment_chute_data(args)