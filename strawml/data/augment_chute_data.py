from __init__ import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import h5py
import random
from skimage.transform import rotate, resize

from make_dataset import decode_binary_image


def augment_chute_data(args):
    hf = h5py.File(args.data, 'r')
    frames = list(hf.keys())
    
    for frame in frames:
        image = decode_binary_image(hf[frame]['image'][...])
        image_diff = decode_binary_image(hf[frame]['image_diff'][...])
        bbox_chute = hf[frame]['annotations']['bbox_chute'][...]
        
        if random.random() < args.fraction:
            for i in range(args.num):
                # do augment
                pass




def get_args() -> argparse.Namespace:
    """Get the arguments for the data augmentation script.
    """
    parser = argparse.ArgumentParser(description='Augment the chute data.')
    parser.add_argument('--data', type=str, default='data/processed/chute_detection.hdf5', help='Directory containing the chute data')
    parser.add_argument('--output_dir', type=str, default='data/augmented', help='Directory to save the augmented data')
    parser.add_argument('--num', type=int, default=3, help='Number of augmentations to create per image')
    parser.add_argument('--fraction', type=float, default=0.75, help='Fraction of images to augment')
    parser.add_argument('--type', type=str, nargs='+', default='rotation translation scaling flipping', help='Type of augmentation to apply. Options: rotation, translation, scaling, flipping')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    augment_chute_data(args)