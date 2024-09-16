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
from skimage import transform
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

def get_nonzero_coordinates(image):
    # Convert the image to a NumPy array (if it isn't already)
    image_array = np.array(image[:,:,0])
    label_array = np.array(image[:,:,1])

    # Find the indices where the image is not zero
    nonzero_indices = np.nonzero(image_array)

    # Combine indices to get coordinates
    coordinates = list(zip(nonzero_indices[1], nonzero_indices[0]))  # (x, y)

    # run through the coordiantes and get the label for each coordinate -> order the coordinates by label number (1, 2, 3, ...)
    coordinates = sorted(coordinates, key=lambda x: label_array[x[1], x[0]])

    return coordinates


def rotate_image_and_bbox(image, image_diff, bbox, angle_degrees):
    bbox = bbox.astype(int)
    # Add an extra dimension to the image that is binary and contains the position of the bounding box
    # This will be used to rotate the bounding box
    bbox_image = np.zeros_like(image[:,:,:2])
    # now we take the image and add the bounding box to the extra dimension

    count = 1
    for i in range(0, len(bbox), 2):
        x, y = bbox[i], bbox[i+1]
        bbox_image[y, x, 0] = 1
        bbox_image[y, x, 1] = count
        count += 1
        
    # Step 1: Rotate the image using skimage's rotate function
    rotated_image = transform.rotate(image, angle_degrees, resize=False)
    rotated_image_diff = transform.rotate(image_diff, angle_degrees, resize=False)

    # We now also rotate the bounding box image
    rotated_bbox_image = transform.rotate(bbox_image, angle_degrees, resize=False)
    rotated_bbox = get_nonzero_coordinates(rotated_bbox_image)
    rotated_bbox = np.array(rotated_bbox).flatten()

    # Draw the rotated bounding box
    # ensure that the bounding box is a rectangle
    if not len(rotated_bbox) == 8:
        return False, image, image_diff, bbox

    x1, y1, x2, y2, x3, y3, x4, y4 = rotated_bbox
    cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(rotated_image, (x2, y2), (x3, y3), (0, 255, 0), 2)
    cv2.line(rotated_image, (x3, y3), (x4, y4), (0, 255, 0), 2)
    cv2.line(rotated_image, (x4, y4), (x1, y1), (0, 255, 0), 2)
    
    cv2.imshow("Rotated Image", rotated_image)
    cv2.waitKey(0)
    return True, rotated_image, rotated_image_diff, rotated_bbox
    
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
        
        if random.random() <= args.fraction:
            for i in range(args.num):
                if args.type == 'rotation':
                    angle = random.randint(-90, 90)
                    success, rotated_image, rotated_image_diff, rotated_bbox = rotate_image_and_bbox(image, image_diff, bbox_chute, angle)
                    if success:
                        # Save the rotated image and bbo
                        group = hf.create_group(f"frame_{frame_nr}")
                        group.create_dataset('image', data=rotated_image)
                        group.create_dataset('image_diff', data=rotated_image_diff)
                        sub_group = group.create_group('annotations')
                        sub_group.create_dataset('bbox_chute', data=rotated_bbox)
                        sub_group.create_dataset('fullness', data=hf[frame]['annotations']['fullness'][...])
                        sub_group.create_dataset('obstructed', data=hf[frame]['annotations']['obstructed'][...])
                        frame_nr += 1
                if args.type == 'translation':
                    ...
                if args.type == 'scaling':
                    ...
                if args.type == 'flipping':
                    ...
    hf.close()

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