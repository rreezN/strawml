from __init__ import *

from PIL import Image
import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
import cv2
import time
import timeit
from tqdm import tqdm

from strawml.data.make_dataset import decode_binary_image
from strawml.data.image_utils import rotate_image_and_bbox

def rotate_to_bbox(image, bbox) -> tuple:
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox

    # Calculate the angle of the bounding box
    angle = np.arctan2(y1 - y4, x1 - x4)
    angle_degrees = np.rad2deg(angle)
    
    # Rotate the image and bounding box
    rotated_image, _, rotated_bbox = rotate_image_and_bbox(image, image, bbox, angle_degrees=angle_degrees)
    
    return rotated_image, rotated_bbox


def crop_to_bbox(rotated_image, rotated_bbox) -> tuple:
    cropped_image = rotated_image.copy()
    cropped_bbox = rotated_bbox.copy()
    
    # Get the bounding box coordinates
    x1, y1, x2, y2, x3, y3, x4, y4 = rotated_bbox
    
    # Get the width and height of the bounding box
    width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
    height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
    
    # Crop the image to the bounding box
    cropped_image = cropped_image[int(min(y1, y2, y3, y4)):int(min(y1, y2, y3, y4) + height), int(min(x1, x2, x3, x4)):int(min(x1, x2, x3, x4) + width)]
    
    # Update the bounding box coordinates
    cropped_bbox[0::2] -= min(x1, x2, x3, x4)
    cropped_bbox[1::2] -= min(y1, y2, y3, y4)
    
    return cropped_image, cropped_bbox


def rotate_and_crop_to_bbox(image, bbox) -> tuple:
    rotated_image, rotated_bbox = rotate_to_bbox(image, bbox)
    cropped_image, cropped_bbox = crop_to_bbox(rotated_image, rotated_bbox)
    
    return cropped_image, cropped_bbox


def visualize_image_and_bbox(original_image, original_bbox, rotated_image, rotated_bbox, cropped_image, cropped_bbox) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].plot([original_bbox[0], original_bbox[2], original_bbox[4], original_bbox[6], original_bbox[0]], 
                 [original_bbox[1], original_bbox[3], original_bbox[5], original_bbox[7], original_bbox[1]], 
                 'g-')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(rotated_image)
    axes[1].plot([rotated_bbox[0], rotated_bbox[2], rotated_bbox[4], rotated_bbox[6], rotated_bbox[0]],
                 [rotated_bbox[1], rotated_bbox[3], rotated_bbox[5], rotated_bbox[7], rotated_bbox[1]], 
                 'g-')
    axes[1].set_title('Rotated Image')
    axes[1].axis('off')
    
    axes[2].imshow(cropped_image)
    axes[2].plot([cropped_bbox[0], cropped_bbox[2], cropped_bbox[4], cropped_bbox[6], cropped_bbox[0]], 
                 [cropped_bbox[1], cropped_bbox[3], cropped_bbox[5], cropped_bbox[7], cropped_bbox[1]],
                 'g-')
    axes[2].set_title('Cropped Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    img = Image.fromarray((cropped_image*255).astype(np.uint8))
    img.save('data/cropped_chute_example.png')
    
    

if __name__ == '__main__':
    import h5py
    
    # Load the image and bounding box
    hf = h5py.File('data/interim/chute_detection.hdf5', 'r')
    frames = list(hf.keys())
    if len(frames) < 1:
        raise ValueError('Need at least one frame to demonstrate rotation.')
    frame = frames[0]
    image = decode_binary_image(hf[frame]['image'][...])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bbox = hf[frame]['annotations']['bbox_chute'][...]
    
    # Rotate and crop the image to the bounding box
    rotated_image, rotated_bbox = rotate_to_bbox(image, bbox)
    cropped_image, cropped_bbox = crop_to_bbox(rotated_image, rotated_bbox)
    
    visualize_image_and_bbox(image, bbox, rotated_image, rotated_bbox, cropped_image, cropped_bbox)
    
    # Combined into one step: Rotate and crop the image to the bounding box
    # cropped_image, cropped_bbox = rotate_and_crop_to_bbox(image, bbox)
    
    # Time the rotation and cropping process
    print("Timing the rotation process...")
    num_iterations = 25
    start_time = timeit.default_timer()
    tqdm_iterator = tqdm(range(num_iterations), unit='images', position=0, leave=True)
    for _ in tqdm_iterator:
        _, _ = rotate_to_bbox(image, bbox)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    print(f'Elapsed time for {num_iterations} iterations: {elapsed_time:.2f} seconds')
    print(f'Average time per iteration: {elapsed_time/num_iterations:.6f} seconds')
    
    print("Timing the cropping process...")
    start_time = timeit.default_timer()
    tqdm_iterator = tqdm(range(num_iterations), unit='images', position=0, leave=True)
    for _ in tqdm_iterator:
        _, _ = crop_to_bbox(rotated_image, rotated_bbox)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    print(f'Elapsed time for {num_iterations} iterations: {elapsed_time:.2f} seconds')
    print(f'Average time per iteration: {elapsed_time/num_iterations:.6f} seconds')


