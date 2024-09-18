from __init__ import *

from PIL import Image
import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
import cv2

from strawml.data.make_dataset import decode_binary_image

def rotate_point(x, y, cx, cy, angle_radians):
    """
    Rotates a point (x, y) around a center point (cx, cy) by a given angle in radians.
    
    Args:
    - x, y: Coordinates of the point to rotate.
    - cx, cy: Center of rotation.
    - angle_radians: The rotation angle in radians.

    Returns:
    - (x_new, y_new): The new coordinates of the rotated point.
    """
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    # Translate point back to origin
    x -= cx
    y -= cy

    # Rotate point
    x_new = x * cos_angle - y * sin_angle
    y_new = x * sin_angle + y * cos_angle

    # Translate point back
    x_new += cx
    y_new += cy

    return x_new, y_new

def rotate_bbox(bbox, img_width, img_height, angle_degrees):
    """
    Rotates the bounding box defined by four corner points around the center of the image by a given angle.

    Args:
    - bbox (list or np.ndarray): Bounding box in [x1, y1, x2, y2, x3, y3, x4, y4] format.
    - img_width (int): Width of the image.
    - img_height (int): Height of the image.
    - angle_degrees (float): Angle to rotate the bounding box in degrees.

    Returns:
    - rotated_bbox (np.ndarray): Rotated bounding box coordinates in [x1, y1, x2, y2, x3, y3, x4, y4] format.
    """
    # Convert angle to radians
    angle_radians = np.deg2rad(-angle_degrees)
    
    # Center of the image (we rotate the bounding box around this point)
    cx, cy = img_width / 2, img_height / 2

    # Initialize a list for rotated corner points
    rotated_coords = []

    # Rotate each corner point of the bbox
    for i in range(0, len(bbox), 2):
        x, y = bbox[i], bbox[i + 1]
        x_new, y_new = rotate_point(x, y, cx, cy, angle_radians)
        rotated_coords.append((x_new, y_new))

    # Clip the rotated coordinates to the image boundaries,
    # and flatten the coordinates to match [x1, y1, x2, y2, x3, y3, x4, y4] format
    rotated_coords = np.array(rotated_coords).flatten()
    rotated_bbox = clip_bbox_to_image(rotated_coords, img_width, img_height)

    return rotated_bbox


def clip_bbox_to_image(bbox, img_width, img_height):
    """
    Clips the bounding box coordinates to ensure they stay within the image bounds.
    
    Args:
    - bbox: Bounding box coordinates as a flattened array [x1, y1, x2, y2, ...].
    - img_width: Width of the image.
    - img_height: Height of the image.
    
    Returns:
    - Clipped bounding box coordinates as a flattened array.
    """
    bbox[0::2] = np.clip(bbox[0::2], 0, img_width - 1)  # Clip x-coordinates
    bbox[1::2] = np.clip(bbox[1::2], 0, img_height - 1)  # Clip y-coordinates
    return bbox


def rotate_image_and_bbox(image, image_diff, bbox, angle_degrees):
    """
    Rotates the image and bounding box by the specified angle and clips the bounding box 
    coordinates to remain within the image borders.

    Args:
    - image (np.ndarray): The original image.
    - image_diff (np.ndarray): The image difference.
    - bbox (np.ndarray): Bounding box coordinates in the format [x1, y1, x2, y2, ...].
    - angle_degrees (float): Angle to rotate the image and bounding box in degrees.

    Returns:
    - rotated_image (np.ndarray): The rotated image.
    - rotated_image_diff (np.ndarray): The rotated image difference.
    - rotated_bbox (np.ndarray): The rotated and clipped bounding box coordinates.
    """
    # Step 1: Rotate the image and image_diff
    rotated_image = transform.rotate(image, angle_degrees, resize=False)
    rotated_image_diff = transform.rotate(image_diff, angle_degrees, resize=False)

    # Step 2: Rotate the bounding box corners
    img_height, img_width = rotated_image.shape[:2]
    rotated_bbox = rotate_bbox(bbox, img_width, img_height, angle_degrees)

    # visualise_augmentation(image, image_diff, bbox, rotated_image, rotated_image_diff, rotated_bbox)
    return rotated_image, rotated_image_diff, rotated_bbox



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
    
    

if __name__ == '__main__':
    import h5py
    
    # Load the image and bounding box
    hf = h5py.File('data/interim/chute_detection.hdf5', 'r')
    frames = list(hf.keys())
    if len(frames) < 1:
        raise ValueError('Need at least one frame to demonstrate rotation.')
    frame = frames[0]
    image = decode_binary_image(hf[frame]['image'][...])
    bbox = hf[frame]['annotations']['bbox_chute'][...]
    
    # Rotate and crop the image to the bounding box
    rotated_image, rotated_bbox = rotate_to_bbox(image, bbox)
    cropped_image, cropped_bbox = crop_to_bbox(rotated_image, rotated_bbox)
    
    visualize_image_and_bbox(image, bbox, rotated_image, rotated_bbox, cropped_image, cropped_bbox)
    
    # Combined into one step: Rotate and crop the image to the bounding box
    # cropped_image, cropped_bbox = rotate_and_crop_to_bbox(image, bbox)
    



