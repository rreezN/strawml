from __init__ import *

from PIL import Image
import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
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

    # Find the rotation angle of the bounding box
    dy = y4 - y2
    dx = x4 - x2
    m = dy/dx
    angle = np.arctan(m)
    d = angle - 45
    
    # Rotate the image and bounding box
    rotated_image, _, rotated_bbox = rotate_image_and_bbox(image, image, bbox, angle_degrees=d)
    
    return rotated_image, rotated_bbox


def visualize_image_and_bbox(image, bbox) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.plot([bbox[0], bbox[2], bbox[4], bbox[6], bbox[0]], [bbox[1], bbox[3], bbox[5], bbox[7], bbox[1]], 'r-')
    plt.show()
    
    

if __name__ == '__main__':
    import h5py
    
    # Load the image and bounding box
    hf = h5py.File('data/interim/chute_detection.hdf5', 'r')
    frames = list(hf.keys())
    image = decode_binary_image(hf[frames[0]]['image'][...])
    bbox = hf[frames[0]]['annotations']['bbox_chute'][...]
    
    # Rotate the image to the bounding box
    rotated_image, rotated_bbox = rotate_to_bbox(image, bbox)
    
    visualize_image_and_bbox(rotated_image, rotated_bbox)



