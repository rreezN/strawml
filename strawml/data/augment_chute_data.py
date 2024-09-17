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


def load_image(hf_path, frame):
    with h5py.File(hf_path, 'r') as hf:
        image = decode_binary_image(hf[frame]['image'][...])
        image_diff = decode_binary_image(hf[frame]['image_diff'][...])
        bbox_chute = hf[frame]['annotations']['bbox_chute'][...]
    return image, image_diff, bbox_chute

def visualise_augmentation(image, image_diff, bbox, augmented_image, augmented_image_diff, augmented_bbox):
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax[0,0].imshow(image)
    ax[0,1].imshow(image_diff)
    # Draw the bounding box on the image and make sure the bounding box is closed
    ax[0,0].plot([bbox[0], bbox[2], bbox[4], bbox[6], bbox[0]], 
                 [bbox[1], bbox[3], bbox[5], bbox[7], bbox[1]], 'r')
    ax[0,0].set_title('Original image')

    ax[1,0].imshow(augmented_image)
    ax[1,1].imshow(augmented_image_diff)
    ax[1,0].plot([augmented_bbox[0], augmented_bbox[2], augmented_bbox[4], augmented_bbox[6], augmented_bbox[0]], 
                 [augmented_bbox[1], augmented_bbox[3], augmented_bbox[5], augmented_bbox[7], augmented_bbox[1]], 'r')
    ax[1,0].set_title('Augmented image')
    plt.tight_layout()
    plt.show()

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
    
def translate_image_and_bbox(image, image_diff, bbox, x, y):
    """
    Translate the image and image_diff by x and y pixels. 
    The bounding box is also translated accordingly.

    Args:
    - image (np.ndarray): The original image.
    - image_diff (np.ndarray): The image difference.
    - bbox (list): A list of 4 values representing the bounding box in the format [x_min, y_min, x_max, y_max].
    - x (int): Number of pixels to translate along the x-axis (horizontal shift).
    - y (int): Number of pixels to translate along the y-axis (vertical shift).

    Returns:
    - translated_image (np.ndarray): The translated image.
    - translated_image_diff (np.ndarray): The translated image_diff.
    - translated_bbox (list): The translated bounding box, clamped to image dimensions.
    """
    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Step 1: Translate the image using numpy roll
    translated_image = np.zeros_like(image)
    translated_image_diff = np.zeros_like(image_diff)

    # Compute the slicing coordinates based on x and y translation
    # Handle translation for x and y independently to ensure correct shape matching
    start_y_src = max(0, -y)
    start_y_dst = max(0, y)
    height = img_height - abs(y)
    
    start_x_src = max(0, -x)
    start_x_dst = max(0, x)
    width = img_width - abs(x)

    # Translate the image and image_diff by x and y pixels
    translated_image[start_y_dst:start_y_dst + height, start_x_dst:start_x_dst + width] = \
        image[start_y_src:start_y_src + height, start_x_src:start_x_src + width]

    translated_image_diff[start_y_dst:start_y_dst + height, start_x_dst:start_x_dst + width] = \
        image_diff[start_y_src:start_y_src + height, start_x_src:start_x_src + width]

    # Step 2: Translate the bounding box by x and y pixels
    translated_bbox = bbox.copy()
    translated_bbox[0::2] += x
    translated_bbox[1::2] += y
    translated_bbox = clip_bbox_to_image(translated_bbox, img_width, img_height)

    # visualise_augmentation(image, image_diff, bbox, translated_image, translated_image_diff, translated_bbox)
    return translated_image, translated_image_diff, translated_bbox

def rescale_image_and_bbox(image, image_diff, bbox, scale_factor):
    """
    Rescale the image, image_diff, and bounding box by the given scale factor.

    Args:
    - image (np.ndarray): The original image.
    - image_diff (np.ndarray): The image difference.
    - bbox (list): A list of 8 values representing the bounding box as 4 corner points [x1, y1, x2, y2, x3, y3, x4, y4].
    - scale_factor (float): The factor by which to scale the image and bounding box.

    Returns:
    - rescaled_image (np.ndarray): The rescaled image.
    - rescaled_image_diff (np.ndarray): The rescaled image difference.
    - rescaled_bbox (list): The rescaled bounding box.
    """

    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Step 1: Rescale the image
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    rescaled_image = transform.resize(image, (new_height, new_width), anti_aliasing=True, preserve_range=True).astype(image.dtype)
    rescaled_image_diff = transform.resize(image_diff, (new_height, new_width), anti_aliasing=True, preserve_range=True).astype(image_diff.dtype)

    # Step 2: Rescale the bounding box
    rescaled_bbox = bbox.copy()
    for i in range(0, len(rescaled_bbox), 2):
        rescaled_bbox[i] = int(rescaled_bbox[i] * scale_factor)  # Rescale x coordinates
        rescaled_bbox[i + 1] = int(rescaled_bbox[i + 1] * scale_factor)  # Rescale y coordinates

    # Step 3: Clip the rescaled bounding box to the new image dimensions
    rescaled_bbox = clip_bbox_to_image(rescaled_bbox, new_width, new_height)

    # Visualize the result (assuming `visualise_augmentation` is defined elsewhere)
    # visualise_augmentation(image, image_diff, bbox, rescaled_image, rescaled_image_diff, rescaled_bbox)

    return rescaled_image, rescaled_image_diff, rescaled_bbox
    
def crop_image_and_bbox(image, image_diff, bbox, x, y, w, h): 
    # crop image with x and y as center points and w and h as width and height of the cropped image
    cropped_image = image[y - h//2:y + h//2, x - w//2:x + w//2]
    cropped_image_diff = image_diff[y - h//2:y + h//2, x - w//2:x + w//2]
    # crop the bounding box
    cropped_bbox = bbox.copy()
    cropped_bbox[0::2] -= x - w//2
    cropped_bbox[1::2] -= y - h//2
    cropped_bbox = clip_bbox_to_image(cropped_bbox, w, h)

    # Now we need to resize the image and the bounding box to the original size
    cropped_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]))
    cropped_image_diff = cv2.resize(cropped_image_diff, (image_diff.shape[1], image_diff.shape[0]))
    cropped_bbox[0::2] = cropped_bbox[0::2] * image.shape[1] / w
    cropped_bbox[1::2] = cropped_bbox[1::2] * image.shape[0] / h

    # visualise_augmentation(image, image_diff, bbox, cropped_image, cropped_image_diff, cropped_bbox)
    return cropped_image, cropped_image_diff, cropped_bbox


def save_frames_to_hdf5(hf_path, frame, frame_nr, augmented_image, augmented_image_diff, augmented_bbox, augmentation):
    # Save the rotated image and bbo
    with h5py.File(hf_path, 'a') as hf:
        group = hf.create_group(f"frame_{frame_nr}")
        group.create_dataset('image', data=augmented_image)
        group.create_dataset('image_diff', data=augmented_image_diff)
        sub_group = group.create_group('annotations')
        sub_group.create_dataset('bbox_chute', data=augmented_bbox)
        sub_group.create_dataset('fullness', data=hf[frame]['annotations']['fullness'][...])
        sub_group.create_dataset('obstructed', data=hf[frame]['annotations']['obstructed'][...])
        # Add the video ID as an attribute to the dataset
        group.attrs['video ID'] = hf[frame].attrs['video ID']
        group.attrs['augmented'] = augmentation


def augment_chute_data(args):
    # Copy file from args.data to args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    os.system(f"cp {args.data} {args.output_dir}/")
    # Open the file
    hf_path = args.output_dir + "/" + args.data.split("/")[-1]    
    hf = h5py.File(hf_path, 'r')   
    # Based on hf.keys() get the largest number of frames and add 1 to get the next frame number
    frame_nr = max([int(frame.split('_')[1]) for frame in hf.keys()]) + 1
    frame_keys = list(hf.keys())
    hf.close()

    for frame in frame_keys:
        prob = random.random() 
        # print(f"Frame: {frame}, Probability: {prob}")
        image, image_diff, bbox_chute = load_image(hf_path, frame)
        if prob <= args.fraction:
            for _ in range(args.num):
                if 'rotation' in args.type:
                    # print(f"Rotation, with frame: {frame_nr}")
                    angle = random.randint(-90, 90)
                    rotated_image, rotated_image_diff, rotated_bbox = rotate_image_and_bbox(image, image_diff, bbox_chute, angle)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, rotated_image, rotated_image_diff, rotated_bbox, "rotation")
                    frame_nr += 1

                if 'translation' in args.type:
                    # print(f"Translation, with frame: {frame_nr}")
                    x_trans = random.randint(-200, 200)
                    y_trans = random.randint(-200, 200)
                    translated_image, translated_image_diff, translated_bbox = translate_image_and_bbox(image, image_diff, bbox_chute, x_trans, y_trans)        
                    save_frames_to_hdf5(hf_path, frame, frame_nr, translated_image, translated_image_diff, translated_bbox, "translation")
                    frame_nr += 1

                if 'cropping' in args.type:
                    # print(f"Cropping, with frame: {frame_nr}")
                    cx = random.randint(1000, 1500)
                    cy = random.randint(200, 1100)
                    w = random.randint(200, 800)
                    h = random.randint(200, 800)
                    cropped_image, cropped_image_diff, cropped_bbox = crop_image_and_bbox(image, image_diff, bbox_chute, cx, cy, w, h)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, cropped_image, cropped_image_diff, cropped_bbox, "cropping")
                    frame_nr += 1

def get_args() -> argparse.Namespace:
    """Get the arguments for the data augmentation script.
    """
    parser = argparse.ArgumentParser(description='Augment the chute data.')
    parser.add_argument('--data', type=str, default='data/interim/chute_detection.hdf5', help='Directory containing the chute data')
    parser.add_argument('--output_dir', type=str, default='data/processed/augmented', help='Directory to save the augmented data')
    parser.add_argument('--num', type=int, default=1, help='Number of augmentations to create per image')
    parser.add_argument('--fraction', type=float, default=0.75, help='Fraction of images to augment')
    parser.add_argument('--type', type=str, nargs='+', default='rotation translation cropping', help='Type of augmentation to apply. Options: rotation, translation, cropping')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    augment_chute_data(args)