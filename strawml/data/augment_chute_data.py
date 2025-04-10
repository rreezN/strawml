from __init__ import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import h5py
import random
import shutil
from skimage import transform
from tqdm import tqdm
import json
from torchvision.transforms import v2 as transforms

from make_dataset import decode_binary_image
from strawml.data.image_utils import rotate_point, rotate_bbox, clip_bbox_to_image, rotate_image_and_bbox

def load_image(hf_path: str, 
               frame_nr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an image, image difference, and bounding box from an HDF5 file, based
    on the frame number and the file path.

    ...

    Parameters
    ----------
    hf_path : str
        The path to the HDF5 file.
    frame_nr : str
        The frame number to load from the HDF5 file.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the image, image difference, and bounding box.
    """
    with h5py.File(hf_path, 'r') as hf:
        image = decode_binary_image(hf[frame_nr]['image'][...])
        image_diff = decode_binary_image(hf[frame_nr]['image_diff'][...])
        bbox_chute = hf[frame_nr]['annotations']['bbox_chute'][...]
    return image, image_diff, bbox_chute

    
    
def visualise_augmentation(image: np.ndarray, 
                           image_diff: np.ndarray, 
                           bbox: np.ndarray, 
                           augmented_image: np.ndarray, 
                           augmented_image_diff: np.ndarray, 
                           augmented_bbox: np.ndarray) -> None:
    """
    Visualise the original and augmented images side by side, with the bounding box drawn on each image.

    ...

    Parameters
    ----------
    image : np.ndarray
        The original image.
    image_diff : np.ndarray
        The image difference.
    bbox : np.ndarray
        The bounding box coordinates.
    augmented_image : np.ndarray
        The augmented image.
    augmented_image_diff : np.ndarray
        The augmented image difference.
    augmented_bbox : np.ndarray
        The augmented bounding box coordinates.
    """
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

def is_continuous(numbers: list) -> bool:
    """
    Checks if a list of numbers is continuous (i.e., the difference between consecutive numbers is 1).

    ...

    Parameters
    ----------
    numbers : list
        A list of numbers to check for continuity.

    Returns
    -------
    bool
        True if the numbers are continuous, False otherwise.
    """
    # Sort the list of numbers 
    sorted_numbers = sorted(numbers)
    # Check if the difference between consecutive numbers is 1
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i + 1] - sorted_numbers[i] != 1:
            print(f"Missing frame: {sorted_numbers[i] + 1}")
            return False
    return True

def get_nonzero_coordinates(image: np.ndarray) -> list:
    """
    Get the coordinates of all non-zero pixels in an image.

    ...

    Parameters
    ----------
    image : np.ndarray
        The image to extract the coordinates from.  
    
    Returns
    -------
    list
        A list of (x, y) coordinates of non-zero pixels in the image.
    """
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
    
def translate_image_and_bbox(image: np.ndarray, 
                             image_diff: np.ndarray, 
                             bbox: np.ndarray, 
                             x: int, 
                             y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Translate the image and image_diff by x and y pixels. The bounding box is 
    also translated accordingly.

    ...

    Parameters
    ----------
    image : np.ndarray
        The original image.
    image_diff : np.ndarray
        The image difference.
    bbox : np.ndarray
        Bounding box coordinates in the format [x1, y1, x2, y2, ...].
    x : int
        Number of pixels to translate along the x-axis (horizontal shift).
    y : int
        Number of pixels to translate along the y-axis (vertical shift).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the translated image, translated image difference, and translated bounding box.
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

def rescale_image_and_bbox(image: np.ndarray, image_diff: np.ndarray, bbox: np.ndarray, scale_factor: float):
    """
    Rescale the image, image_diff, and bounding box by the given scale factor.

    ...

    Parameters
    ----------
    image : np.ndarray
        The original image.
    image_diff : np.ndarray
        The image difference.
    bbox : np.ndarray
        Bounding box coordinates in the format [x1, y1, x2, y2, ...].
    scale_factor : float
        The factor by which to scale the image and bounding box.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the rescaled image, rescaled image difference, and rescaled bounding box.
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
    
def crop_image_and_bbox(image: np.ndarray, 
                        image_diff: np.ndarray, 
                        bbox: np.ndarray, 
                        x: int, 
                        y: int, 
                        w: int, 
                        h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop the image and image_diff around the center point (x, y) with width w and height h.
    The bounding box is adjusted accordingly.

    ...

    Parameters
    ----------
    image : np.ndarray
        The original image.
    image_diff : np.ndarray
        The image difference.
    bbox : np.ndarray
        Bounding box coordinates in the format [x1, y1, x2, y2, ...].
    x : int
        The x-coordinate of the center point.
    y : int
        The y-coordinate of the center point.
    w : int
        The width of the cropped image.
    h : int
        The height of the cropped image.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the cropped image, cropped image difference, and cropped bounding box.
    """
    # Ensure the cropping coordinates are within the image bounds
    y1, y2 = max(0, y - h // 2), min(image.shape[0], y + h // 2)
    x1, x2 = max(0, x - w // 2), min(image.shape[1], x + w // 2)
    
    # Crop the image and image_diff
    cropped_image = image[y1:y2, x1:x2]
    cropped_image_diff = image_diff[y1:y2, x1:x2]
    
    # Adjust the bounding box
    cropped_bbox = bbox.copy()
    cropped_bbox[0::2] -= x1
    cropped_bbox[1::2] -= y1
    cropped_bbox = clip_bbox_to_image(cropped_bbox, x2 - x1, y2 - y1)
    
    # Resize the cropped image and bounding box to the original size
    cropped_image = transform.resize(cropped_image, (image.shape[0], image.shape[1]), anti_aliasing=True, preserve_range=True).astype(np.uint8)
    cropped_image_diff = transform.resize(cropped_image_diff, (image_diff.shape[0], image_diff.shape[1]), anti_aliasing=True, preserve_range=True).astype(np.uint8)
    cropped_bbox[0::2] = cropped_bbox[0::2] * image.shape[1] / (x2 - x1)
    cropped_bbox[1::2] = cropped_bbox[1::2] * image.shape[0] / (y2 - y1)
    
    return cropped_image, cropped_image_diff, cropped_bbox

def gamma_adjust(image: np.ndarray,
                image_diff: np.ndarray,
                bbox: np.ndarray,
                gamma: float,
                ):
    """
    Alters the color composition of the image and image_diff using the given parameters.
    Done to account for different lighting conditions in the dataset.

    ...

    Parameters
    ----------
    image : np.ndarray
        The original image.
    image_diff : np.ndarray
        The image difference.
    bbox : np.ndarray
        Bounding box coordinates in the format [x1, y1, x2, y2, ...].
    gamma : float
        The gamma factor to apply to the image.
    gaussian_noise : float
        The standard deviation of the Gaussian noise to add to the image.        

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the color-adjusted image, color-adjusted image difference, and bounding box.
    """
    from skimage import exposure
    # Apply gamma correction and add Gaussian noise to the image
    image = exposure.adjust_gamma(image, gamma)

    # Apply gamma correction and add Gaussian noise to the image difference
    image_diff = exposure.adjust_gamma(image_diff, gamma)

    return image, image_diff, bbox

def gaussian_noise(image: np.ndarray, 
                   image_diff: np.ndarray, 
                   bbox: np.ndarray, 
                   mean: float, 
                   std: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to the image and image_diff.

    ...

    Parameters
    ----------
    image : np.ndarray
        The original image.
    image_diff : np.ndarray
        The image difference.
    bbox : np.ndarray
        Bounding box coordinates in the format [x1, y1, x2, y2, ...].
    mean : float
        The mean of the Gaussian noise.
    std : float
        The standard deviation of the Gaussian noise.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the noisy image, noisy image difference, and bounding box.
    """
    # Add Gaussian noise to the image and image_diff
    noisy_image = cv2.add(image, np.random.normal(mean, std, image.shape).astype(np.uint8))
    noisy_image_diff = cv2.add(image_diff, np.random.normal(mean, std, image_diff.shape).astype(np.uint8))

    return noisy_image, noisy_image_diff, bbox

def color_jitter(image: np.ndarray,
                 image_diff: np.ndarray,
                bbox: np.ndarray,
                brightness: float,
                hue: float,
                contrast: float,
                saturation: float,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    
    # Create torch transform
    transform = transforms.ColorJitter(brightness=brightness, hue=hue, contrast=contrast, saturation=saturation)
    image = transform(image)
    image_diff = transform(image_diff)
    
    return image, image_diff, bbox

def gaussian_blur(image: np.ndarray,
                    image_diff: np.ndarray,
                    bbox: np.ndarray,
                    kernel_size: int,
                    sigma: float,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    image = transform(image)
    image_diff = transform(image_diff)
    
    return image, image_diff, bbox

def random_invert(image: np.ndarray,
                    image_diff: np.ndarray,
                    bbox: np.ndarray,
                    p: float,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = transforms.RandomInvert(p=p)
    image = transform(image)
    image_diff = transform(image_diff)
    
    return image, image_diff, bbox

def jpeg_compression(image: np.ndarray,
                    image_diff: np.ndarray,
                    bbox: np.ndarray,
                    quality: int,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """compress the image using JPEG compression

    Args:
        quality (int): The quality of the JPEG compression 0-100 (higher is more compressed)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    quality = random.randint(100-quality, 100)
    transform = transforms.JPEG(quality=quality)
    image = transform(image)
    image_diff = transform(image_diff)

    return image, image_diff, bbox

def rgbshift(image: np.ndarray,
             image_diff: np.ndarray,
             bbox: np.ndarray,
             r_shift: tuple,
             g_shift: tuple,
             b_shift: tuple,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift the RGB channels of the image
    
    
    Args:
        r_shift (tuple): The shift of the red channel
        g_shift (tuple): The shift of the green channel
        b_shift (tuple): The shift of the blue channel
        
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    r_shift = random.randint(r_shift[0], r_shift[1])
    g_shift = random.randint(g_shift[0], g_shift[1])
    b_shift = random.randint(b_shift[0], b_shift[1])
    
    image[:,:,0] = np.clip(image[:,:,0] + r_shift, 0, 255)
    image[:,:,1] = np.clip(image[:,:,1] + g_shift, 0, 255)
    image[:,:,2] = np.clip(image[:,:,2] + b_shift, 0, 255)
    
    image_diff[:,:,0] = np.clip(image_diff[:,:,0] + r_shift, 0, 255)
    image_diff[:,:,1] = np.clip(image_diff[:,:,1] + g_shift, 0, 255)
    image_diff[:,:,2] = np.clip(image_diff[:,:,2] + b_shift, 0, 255)
    
    return image, image_diff, bbox


def save_frames_to_hdf5(hf_path: str, 
                        frame: str, 
                        frame_nr: int, 
                        augmented_image: np.ndarray, 
                        augmented_image_diff: np.ndarray, 
                        augmented_bbox: np.ndarray, 
                        augmentation: str) -> None:
    
    """
    Function to save the augmented image and bounding box to an HDF5 file.

    ...

    Parameters
    ----------
    hf_path : str
        The path to the HDF5 file.
    frame : str
        The frame number to load from the HDF5 file.
    frame_nr : int
        The frame number to save the augmented image to.
    augmented_image : np.ndarray
        The augmented image.
    augmented_image_diff : np.ndarray
        The augmented image difference.
    augmented_bbox : np.ndarray
        The augmented bounding box.
    augmentation : str
        The type of augmentation applied to the image.
    """
    # Save the rotated image and bbo
    with h5py.File(hf_path, 'a') as hf:
        augmented_image = cv2.imencode('.jpg', augmented_image)[1]
        augmented_image_diff = cv2.imencode('.jpg', augmented_image_diff)[1]
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

def augment_chute_data(args: argparse.Namespace) -> None:
    """
    Main wrapper function to augment the chute data.

    ...

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the script.
    """
    # Copy file from args.data to args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # delete the file if it already exists
    if os.path.exists(args.output_dir + "/" + args.data.split("/")[-1]):
        os.remove(f"{args.output_dir}/{args.data.split('/')[-1]}")
    shutil.copy(args.data, args.output_dir)
    # Open the file
    hf_path = args.output_dir + "/" + args.data.split("/")[-1]    
    hf = h5py.File(hf_path, 'r')   
    # Based on hf.keys() get the largest number of frames and add 1 to get the next frame number
    frame_nr = max([int(frame.split('_')[1]) for frame in hf.keys()]) + 1
    frame_keys = list(hf.keys())
    hf.close()
    pbar = tqdm(total=len(frame_keys), desc="Augmenting frames", position=0)
    frame_to_augment_dict = {}
    for frame in frame_keys:
        prob = random.random() 
        image, image_diff, bbox_chute = load_image(hf_path, frame)
        if image is None:
            print(f"Could not load image for frame: {frame}")
            continue
        augment_frame_nrs = []
        if prob <= args.fraction:
            pbar.set_description_str(f"Frame: {frame}, Probability: {round(prob, 2)}")
            for _ in range(args.num):
                if 'rotation' in args.type:
                    # print(f"Rotation, with frame: {frame_nr}")
                    angle = random.randint(-45, 45)
                    rotated_image, rotated_image_diff, rotated_bbox = rotate_image_and_bbox(image, image_diff, bbox_chute, angle)
                    # visualise_augmentation(image, image_diff, bbox_chute, rotated_image, rotated_image_diff, rotated_bbox)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, rotated_image, rotated_image_diff, rotated_bbox, "rotation")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1

                if 'translation' in args.type:
                    # print(f"Translation, with frame: {frame_nr}")
                    x_trans = random.randint(-500, 500)
                    y_trans = random.randint(-500, 500)
                    translated_image, translated_image_diff, translated_bbox = translate_image_and_bbox(image, image_diff, bbox_chute, x_trans, y_trans)  
                    save_frames_to_hdf5(hf_path, frame, frame_nr, translated_image, translated_image_diff, translated_bbox, "translation")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1

                if 'cropping' in args.type:
                    # print(f"Cropping, with frame: {frame_nr}")
                    cx = random.randint(1200, 1500)
                    cy = random.randint(200, 1100)
                    w = random.randint(400, 800)
                    h = random.randint(400, 800)
                    cropped_image, cropped_image_diff, cropped_bbox = crop_image_and_bbox(image, image_diff, bbox_chute, cx, cy, w, h)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, cropped_image, cropped_image_diff, cropped_bbox, "cropping")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1

                if 'gamma' in args.type:
                    # TODO: Split color and noise augmentations
                    # print(f"Color, with frame: {frame_nr}")
                    gamma = random.uniform(0.5, 2)
                    colored_image, colored_image_diff, colored_bbox = gamma_adjust(image, image_diff, bbox_chute, gamma)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, colored_image, colored_image_diff, colored_bbox, "color")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                    
                if 'gaussian_noise' in args.type:
                    # print(f"Gaussian noise, with frame: {frame_nr}")
                    mean = 0
                    std = random.uniform(0, 1)
                    noisy_image, noisy_image_diff, noisy_bbox = gaussian_noise(image, image_diff, bbox_chute, mean, std)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, noisy_image, noisy_image_diff, noisy_bbox, "gaussian_noise")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                
        frame_to_augment_dict[frame] = augment_frame_nrs
        pbar.update(1)
    # save the frame to augment dictionary to jsn file
    with open(args.output_dir + "/frame_to_augment.json", "w") as f:
        json.dump(frame_to_augment_dict, f)
        

def augment_straw_level_data(args: argparse.Namespace) -> None:
    """
    Main wrapper function to augment the straw data.

    ...

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the script.
    """
    # Copy file from args.data to args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    hf_path = args.output_dir + "/" + 'straw_level.hdf5'   
    # delete the file if it already exists
    if os.path.exists(hf_path):
        os.remove(hf_path)
    shutil.copy(args.data, args.output_dir)
    os.rename(args.output_dir + "/" + args.data.split("/")[-1], hf_path)
    # Open the file
    hf = h5py.File(hf_path, 'r')   
    # Based on hf.keys() get the largest number of frames and add 1 to get the next frame number
    frame_nr = max([int(frame.split('_')[1]) for frame in hf.keys()]) + 1
    frame_keys = list(hf.keys())
    hf.close()
    pbar = tqdm(total=len(frame_keys), desc="Augmenting frames", position=0)
    frame_to_augment_dict = {}
    for frame in frame_keys:
        prob = random.random() 
        image, image_diff, bbox_chute = load_image(hf_path, frame)
        if image is None:
            print(f"Could not load image for frame: {frame}")
            continue
        augment_frame_nrs = []
        if prob <= args.fraction:
            pbar.set_description_str(f"Frame: {frame}, Probability: {round(prob, 2)}")
            for _ in range(args.num):
                if 'gamma' in args.type:
                    # print(f"Color, with frame: {frame_nr}")
                    gamma = random.uniform(0.5, 2)
                    colored_image, colored_image_diff, colored_bbox = gamma_adjust(image, image_diff, bbox_chute, gamma)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, colored_image, colored_image_diff, colored_bbox, "color")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                    
                if 'gaussian_noise' in args.type:
                    # print(f"Gaussian noise, with frame: {frame_nr}")
                    mean = 0
                    std = random.uniform(0, 1)
                    noisy_image, noisy_image_diff, noisy_bbox = gaussian_noise(image, image_diff, bbox_chute, mean, std)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, noisy_image, noisy_image_diff, noisy_bbox, "gaussian_noise")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                
                if 'color_jitter' in args.type:
                    cj_image, cj_image_diff, cj_bbox = color_jitter(image, image_diff, bbox_chute, 0.5, 0.5, 0.5, 0.5)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, cj_image, cj_image_diff, cj_bbox, "color_jitter")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                
                if 'gaussian_blur' in args.type:
                    gb_image, gb_image_diff, gb_bbox = gaussian_blur(image, image_diff, bbox_chute, 5, 1)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, gb_image, gb_image_diff, gb_bbox, "gaussian_blur")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                    
                if 'random_invert' in args.type:
                    ri_image, ri_image_diff, ri_bbox = random_invert(image, image_diff, bbox_chute, 1.0)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, ri_image, ri_image_diff, ri_bbox, "random_invert")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                
                if 'jpeg_compression' in args.type:
                    jpg_image, jpg_image_diff, jpg_bbox = jpeg_compression(image, image_diff, bbox_chute, 50)
                    save_frames_to_hdf5(hf_path, frame, frame_nr, jpg_image, jpg_image_diff, jpg_bbox, "jpeg_compression")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                
                if 'rgbshift' in args.type:
                    rgb_image, rgb_image_diff, rgb_bbox = rgbshift(image, image_diff, bbox_chute, (-50, 50), (-50, 50), (-50, 50))
                    save_frames_to_hdf5(hf_path, frame, frame_nr, rgb_image, rgb_image_diff, rgb_bbox, "rgbshift")
                    augment_frame_nrs += [frame_nr]
                    frame_nr += 1
                
                
        frame_to_augment_dict[frame] = augment_frame_nrs
        pbar.update(1)
    # save the frame to augment dictionary to jsn file
    with open(args.output_dir + "/frame_to_augment.json", "w") as f:
        json.dump(frame_to_augment_dict, f)       
            

def get_args() -> argparse.Namespace:
    """
    Get the arguments for the data augmentation script.

    Returns
    -------
    argparse.Namespace
        The arguments passed to the script
    """
    parser = argparse.ArgumentParser(description='Augment the chute data.')
    parser.add_argument('mode', type=str, help='Mode of operation. Options: chute, straw', choices=['chute', 'straw'])
    parser.add_argument('--data', type=str, default='data/interim/chute_detection.hdf5', help='Directory containing the chute data')
    parser.add_argument('--output_dir', type=str, default='data/processed/augmented', help='Directory to save the augmented data')
    parser.add_argument('--num', type=int, default=1, help='Number of augmentations to create per image')
    parser.add_argument('--fraction', type=float, default=0.75, help='Fraction of images to augment')
    # parser.add_argument('--balance', type=bool, default=False, help='Balance the dataset by augmenting the minority class')
    parser.add_argument('--type', type=str, nargs='+', default='rotation translation cropping gamma gaussian_noise color_jitter gaussian_blur random_invert jpeg_compression rgbshift', help='Type of augmentation to apply. Options: rotation, translation, cropping, gamma, gaussian_noise, color_jitter, gaussian_blur, random_invert, jpeg_compression, rgbshift')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'chute':
        augment_chute_data(args)
    elif args.mode == 'straw':
        augment_straw_level_data(args)
    else:
        raise ValueError("Invalid mode. Options: chute, straw")
