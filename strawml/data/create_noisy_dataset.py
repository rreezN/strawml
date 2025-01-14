from __init__ import *

from __init__ import *
import torch
import numpy as np
import h5py
import os
from PIL import ImageOps, Image
from collections.abc import Generator
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
# from wand.image import Image


from strawml.data.make_dataset import decode_binary_image
from strawml.data.image_utils import rotate_image_and_bbox, rotate_bbox, clip_bbox_to_image
from strawml.models.straw_classifier import chute_cropper as cc

def overlay_image_alpha(img, img_overlay, x, y):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    
    # Get alpha mask
    alpha_mask = img_overlay[:, :, 3] / 255.0
    
    img_overlay = img_overlay[:, :, :3]
    img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
    
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    
def extend_overlay_to_fit(background, overlay, x, y):
    """
    Extend the edges of the overlay image to cover the entire background using edge padding.
    """
    # Calculate the padding required on each side
    top_padding = max(0, y)
    bottom_padding = max(0, background.shape[0] - (y + overlay.shape[0]))
    left_padding = max(0, x)
    right_padding = max(0, background.shape[1] - (x + overlay.shape[1]))

    # Pad the overlay by extending the edges
    overlay_padded = cv2.copyMakeBorder(
        overlay,
        top=top_padding,
        bottom=bottom_padding,
        left=left_padding,
        right=right_padding,
        borderType=cv2.BORDER_REPLICATE
    )

    # Crop the padded overlay to match the background size exactly
    overlay_resized = overlay_padded[:background.shape[0], :background.shape[1]]

    return overlay_resized

def overlay_image_addition_with_edge_padding(background, overlay, x, y):
    """
    Overlay `overlay` onto `background` at (x, y) by adding pixel values.
    Extend the overlay using edge padding to cover the entire background.
    """
    # Extend the overlay to match the background size
    overlay_extended = extend_overlay_to_fit(background, overlay, x, y)

    # Add the pixel values from the overlay to the background
    result = cv2.add(background, overlay_extended)

    return result

class BarrelDeformer:
    def __init__(self, w, h, k1=0.2, k2=0.05):
        self.w = w
        self.h = h
        # adjust k_1 and k_2 to achieve the required distortion
        self.k_1 = k1
        self.k_2 = k2
    
    def transform(self, x, y):
        # center and scale the grid for radius calculation (distance from center of image)
        x_c, y_c = self.w / 2, self.h / 2 
        x = (x - x_c) / x_c
        y = (y - y_c) / y_c
        radius = np.sqrt(x**2 + y**2) # distance from the center of image
        m_r = 1 + self.k_1*radius + self.k_2*radius**2 # radial distortion model
        # apply the model 
        x, y = x * m_r, y * m_r
        # reset all the shifting
        x, y = x*x_c + x_c, y*y_c + y_c
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = 20
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]

def random_augment_frame(frame_data: np.ndarray):
    # Change brightness
    def increase_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img
    
    def noisy(image):
        row,col,ch= image.shape
        mean = 5
        var = 5
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    
    new_image = frame_data.copy()
    if np.random.rand() < 0.5:
        new_image = noisy(new_image)
        new_image = new_image.astype(np.uint8)
    
    new_image = increase_brightness(new_image, value=np.random.randint(10, 50))
      
    return new_image

def insert_lens_flare(frame_data: np.ndarray):
    path_to_flares = 'data/noisy_overlays/flares'
    flares = os.listdir(path_to_flares)
    flare_file = np.random.choice(flares)
    flare = cv2.imread(os.path.join(path_to_flares, flare_file))
    flare = cv2.cvtColor(flare, cv2.COLOR_BGR2RGB)
    
    new_image = frame_data.copy()
    min_x = flare.shape[1]//2
    min_y = flare.shape[0]//2
    x = np.random.randint(min_x, max(min_x, new_image.shape[1] - min_x+1))
    y = np.random.randint(min_y, max(min_y, new_image.shape[0] - min_y+1))
    # new_image = overlay_image_addition(new_image, flare, x, y)
    new_image = overlay_image_addition_with_edge_padding(new_image, flare, x, y)
    
    return new_image
    
def insert_dust(frame_data: np.ndarray):
    path_to_dust = 'data/noisy_overlays/dust'
    dust = os.listdir(path_to_dust)
    dust_file = np.random.choice(dust)
    dust = cv2.imread(os.path.join(path_to_dust, dust_file), cv2.IMREAD_UNCHANGED)
    new_size = (frame_data.shape[1], frame_data.shape[0])
    dust = cv2.resize(dust, new_size)

    new_image = frame_data.copy()
    
    overlay_image_alpha(new_image, dust, 0, 0)
    
    return new_image
    
def insert_scratches(frame_data: np.ndarray):
    # Code adapted from: https://stackoverflow.com/a/76760722
    def bezier(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Generator[np.ndarray, None, None]:
        def calc(t):
            return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3

        # get the approximate pixel count of the curve
        approx = cv2.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
        for t in np.linspace(0, 1, round(approx * 1.2)):
            yield np.round(calc(t)).astype(np.int32)


    def generate_scratch(img: np.ndarray, max_length: float, end_brush_range: tuple[float, float], mid_brush_range: tuple[float, float]) -> np.ndarray:
        H, W, C = img.shape
        # generate the 2 end points of the bezier curve
        x, y, rho1, theta1 = np.random.uniform([0] * 4, [W, H, max_length, np.pi * 2])
        p1 = np.array([x, y, 0])
        p3 = p1 + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]

        # generate the second point, make sure that it cannot be too far away from the middle point of the 2 end points
        rho2, theta2 = np.random.uniform([0], [rho1 / 2, np.pi * 2])
        p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]

        # generate the brush sizes of the 3 points
        p1[2], p2[2], p3[2] = np.random.uniform(*np.transpose([end_brush_range, mid_brush_range, end_brush_range]))

        for x, y, brush in bezier(p1, p2, p3):
            cv2.circle(img, (x, y), brush, (255, 255, 255), -1)
        return img
    
    MAX_LENGTH = 150 # maximum length of the scratch
    END_BRUSH_RANGE = (0, 1) # range of the brush size at the end points
    MID_BRUSH_RANGE = (2, 5) # range of the brush size at the mid point
    SCRATCH_CNT = 60
    
    for _ in range(SCRATCH_CNT):
        frame_data = generate_scratch(frame_data, MAX_LENGTH, END_BRUSH_RANGE, MID_BRUSH_RANGE)
    
    return frame_data
    
def insert_people(frame_data: np.ndarray):
    people_path = 'data/noisy_overlays/persons'
    people = os.listdir(people_path)
    person_file = np.random.choice(people)
    person = cv2.imread(os.path.join(people_path, person_file), cv2.IMREAD_UNCHANGED)
    person = cv2.resize(person, (person.shape[1]*8, person.shape[0]*8))
    
    new_image = frame_data.copy()
    min_x = person.shape[1]//2
    min_y = person.shape[0]//2
    
    middle_x = new_image.shape[1]//2
    person_middle_x = person.shape[1]//2
    x = np.random.randint(middle_x - 250 - person_middle_x, middle_x + 250 - person_middle_x)
    
    # x = np.random.randint(min_x, max(min_x, new_image.shape[1] - min_x+1))
    y = np.random.randint(min_y, max(min_y, new_image.shape[0] - min_y+1))
    overlay_image_alpha(new_image, person, x, y)
       
    return new_image
    
def partial_gaussian_blur(frame_data: np.ndarray):
    blurred = frame_data.copy()
    kernel_size = 15
    sigma = 5
    blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), sigma)
    
    
    blur_count = np.random.randint(1, 5)
    new_image = frame_data.copy()
    mask = np.zeros_like(blurred)
    for _ in range(blur_count):
        x = np.random.randint(0, blurred.shape[1])
        y = np.random.randint(0, blurred.shape[0])
        size = np.random.randint(50, 500)
        mask = cv2.circle(mask, (x, y), size, (255, 255, 255), -1)
        
    new_image = np.where(mask!=(255, 255, 255), new_image, blurred)
    
    return new_image

def translate_image(frame_data: np.ndarray, bboxes: list = None):
    height, width = frame_data.shape[:2]
    max_x_shift = 500
    max_y_shift = 50
    shift_x = np.random.randint(-max_x_shift, max_x_shift)
    shift_y = np.random.randint(-max_y_shift, max_y_shift)
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    if bboxes is not None:
        for bbox in bboxes:
            x_coords = bbox[0::2]
            y_coords = bbox[1::2]
            x_coords = [x + shift_x for x in x_coords]
            y_coords = [y + shift_y for y in y_coords]
            bbox[0::2] = x_coords
            bbox[1::2] = y_coords
            # Limit the bbox coordinates to the image size
            bbox[0::2] = [max(0, min(width, x)) for x in bbox[0::2]]
            bbox[1::2] = [max(0, min(height, y)) for y in bbox[1::2]]
    
    frame_data = cv2.warpAffine(frame_data, translation_matrix, (width, height))
    
    # print(frame_data.shape)
    
    if bboxes is not None:
        return frame_data, bboxes
    
    return frame_data

def rotate_image(frame_data: np.ndarray, bboxes: list = None):
    angle = np.random.randint(-25, 25)
    height, width = frame_data.shape[:2]
    
    # rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    # frame_data = cv2.warpAffine(frame_data, rotation_matrix, (width, height))
    
    # if bboxes is not None:
    #     bbox_chute = bboxes[0]
    #     bbox_straw = bboxes[1]
    #     bbox_chute = rotate_bbox(bbox_chute, width, height, angle)
    #     bbox_straw = rotate_bbox(bbox_straw, width, height, angle)
    #     bbox_chute = clip_bbox_to_image(bbox_chute, width, height)
    #     bbox_straw = clip_bbox_to_image(bbox_straw, width, height)
    #     bboxes = [bbox_chute, bbox_straw]
            
    
    if bboxes is not None:
        bbox_chute = bboxes[0]
        bbox_straw = bboxes[1]
        frame_data_copy = frame_data.copy()
        frame_data, _, bbox_chute = rotate_image_and_bbox(frame_data, None, bbox_chute, angle)
        _, _, bbox_straw = rotate_image_and_bbox(frame_data_copy, None, bbox_straw, angle)
        bboxes = [bbox_chute, bbox_straw]
    else:
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        frame_data = cv2.warpAffine(frame_data, rotation_matrix, (width, height))
    
    # frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
    # frame_data = cv2.polylines(frame_data, [np.array(bboxes[0]).reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)
    # cv2.imshow('frame', frame_data)
    # cv2.waitKey(0)
    
    # print(frame_data.shape)
    
    if bboxes is not None:
        return frame_data, bboxes
    
    return frame_data

def distort_image(frame_data: np.ndarray, bboxes: list = None):
    h, w = frame_data.shape[:2]
    barrel_deformer = BarrelDeformer(w, h)
    im_deformed = frame_data.copy()
    
    # Create channel for bbox where the corners of the bbox are set to 255, the rest is 0
    if bboxes is not None:
        bbox_chute = bboxes[0]
        bbox_straw = bboxes[1]
        bbox_chute = np.array(bbox_chute)
        bbox_straw = np.array(bbox_straw)
        
        bbox_chute = bbox_chute.reshape(-1, 1, 2).astype(np.int32)
        bbox_straw = bbox_straw.reshape(-1, 1, 2).astype(np.int32)
        
        bbox_chute_channel = np.zeros((h, w), dtype=np.uint8)
        bbox_straw_channel = np.zeros((h, w), dtype=np.uint8)
        
        bbox_chute_channel[bbox_chute[:, 0, 1], bbox_chute[:, 0, 0]] = 255
        bbox_straw_channel[bbox_straw[:, 0, 1], bbox_straw[:, 0, 0]] = 255
        
        bbox_image = np.zeros((h, w, 3), dtype=np.uint8)
        bbox_image[..., 0] = bbox_chute_channel
        bbox_image[..., 1] = bbox_straw_channel
        bbox_image[..., 2] = 0
        
    im_deformed = Image.fromarray(im_deformed)
    im_deformed.putalpha(255)
    im_deformed = ImageOps.deform(im_deformed, barrel_deformer)
    im_deformed = np.array(im_deformed)
    
    # Extract the positions of the bboxes from the deformed image
    if bboxes is not None:
        bbox_image = Image.fromarray(bbox_image)
        bbox_image.putalpha(255)
        bbox_image = ImageOps.deform(bbox_image, barrel_deformer)
        bbox_image = np.array(bbox_image)
        # bbox_image = im_deformed[:, :, 2:]
        bbox_chute = np.argwhere(bbox_image[:, :, 0] != 0)
        bbox_straw = np.argwhere(bbox_image[:, :, 1] != 0)
        bboxes = [bbox_chute, bbox_straw]

    # if bboxes is not None:
    #     bboxes = [barrel_deformer.transform_bbox(bbox) for bbox in bboxes]
    
    if bboxes is not None:
        return im_deformed, bboxes
    
    return im_deformed

def distort_image_wand(frame_data: np.ndarray, bboxes: list = None):
    frame_data = frame_data.copy()
    if bboxes is not None:
        bbox_chute = bboxes[0]
        bbox_straw = bboxes[1]
        bbox_chute_channel = np.zeros((frame_data.shape[0], frame_data.shape[1]), dtype=np.uint8)
        bbox_straw_channel = np.zeros((frame_data.shape[0], frame_data.shape[1]), dtype=np.uint8)
        bbox_chute = bbox_chute.reshape(-1, 1, 2).astype(np.int32)
        bbox_straw = bbox_straw.reshape(-1, 1, 2).astype(np.int32)
        
        bbox_chute_channel[bbox_chute[:, 0, 1], bbox_chute[:, 0, 0]] = 255
        bbox_straw_channel[bbox_straw[:, 0, 1], bbox_straw[:, 0, 0]] = 255
        
        bbox_image = np.zeros((frame_data.shape[0], frame_data.shape[1], 3), dtype=np.uint8)
        bbox_image[..., 0] = bbox_chute_channel
        bbox_image[..., 1] = bbox_straw_channel
        bbox_image[..., 2] = 0
        bbox_image = Image.from_array(bbox_image)
        bbox_image.virtual_pixel = 'transparent'
        bbox_image.distort('barrel', (0.2, 0.0, 0.0, 1.0))
        bbox_image = np.array(bbox_image)
        # frame_data = np.concatenate([frame_data, np.zeros((frame_data.shape[0], frame_data.shape[1], 2), dtype=np.uint8)], axis=-1)
        # frame_data[:, :, 3] = bbox_chute_channel
        # frame_data[:, :, 4] = bbox_straw_channel
    
    img = Image.from_array(frame_data)
    img.virtual_pixel = 'transparent'
    img.distort('barrel', (0.2, 0.0, 0.0, 1.0))
    img = np.array(img)
    
    if bboxes is not None:
        bbox_chute = np.argwhere(bbox_image[:, :, 0] != 0)
        bbox_straw = np.argwhere(bbox_image[:, :, 1] != 0)
        bboxes = [bbox_chute, bbox_straw]
        
        return img, bboxes
    
    return img

def create_noisy_dataset(hdf5: h5py, save_path: str):
    # Get a list of all the keys in the hdf5 file
    frame_names = list(hdf5.keys())
    # Sort frame_names
    frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1]))
    
    # Create a new hdf5 file to store the noisy data
    noisy_hf = h5py.File(save_path, 'w')
    
    pbar = tqdm(total=len(frame_names), desc='Creating noisy dataset')
    # Loop through all the keys in the hdf5 file
    for current_frame in tqdm(frame_names):
        # Get the frame
        frame = hdf5[current_frame]
        frame_data = decode_binary_image(frame['image'][...])
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        
        bbox_chute = frame['annotations']['bbox_chute'][...]
        bbox_straw = frame['annotations']['bbox_straw'][...]
        
        bboxes = [bbox_chute, bbox_straw]
        
        num_changes = np.random.randint(1, 7)

        changes = np.random.choice(['lens_flare', 'random_augmet', 'dust', 'scratches', 'people', 'partial_gaussian_blur', 'distort'], 
                                   num_changes, replace=False)
        
        pbar.set_description(f'Creating noisy dataset, num_changes: {num_changes}, changes: {changes}')
        if 'lens_flare' in changes:
            frame_data = insert_lens_flare(frame_data)
        if 'dust' in changes:
            frame_data = insert_dust(frame_data)
        if 'scratches' in changes:
            frame_data = insert_scratches(frame_data)
        if 'people' in changes:
            frame_data = insert_people(frame_data)
        if 'partial_gaussian_blur' in changes:
            frame_data = partial_gaussian_blur(frame_data)
        if 'random_augment' in changes:
            frame_data = random_augment_frame(frame_data)
        if 'distort' in changes:
            frame_data, bboxes = translate_image(frame_data, bboxes)
            frame_data, bboxes = rotate_image(frame_data, bboxes)
            frame_data, bboxes = distort_image(frame_data, bboxes)
 
        
        # Display the frame
        frame_data = cv2.resize(frame_data, (frame_data.shape[1]//2, frame_data.shape[0]//2))
        # Resize bboxes for display
        bboxes = [[x//2 for x in bbox] for bbox in bboxes]
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
        frame_data = cv2.polylines(frame_data, [np.array(bboxes[0]).reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)
        frame_data = cv2.polylines(frame_data, [np.array(bboxes[1]).reshape(-1, 1, 2).astype(np.int32)], True, (255, 255, 0), 2)
        cv2.imshow('frame', frame_data)
        cv2.waitKey(0)
        
        # Save the frame to the new hdf5 file
        # Image and image diff
        frame_group = noisy_hf.create_group(current_frame)
        frame_group.create_dataset('image', data=frame_data)
        frame_group.create_dataset('image_diff', data=hdf5[current_frame]['image_diff'][...])
        
        # Annotations
        frame_group.create_group('annotations')
        anno_group = frame_group['annotations']
        anno_group.create_dataset('bbox_chute', data=hdf5[current_frame]['annotations']['bbox_chute'][...])
        anno_group.create_dataset('bbox_straw', data=hdf5[current_frame]['annotations']['bbox_straw'][...])
        anno_group.create_dataset('fullness', data=hdf5[current_frame]['annotations']['fullness'][...])
        anno_group.create_dataset('obstructed', data=hdf5[current_frame]['annotations']['obstructed'][...])
        anno_group.create_dataset('sensor_fullness', data=hdf5[current_frame]['annotations']['sensor_fullness'][...])
        
        pbar.update(1)
        
    pbar.close()
    
    

if __name__ == '__main__':
    # Load the hdf5 file
    path_to_hdf5 = 'data/processed/sensors.hdf5'
    save_path = 'data/processed/noisy_sensors.hdf5'
    hf = h5py.File(path_to_hdf5, 'r')
    
    create_noisy_dataset(hf, save_path)
    









