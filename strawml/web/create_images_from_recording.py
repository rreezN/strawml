from __init__ import *

import h5py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from strawml.data.make_dataset import decode_binary_image

def create_images_from_recording(data_path: str, output_folder: str, show_images: bool = False) -> None:
    """Creates images from the images HDF5 file and saves them to the output folder.
    
    Args:
        data_path (str): The path to the images HDF5 file.
        output_folder (str): The path to the output folder.
    
    Raises:
        FileNotFoundError: If the images HDF5 file does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {data_path} does not exist.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with h5py.File(data_path, 'r') as f:
        keys = list(f.keys())
        tqdm_keys = tqdm(keys, desc='Creating images')
        for i in range(len(keys)):
            tqdm_keys.set_postfix({'image': i})
            tqdm_keys.update()
            image = f[keys[i]]['image'][:]
            image = decode_binary_image(image)
            
            if 'scada' in f[keys[i]].keys():
                scada_group = f[keys[i]]['scada']
                scada_percent = scada_group['percent'][...].item()
                scada_line = scada_group['pixel'][...]
            else:
                scada_percent = None
                scada_line = None
                
            if 'yolo' in f[keys[i]].keys():
                yolo_group = f[keys[i]]['yolo']
                yolo_percent = yolo_group['percent'][...].item()
                yolo_line = yolo_group['pixel'][...]
            else: # TODO: This needs handling because yolo bboxes are just not recorded if they don't exist
                yolo_percent = 0.0
                yolo_line = None
            
            line_offset = 200
            line_thickness = 10
            font_scale = 2
            # Plot the yolo and scada lines and percent on the image
            if scada_line is not None:
                scada_coords = (int(35+scada_line[0]), int(scada_line[1]))
                image = cv2.line(image, scada_coords, (scada_coords[0]+line_offset, scada_coords[1]), (0, 255, 0), line_thickness)
                image = cv2.putText(image, f'{scada_percent:.2f}%', (scada_coords[0]+line_offset, scada_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), line_thickness//2)
            if yolo_line is not None:
                yolo_coords = (int(35+yolo_line[0]), int(yolo_line[1]))
                image = cv2.line(image, yolo_coords, (yolo_coords[0]+line_offset, yolo_coords[1]), (0, 0, 255), line_thickness)
                image = cv2.putText(image, f'{yolo_percent:.2f}%', (yolo_coords[0]+line_offset, yolo_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness//2)
            else:
                if scada_line is not None:
                    yolo_coords = scada_coords
                    yolo_coords = (yolo_coords[0], image.shape[0]-line_thickness-64)
                else: # If there are no scada or yolo lines, just put the text in the bottom middle
                    yolo_coords = (int(image.shape[1]//2), image.shape[0]-line_thickness-64)
                image = cv2.line(image, yolo_coords, (yolo_coords[0]+line_offset, yolo_coords[1]), (0, 0, 255), line_thickness)
                image = cv2.putText(image, f'0.0%',  (yolo_coords[0]+line_offset, yolo_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness//2)
            
            # Resize the image
            image_width = image.shape[1]
            image_height = image.shape[0]
            downscale_factor = 2
            image = cv2.resize(image, (int(image_width / downscale_factor), int(image_height / downscale_factor)))
            if show_images:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image)
                plt.show()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_path = os.path.join(output_folder, f'{keys[i]}.jpg')
            cv2.imwrite(image_path, image)


if __name__ == '__main__':
    create_images_from_recording('data/processed/recording.hdf5', 'data/processed/recordings', False)