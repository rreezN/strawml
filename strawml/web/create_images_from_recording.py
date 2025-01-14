from __init__ import *

import h5py
import os
import cv2
import piexif
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

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
    
    total_size = 0
    total_size_mb = 0
    total_size_gb = 0
    
    
    with h5py.File(data_path, 'r') as f:
        keys = list(f.keys())
        tqdm_keys = tqdm(keys, desc='Creating images')
        for i in range(len(keys)):
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
                yolo_percent = None
                yolo_line = None
            
            if 'convnextv2' in f[keys[i]].keys():
                convnextv2_group = f[keys[i]]['convnextv2']
                convnextv2_percent = convnextv2_group['percent'][...].item()
                convnextv2_line = convnextv2_group['pixel'][...]
            else:
                convnextv2_percent = None
                convnextv2_line = None
            
            line_thickness = 5
            font_scale = 2
            # Plot the yolo and scada lines and percent on the image
            if scada_line is not None:
                scada_coords_left = (int(scada_line[0][0]), int(scada_line[0][1]))
                scada_coords_right = (int(scada_line[1][0]), int(scada_line[1][1]))
                image = cv2.line(image, scada_coords_left, scada_coords_right, (0, 255, 0), line_thickness)
                # image = cv2.putText(image, f'{scada_percent:.2f}%', scada_coords_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), line_thickness)
                text_coords = (scada_coords_left[0] - 50, scada_coords_left[1])
                image = cv2.putText(image, 'A', scada_coords_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), line_thickness)
            if yolo_line is not None:
                yolo_coords_left = (int(yolo_line[0][0]), int(yolo_line[0][1]))
                yolo_coords_right = (int(yolo_line[1][0]), int(yolo_line[1][1]))
                image = cv2.line(image, yolo_coords_left, yolo_coords_right, (0, 0, 255), line_thickness)
                # image = cv2.putText(image, f'{yolo_percent:.2f}%', yolo_coords_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
                text_coords = (yolo_coords_left[0] - 50, yolo_coords_left[1])
                image = cv2.putText(image, 'B', yolo_coords_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
            if convnextv2_line is not None:
                convnextv2_coords_left = (int(convnextv2_line[0][0]), int(convnextv2_line[0][1]))
                convnextv2_coords_right = (int(convnextv2_line[1][0]), int(convnextv2_line[1][1]))
                image = cv2.line(image, convnextv2_coords_left, convnextv2_coords_right, (255, 0, 0), line_thickness)
                # image = cv2.putText(image, f'{convnextv2_percent:.2f}%', convnextv2_coords_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), line_thickness)
                text_coords = (convnextv2_coords_left[0] - 50, convnextv2_coords_left[1])
                image = cv2.putText(image, 'C', convnextv2_coords_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), line_thickness)
            # else:
                # print(f'No yolo line for {i}:{keys[i]}')
                # if scada_line is not None:
                #     yolo_coords_left = scada_coords_left
                #     yolo_coords_right = scada_coords_right
                #     yolo_coords_left = (yolo_coords_left[0], image.shape[0]-line_thickness-64)
                #     yolo_coords_right = (yolo_coords_right[0], image.shape[0]-line_thickness-64)
                # else: # If there are no scada or yolo lines, just put the text in the bottom middle
                #     yolo_coords_left = (int(image.shape[1]//2)-50, image.shape[0]-line_thickness-64)
                #     yolo_coords_right = (int(image.shape[1]//2)+50, image.shape[0]-line_thickness-64)
                    
                # image = cv2.line(image, yolo_coords_left, yolo_coords_right, (0, 0, 255), line_thickness)
                # image = cv2.putText(image, f'0.0%',  yolo_coords_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness)
            
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
            
            # Save the image
            image_path = os.path.join(output_folder, f'{keys[i]}.jpg')
            cv2.imwrite(image_path, image)
            
            # Add metadata using piexif
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
            scada_string = ''
            if scada_percent is not None:
                scada_string = f"scada_percent={scada_percent:.2f}"
            yolo_string = ''
            if yolo_percent is not None:
                yolo_string = f"yolo_percent={yolo_percent:.2f}"
            if convnextv2_percent is not None:
                convnextv2_string = f"convnextv2_percent={convnextv2_percent:.2f}"
            percent = f"{scada_string}, {yolo_string}, {convnextv2_string}"
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = percent
            exif_bytes = piexif.dump(exif_dict)

            pil_image = Image.open(image_path)
            pil_image.save(image_path, "jpeg", quality=95, exif=exif_bytes)
            pil_image.close()
            
            
            total_size += os.path.getsize(image_path)
            total_size_mb = total_size / 1024 / 1024
            total_size_gb = total_size_mb / 1024
            tqdm_keys.set_postfix({'size (mb)': total_size_mb, 'exif': exif_dict["0th"][piexif.ImageIFD.ImageDescription]}) if total_size_mb < 1024 else tqdm_keys.set_postfix({'size (gb)': total_size_gb, 'exif': exif_dict["0th"][piexif.ImageIFD.ImageDescription]})
        
    tqdm_keys.close()

    print(f"{len(keys)} images saved to {output_folder}! Total size: {total_size_gb:.2f} GB")

if __name__ == '__main__':
    create_images_from_recording('data/processed/recording_vertical_all_frames_processed.hdf5', 'data/processed/recordings', False)