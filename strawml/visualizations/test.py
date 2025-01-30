from __init__ import *
import h5py
import numpy as np
from strawml.models.chute_finder.yolo import ObjectDetect
from yolov11_cam.yolo_cam.eigen_cam import EigenCAM
from yolov11_cam.yolo_cam.utils.image import show_cam_on_image, scale_cam_image
from strawml.data.make_dataset import decode_binary_image
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Load the data
data_rotated = "data/predictions/recording_rotated_all_frames_processed_combined.hdf5"
data_vertical = "data/predictions/recording_vertical_all_frames_processed_combined.hdf5"
# first we extract the keys from rotated and vertical datasets
with h5py.File(data_rotated, 'r') as f:
    rotated_keys = list(f.keys())
with h5py.File(data_vertical, 'r') as f:
    vertical_keys = list(f.keys())

# list all files in the noisy_datasets folder
import os

# data_files = os.listdir("data/noisy_datasets")

data_file = f"data/predictions/recording_combined_all_frames_processed.hdf5"
# then we run through the datafile and check which keys are in the rotated and vertical datasets and then add a tag "type" : "rotated" or "vertical"
with h5py.File(data_file, 'r+') as f:
    pbar = tqdm(f.keys())
    pbar.set_description(f"Processing {data_file}")
    for key in pbar:
        if key in rotated_keys:
            f[key].attrs['type'] = 'rotated'
        elif key in vertical_keys:
            f[key].attrs['type'] = 'vertical'
        else:
            print(f"Key {key} not found in rotated or vertical datasets")



# for file in data_files:
#     data_file = f"data/noisy_datasets/{file}"
#     # then we run through the datafile and check which keys are in the rotated and vertical datasets and then add a tag "type" : "rotated" or "vertical"
#     with h5py.File(data_file, 'r+') as f:
#         pbar = tqdm(f.keys())
#         pbar.set_description(f"Processing {data_file}")
#         for key in pbar:
#             if key in rotated_keys:
#                 f[key].attrs['type'] = 'rotated'
#             elif key in vertical_keys:
#                 f[key].attrs['type'] = 'vertical'
#             else:
#                 print(f"Key {key} not found in rotated or vertical datasets")

# # Load OD model
# model = YOLO("models/obb_best.pt").to("cuda")
# target_layers =[model.model.model[-4], model.model.model[-2]]

# # Load the CAM model
# cam = EigenCAM(model, target_layers,task='od')

# frame_names = list(data.keys())
# # sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[-1]))
# for frame in frame_names:
#     img = decode_binary_image(data[frame]['image'][...])
#     # img = cv2.resize(img, (512, 512))
#     rgb_img = img.copy()
#     img = np.float32(img) / 255

#     grayscale_cam = cam(rgb_img)[0, :, :]
#     cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
#     plt.imshow(cam_image)
#     plt.show()
