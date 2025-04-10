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



# data_file = f"data/interim/sensors_with_strawbbox.hdf5"
# # then we run through the datafile and check which keys are in the rotated and vertical datasets and then add a tag "type" : "rotated" or "vertical"
# with h5py.File(data_file, 'r+') as f:
#     pbar = tqdm(f.keys())
#     pbar.set_description(f"Processing {data_file}")
#     new_size = (640, 640)
#     # rewrite the image data to the dataset
#     for key in pbar:
#         img = decode_binary_image(f[key]['image'][...])
#         og_h, og_w = img.shape[:2]
#         # resize to 640x640
#         img = cv2.resize(img, new_size)
#         # encode the image to binary
#         pbar.set_description(f"Processing {key}, {img.shape}")
#         img = cv2.imencode('.jpg', img)[1]
#         if 'bbox_straw' not in f[key]['annotations']:
#             del f[key]['image']
#             f[key].create_dataset('image', data=img)
#             continue
#         bbox = f[key]['annotations']['bbox_straw'][...]
#         # resize the bbox to the new size
#         h_scale = new_size[0] / og_h
#         w_scale = new_size[1] / og_w

#         x1, y1, x2, y2, x3, y3, x4, y4 = bbox
#         # resize the coordinates to the new image size
#         h_scale = new_size[1] / og_h
#         w_scale = new_size[0] / og_w
#         x1, x2, x3, x4 = [x * w_scale for x in [x1, x2, x3, x4]]
#         y1, y2, y3, y4 = [y * h_scale for y in [y1, y2, y3, y4]]
#         bbox = np.array([x1, y1, x2, y2, x3, y3, x4, y4])

#         # drop the old image and bbox data
#         del f[key]['image']
#         del f[key]['annotations']['bbox_straw']
#         # create new datasets
#         f[key].create_dataset('image', data=img)
#         f[key]['annotations'].create_dataset('bbox_straw', data=bbox)

# # Load the data
# data_rotated = "data/predictions/recording_rotated_all_frames_processed_combined.hdf5"
# data_vertical = "data/predictions/recording_vertical_all_frames_processed_combined.hdf5"
# # first we extract the keys from rotated and vertical datasets
# with h5py.File(data_rotated, 'r') as f:
#     rotated_keys = list(f.keys())
# with h5py.File(data_vertical, 'r') as f:
#     vertical_keys = list(f.keys())

# # list all files in the noisy_datasets folder
# import os

# # data_files = os.listdir("data/noisy_datasets")

# data_file = f"data/predictions/recording_combined_all_frames_processed.hdf5"
# # then we run through the datafile and check which keys are in the rotated and vertical datasets and then add a tag "type" : "rotated" or "vertical"
# with h5py.File(data_file, 'r+') as f:
#     pbar = tqdm(f.keys())
#     pbar.set_description(f"Processing {data_file}")
#     for key in pbar:
#         if key in rotated_keys:
#             f[key].attrs['type'] = 'rotated'
#         elif key in vertical_keys:
#             f[key].attrs['type'] = 'vertical'
#         else:
#             print(f"Key {key} not found in rotated or vertical datasets")

# typoo = 'rotated'
# large_data_file = f"data/predictions/new_run/recording_{typoo}_all_frames_processed.hdf5"
# main_file = f"data/predictions/new_run/recording_{typoo}_all_frames_processed_combined.hdf5"

# with h5py.File(large_data_file, 'r') as f:
#     with h5py.File(main_file, 'r+') as main_f:

#         pbar = tqdm(main_f.keys())
#         pbar.set_description(f"Processing {large_data_file}")
#         for key in pbar:
#             # Go through and check if yolo, yolo_smooth, convnext and convnext_smooth are in the main file
#             if key in f.keys():
#                 if 'yolo' in main_f[key].keys():
#                     del main_f[key]['yolo']
#                 if 'yolo_smooth' in main_f[key].keys():
#                     del main_f[key]['yolo_smooth']
#                 if 'convnext' in main_f[key].keys():
#                     del main_f[key]['convnext']
#                 if 'convnext_smooth' in main_f[key].keys():
#                     del main_f[key]['convnext_smooth']
#                 main_f.copy(f[key]['yolo'], main_f[key], 'yolo')
#                 main_f.copy(f[key]['yolo_smooth'], main_f[key], 'yolo_smooth')
#                 main_f.copy(f[key]['convnext'], main_f[key], 'convnext')
#                 main_f.copy(f[key]['convnext_smooth'], main_f[key], 'convnext_smooth')
#             else:
#                 print(f"Key {key} not found in the dataset")

                    




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

#     grayscale_cam  cam(rgb_img)[0, :, :]
#     cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
#     plt.imshow(cam_image)
#     plt.show()


import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor
from tqdm import tqdm
import cv2 
# Load data
data_path = 'data/predictions/new_run/recording_vertical_all_frames_processed_combined.hdf5'
hf = h5py.File(data_path, 'r')

timestamps = list(hf.keys())
if "frame" == timestamps[0].split("_")[0]:
    timestamps = sorted(timestamps, key=lambda x: int(x.split('_')[1]))
else:
    timestamps = sorted(timestamps, key=lambda x: float(x))

# Extract values for plotting
values = [hf[t]['yolo_clipped_2.5']['percent'][()] for t in timestamps]

# Initialize figure and axis
fig, (ax_img, ax_plot) = plt.subplots(2, 1, figsize=(10, 10))
# Plot the predictions
time_indices = np.arange(len(values))
ax_plot.plot(time_indices, values, marker='o', linestyle='-', label='Predictions')
current_point, = ax_plot.plot([], [], 'ro', markersize=10)  # Indicator for the current frame
ax_plot.set_xlabel("Frame Index")
ax_plot.set_ylabel("Prediction Value")
ax_plot.legend()

# Display first image
img_display = ax_img.imshow(np.zeros((1440, 2560)), cmap='gray')  # Placeholder image
ax_img.axis("off")

# State tracking
index = 0

def update_display():
    """ Update the image and plot marker based on the current index. """
    global index
    key = timestamps[index]
    image_data = hf[key]['image'][...]
    print(key)
    image_array = np.frombuffer(image_data, np.uint8)
    # Decode the image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img_display.set_data(image)
    
    # Update plot indicator
    current_point.set_data(index, values[index])
    
    fig.canvas.draw_idle()

# Key event handler
def on_key(event):
    global index
    if event.key == "right" and index < len(timestamps) - 1:
        index += 1
    elif event.key == "left" and index > 0:
        index -= 1
    update_display()

# Connect event
global cid
cid = fig.canvas.mpl_connect("key_press_event", on_key)

# Initial display
update_display()
plt.show()

# Close file after execution
hf.close()
