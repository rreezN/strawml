from __init__ import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader
import h5py
import cv2
import sklearn.manifold as skm

from strawml.data.make_dataset import decode_binary_image
from strawml.data.dataloader import Chute
from strawml.models.straw_classifier.chute_cropper import rotate_and_crop_to_bbox


# TODO: Big list in chatgpt: https://chatgpt.com/share/66f6ca92-e2fc-8003-a85f-cb4b4c023eeb


def get_frames_by_class(frames: h5py.File) -> dict:
    """Get a dictionary of frames by class.
    """
    print('Sorting frames by class...')
    frames_by_class = {"augmented": {}, "original": {}}
    frame_names = list(frames.keys())
    
    print("Removing all cropped, rotated or translated images from the dataset (data_purpose='straw level monitoring')")
    for frame_name in list(frames.keys()):
        attributes = frames[frame_name].attrs
        if 'augmented' in attributes:
            banned_augmentations = ['cropping', 'translation', 'rotation']
            augmentations = frames[frame_name].attrs['augmented']
            if any(x in augmentations for x in banned_augmentations):
                frame_names.remove(frame_name)
            else:
                current_class = float(frames[frame_name]['annotations']['fullness'][...])
                frames_by_class['augmented'][current_class] = frames_by_class['augmented'].get(current_class, []) + [frame_name]
        else:
            current_class = float(frames[frame_name]['annotations']['fullness'][...])
            frames_by_class['original'][current_class] = frames_by_class['original'].get(current_class, []) + [frame_name]
        
    
    
    # print(frames[frame_names[0]]['annotations']['fullness'][...])
    # for frame_name in frame_names:
    #     current_class = float(frames[frame_name]['annotations']['fullness'][...])
    #     frames_by_class[current_class] = frames_by_class.get(current_class, []) + [frame_name]
    
    frames_by_class['augmented'] = dict(sorted(frames_by_class['augmented'].items()))
    frames_by_class['original'] = dict(sorted(frames_by_class['original'].items()))
    
    # for key in frames_by_class.keys():
    #     print(f'Class {key}: {len(frames_by_class[key])} frames')
    
    return frames_by_class
    

def plot_class_distribution(class_dict: dict, frames: h5py.File) -> None:
    """Plot the class distribution of the dataset.
    """
    print("Plotting class distribution...")
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))
    
    counts_original = [len(class_dict['original'].get(float(key/100), [])) for key in classes]
    counts_augmented = [len(class_dict['augmented'].get(float(key/100), [])) for key in classes]
    
    counts = {
        'original': counts_original, 
        'augmented': counts_augmented,
        }
    
    total_frames = sum(counts_augmented + counts_original)
    
    # Create a 2x11 figure with bar chart in top left corner, and examples from each class in the remaining slots
    fig = plt.figure(figsize=(14,7))
    gs = gridspec.GridSpec(2, 13, figure=fig, wspace=0.3, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    bottom = np.zeros(len(classes))
    left = np.zeros(len(classes))
    for data_type, count in counts.items():
        ax1.barh(classes, count, align='center', label=data_type, height=4, left=left)
        left += count
    
    ax1.legend()
    ax1.set_yticks(np.arange(0, max(classes*100) + 1, 10))
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Class')
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(np.array(counts['original']) + np.array(counts['augmented'])) + 10)

    first_image = load_image(class_dict, classes, 0)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(first_image)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f'Class {classes[0]}')

    
    ax3 = fig.add_subplot(gs[1, 0:3])
    ax3.axis('off')
    ax3.text(0.5, 0.5, f'{total_frames} total\nframes', fontsize=16, fontweight='bold', ha='center', va='center')
    
    
    class_counter = 1
    for row in range(2):
        for col in range(3, 13):
            ax = fig.add_subplot(gs[row, col])
            image = load_image(class_dict, classes, class_counter)
            image = cv2.resize(image, (204, 1370))
            ax.imshow(image)
            ax.set_title(f'Class {classes[class_counter]}')
            ax.set_xticks([])
            ax.set_yticks([])
            class_counter += 1
    
    
    
    
    plt.suptitle(f'Class Distribution and Examples ({frames.filename})')
    plt.show()


def load_image(class_dict, classes, class_number) -> np.ndarray:
    augmented = np.random.randint(0, 2) == 1
    if augmented and class_dict['augmented'].get(float(classes[class_number]/100), None) is not None:
        image_index = np.random.randint(0, len(class_dict['augmented'][float(classes[class_number]/100)]))
        first_image = decode_binary_image(frames[class_dict['augmented'][float(classes[class_number]/100)][image_index]]['image'][...]) 
        first_image, bbox = rotate_and_crop_to_bbox(first_image, frames[class_dict['augmented'][float(classes[class_number]/100)][image_index]]['annotations']['bbox_chute'][...])
    else:
        image_index = np.random.randint(0, len(class_dict['original'][float(classes[class_number]/100)]))
        first_image = decode_binary_image(frames[class_dict['original'][float(classes[class_number]/100)][image_index]]['image'][...])
        first_image, bbox = rotate_and_crop_to_bbox(first_image, frames[class_dict['original'][float(classes[class_number]/100)][image_index]]['annotations']['bbox_chute'][...])
    return cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    frames = h5py.File('data/processed/augmented/chute_detection.hdf5', 'r')
    class_dictionary = get_frames_by_class(frames)
    plot_class_distribution(class_dictionary, frames)
    
    
    







