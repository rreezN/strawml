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
    frames_by_class = {}
    frame_names = list(frames.keys())
    
    print("Removing all cropped, rotated or translated images from the dataset (data_purpose='straw level monitoring')")
    for frame_name in list(frames.keys()):
        attributes = frames[frame_name].attrs
        if 'augmented' in attributes:
            banned_augmentations = ['cropping', 'translation', 'rotation']
            augmentations = frames[frame_name].attrs['augmented']
            if any(x in augmentations for x in banned_augmentations):
                frame_names.remove(frame_name)
    
    
    # print(frames[frame_names[0]]['annotations']['fullness'][...])
    for frame_name in frame_names:
        current_class = float(frames[frame_name]['annotations']['fullness'][...])
        frames_by_class[current_class] = frames_by_class.get(current_class, []) + [frame_name]
    
    frames_by_class = dict(sorted(frames_by_class.items()))
    
    # for key in frames_by_class.keys():
    #     print(f'Class {key}: {len(frames_by_class[key])} frames')
    
    return frames_by_class
    

def plot_class_distribution(class_dict: dict, frames: h5py.File) -> None:
    """Plot the class distribution of the dataset.
    """
    print("Plotting class distribution...")
    classes = list([int(x*100) for x in class_dict.keys()])
    counts = [len(class_dict[key]) for key in class_dict.keys()]
    
    total_frames = sum(counts)
    
    # Create a 2x11 figure with bar chart in top left corner, and examples from each class in the remaining slots
    fig = plt.figure(figsize=(14,7))
    gs = gridspec.GridSpec(2, 13, figure=fig, wspace=0.3, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.barh(classes, counts, align='center', height=4)
    ax1.set_yticks(np.arange(0, max(classes*100) + 1, 10))
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Class')
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(counts) + 10)

    image_index = np.random.randint(0, len(class_dict[float(classes[0]/100)]))
    first_image = decode_binary_image(frames[class_dict[float(classes[0]/100)][image_index]]['image'][...])
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    first_image, bbox = rotate_and_crop_to_bbox(first_image, frames[class_dict[float(classes[0]/100)][image_index]]['annotations']['bbox_chute'][...])
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
            image_index = np.random.randint(0, len(class_dict[float(classes[class_counter]/100)]))
            ax = fig.add_subplot(gs[row, col])
            image = decode_binary_image(frames[class_dict[float(classes[class_counter]/100)][image_index]]['image'][...])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image, bbox = rotate_and_crop_to_bbox(image, frames[class_dict[float(classes[class_counter]/100)][image_index]]['annotations']['bbox_chute'][...])
            image = cv2.resize(image, (204, 1370))
            ax.imshow(image)
            ax.set_title(f'Class {classes[class_counter]}')
            ax.set_xticks([])
            ax.set_yticks([])
            class_counter += 1
    
    
    
    
    plt.suptitle(f'Class Distribution, Examples and t-SNE ({frames.filename})')
    plt.show()


def calculate_tsne(features: np.ndarray) -> np.ndarray:
    """Calculate the t-SNE representation of the images.
    """
    print('Calculating t-SNE representation...')
    tsne = skm.TSNE(n_components=2, random_state=42)
    tsne_representation = tsne.fit_transform(features)
    
    return tsne_representation

def normalize_tsne(x: np.ndarray) -> np.ndarray:
    """Normalize the array.
    """
    value_range = np.max(x) - np.min(x)
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range
    

if __name__ == '__main__':
    frames = h5py.File('data/processed/augmented/chute_detection.hdf5', 'r')
    class_dictionary = get_frames_by_class(frames)
    plot_class_distribution(class_dictionary, frames)
    
    
    







