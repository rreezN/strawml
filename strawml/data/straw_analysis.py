from __init__ import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import re

from strawml.data.make_dataset import decode_binary_image
from strawml.models.straw_classifier.chute_cropper import rotate_and_crop_to_bbox


# TODO: Big list in chatgpt: https://chatgpt.com/share/66f6ca92-e2fc-8003-a85f-cb4b4c023eeb


def get_frames_by_class(frames: h5py.File) -> dict:
    """Get a dictionary of frames by class.
    """
    print('Sorting frames by class...')
    frames_by_class = {"augmented": {}, "original": {}}
    frame_names = list(frames.keys())
    
    # banned_augmentations = ['cropping', 'translation', 'rotation', 'color']
    banned_augmentations = ['cropping', 'translation', 'rotation']
    print(f"Removing {banned_augmentations} images from the dataset (data_purpose='straw level monitoring')")
    for frame_name in list(frames.keys()):
        attributes = frames[frame_name].attrs
        if 'augmented' in attributes:
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

def get_all_frames_by_class_order(frames: h5py.File, class_dict: dict) -> list:
    """Get a list of all frames in the dataset.
    """
    classes = np.array(list(class_dict['augmented'].keys()) + list(class_dict['original'].keys()))
    classes = np.unique(classes)
    classes = np.sort(classes)
    all_frames = []
    for class_key in classes:
        all_frames += class_dict['augmented'].get(class_key, []) + class_dict['original'].get(class_key, [])
    return all_frames

def get_idx_of_class(frames: h5py.File, class_dict: dict, class_number: int) -> list:
    """Get the indices of all frames in a class.
    """
    
    classes = np.array(list(class_dict['augmented'].keys()) + list(class_dict['original'].keys()))
    classes = np.unique(classes)
    classes = np.sort(classes)
    all_frames = get_all_frames_by_class_order(frames, class_dict)
    idx = []
    frame_names = list(class_dict['augmented'].get(float(class_number*5/100), [])) + list(class_dict['original'].get(float(class_number*5/100), []))
    for frame_name in frame_names:
        idx.append(all_frames.index(frame_name))

    return idx


def plot_class_distribution(class_dict: dict, frames: h5py.File) -> None:
    """Plot the class distribution of the dataset.
    """
    
    print(" ---- Plotting Class Distribution ---- ")
    
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

    first_image = load_image_by_class(class_dict, classes, 0)
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
            image = load_image_by_class(class_dict, classes, class_counter)
            image = cv2.resize(image, (204, 1370))
            ax.imshow(image)
            ax.set_title(f'Class {classes[class_counter]}')
            ax.set_xticks([])
            ax.set_yticks([])
            class_counter += 1
    
    
    
    
    plt.suptitle(f'Straw Class Distribution and Examples ({frames.filename})')
    plt.savefig('reports/figures/straw_analysis/data/class_distribution.png', dpi=300)
    plt.show()

def load_image_by_class(class_dict, classes, class_number) -> np.ndarray:
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

def plot_pixel_intensities(class_dict: dict, frames: h5py.File) -> None:
    """
    Plot histograms of pixel intensities for each class.
    """
    
    print(" ---- Plotting Pixel Intensities ---- ")
    
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))
    
    # Create a 2x11 figure with pixel intensity hist of all images in top left corner and pixel intensity hist of each class in the remaining slots
    fig = plt.figure(figsize=(14,7))
    gs = gridspec.GridSpec(2, 11, figure=fig, wspace=0.3, hspace=0.3)
    
    ylim = 0.02
    yticks = [0, ylim/2, ylim]
    
    histtype = 'step'
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('All Classes')
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, ylim)
    ax1.set_yticks(yticks)
    ax1.set_xticks([0, 128, 255])
    ax1.set_ylabel('Density')

        
    print('Plotting pixel intensity histograms for each class...')
    class_counter = 0
    total_class_counts = np.zeros((3, 256))
    for row in range(2):
        for col in range(0, 11):
            if row == 0: col += 1
            if row == 0 and col == 11: break
            class_counts = np.zeros((3, 256))
            ax = fig.add_subplot(gs[row, col])
            print(f'Row: {row}, Col: {col}, Class {classes[class_counter]}')
            for frame_name in tqdm(class_dict['augmented'].get(float(classes[class_counter]/100), []) + class_dict['original'].get(float(classes[class_counter]/100), [])):
                image = decode_binary_image(frames[frame_name]['image'][...])
                image, bbox = rotate_and_crop_to_bbox(image, frames[frame_name]['annotations']['bbox_chute'][...])
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                for channel in range(3):
                    class_counts[channel] = np.histogram(image[:, :, channel].ravel(), bins=256, range=(0, 256))[0]
                total_class_counts += class_counts
            
            ax.hist(np.arange(0, 256), bins=256, weights=class_counts[0], density=True, histtype=histtype, alpha=0.25, color='r')
            ax.hist(np.arange(0, 256), bins=256, weights=class_counts[1], density=True, histtype=histtype, alpha=0.25, color='g')
            ax.hist(np.arange(0, 256), bins=256, weights=class_counts[2], density=True, histtype=histtype, alpha=0.25, color='b')
            ax.set_title(f'Class {classes[class_counter]}')
            ax.set_xticks([0, 128, 255])
            
            ax.set_ylim(0, ylim)
            ax.set_xlim(0, 256)
            if col == 0:
                 ax.set_ylabel('Density')
                 ax.set_yticks(yticks)
            else:
                ax.set_yticks([])
            if row == 1 and col == 5:
                ax.set_xlabel('Pixel Intensity')
                
            
            class_counter += 1
            
        

    ax1.hist(np.arange(0, 256), bins=256, weights=total_class_counts[0], density=True, histtype=histtype, alpha=0.25, color='r', label='Red Channel')
    ax1.hist(np.arange(0, 256), bins=256, weights=total_class_counts[1], density=True, histtype=histtype, alpha=0.25, color='g', label='Green Channel')
    ax1.hist(np.arange(0, 256), bins=256, weights=total_class_counts[2], density=True, histtype=histtype, alpha=0.25, color='b', label='Blue Channel')
    
    fig.legend(loc='center', ncols=3, bbox_to_anchor=(0.475, 0.935))
    
    plt.suptitle(f'Straw Pixel Intensity Histograms ({frames.filename})')
    plt.savefig('reports/figures/straw_analysis/data/pixel_intensity_histograms.png', dpi=300)
    plt.show()
    
def plot_pixel_means_and_variance(class_dict: dict, frames: h5py.File) -> None:
    """
    Plot means and variance of pixel intensities for each class.
    """  
    
    print(" ---- Plotting Pixel Means And Variance ---- ")
    
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))
    
    
    print('Plotting pixel means and variance for each class...')

    means_all = []
    variances_all = []
    stds_all = []
    for class_counter in range(len(classes)):
            means = []
            variances = []
            stds = []
            for frame_name in tqdm(class_dict['augmented'].get(float(classes[class_counter]/100), []) + class_dict['original'].get(float(classes[class_counter]/100), [])):
                image = decode_binary_image(frames[frame_name]['image'][...])
                image, bbox = rotate_and_crop_to_bbox(image, frames[frame_name]['annotations']['bbox_chute'][...])
                means = means + [np.mean(image)]
                variances = variances + [np.var(image)]
                stds = stds + [np.std(image)]
            means_all = means_all + [np.mean(means)]
            variances_all = variances_all + [np.mean(variances)]
            stds_all = stds_all + [np.mean(stds)]

            print(f'Mean of class {class_counter*5}: {np.mean(means)}')
            print(f'Variance of class {class_counter*5}: {np.mean(variances)}')
            print(f'Std of class {class_counter*5}: {np.mean(stds)}')
    
    
    fig = plt.figure(figsize=(16,4))
    plt.errorbar(classes, means_all, yerr=stds_all, fmt='o', label='Mean', capsize=5)
    plt.title('Mean Pixel Intensity of Each Class')
    plt.xlabel('Class')
    plt.xticks(classes)
    plt.ylabel('Mean Pixel Intensity')
    plt.savefig('reports/figures/straw_analysis/data/pixel_intensity_means.png', dpi=300)
    plt.show()   

def plot_mean_image_per_class(class_dict: dict, frames: h5py) -> np.ndarray:
    """Plots the mean image of each class.
    """

    print(" ---- Plotting Mean Image Per Class ---- ")
    
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))
    
    fig, ax = plt.subplots(1, 21, figsize=(17, 4))
    ax = ax.ravel()
    
    mean_images = []
    for class_counter in range(len(classes)):
        print(f'Class {classes[class_counter]}')
        mean_image = np.zeros((1370, 204, 3))
        count = 0
        for frame_name in tqdm(class_dict['augmented'].get(float(classes[class_counter]/100), []) + class_dict['original'].get(float(classes[class_counter]/100), [])):
            image = decode_binary_image(frames[frame_name]['image'][...])
            image, bbox = rotate_and_crop_to_bbox(image, frames[frame_name]['annotations']['bbox_chute'][...])
            image = cv2.resize(image, (204, 1370))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
            mean_image += image    
            count += 1
        
        mean_image = mean_image / count
        ax[class_counter].imshow(mean_image)
        ax[class_counter].set_title(f'Class {classes[class_counter]}')
        ax[class_counter].axis('off')
        mean_images.append(mean_image)
        
    plt.suptitle(f'Mean Image of Each Class ({frames.filename})')
    plt.tight_layout()
    plt.savefig('reports/figures/straw_analysis/data/mean_images.png', dpi=300)
    plt.show()
    
    return mean_images
    
def plot_variance_image_per_class(class_dict: dict, frames: h5py, mean_image: np.ndarray) -> None:
    """Plots the variance image of each class.
    """

    print(" ---- Plotting Variance Images For Each Class ---- ")
    
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))
    
    fig, ax = plt.subplots(1, 21, figsize=(17, 4))
    ax = ax.ravel()
    
    for class_counter in range(len(classes)):
        print(f'Class {classes[class_counter]}')
        variance_image = np.zeros((1370, 204, 3))

        images_in_class = 0
        # Calculate the variance image (the variance of each pixel across all images in the class)
        for frame_name in tqdm(class_dict['augmented'].get(float(classes[class_counter]/100), []) + class_dict['original'].get(float(classes[class_counter]/100), [])):
            image = decode_binary_image(frames[frame_name]['image'][...])
            image, bbox = rotate_and_crop_to_bbox(image, frames[frame_name]['annotations']['bbox_chute'][...])
            image = cv2.resize(image, (204, 1370))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
            variance_image += (image - mean_image[class_counter])**2
            images_in_class += 1
            
        # scale image to be in [0, 1]
        if np.max(variance_image) != 0 and np.min(variance_image) != 0:
            variance_image = (((variance_image - np.min(variance_image) * (1 - 0)) / (np.max(variance_image) - np.min(variance_image))) + 0)
        
        ax[class_counter].imshow(variance_image)
        ax[class_counter].set_title(f'Class {classes[class_counter]}')
        ax[class_counter].axis('off')

    plt.suptitle(f'Variance Image of Each Class ({frames.filename})')
    plt.tight_layout()
    plt.savefig('reports/figures/straw_analysis/data/variance_images.png', dpi=300)
    plt.show()    

def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate the mean squared error between two images.
    """
    return np.mean((image1 - image2)**2)

def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate the structural similarity index between two images.
    """
    return ssim(image1, image2)

def plot_mse_matrix(class_dict: dict, frames: h5py.File, mean_images: list) -> None:
    """Create a matrix of mean squared errors between all classes.
    """

    print(" ---- Plotting MSE Matrix ---- ")
    
    print('Calculating MSE matrix...')
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))
    
    mse_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            print(f'Calculating MSE between class {classes[i]} and class {classes[j]}')
            mse_matrix[i, j] = calculate_mse(mean_images[i], mean_images[j])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cax = ax.matshow(mse_matrix, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title('Mean Squared Error Matrix')
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.savefig('reports/figures/straw_analysis/data/mse_matrix.png', dpi=300)
    plt.show()

def edge_detection(image: np.ndarray, threshold1: int = 25, threshold2: int = 100) -> np.ndarray:
    """Detect edges in an image using the Canny edge detector.
    """
    edges = cv2.Canny(image, threshold1, threshold2)
    # edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    return edges

def plot_edge_detection(class_dict: dict, frames: h5py.File) -> None:
    """Plots the results of edge detection on a random image from each class.
    """

    print(" ---- Plotting Edge Detection ---- ")
    
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))

    fig, ax = plt.subplots(5, 21, figsize=(17, 24))
    
    # threshold1 = 25
    # threshold2 = 50
    
    threshold1s = [25, 50, 75, 100]
    threshold2s = [50, 100, 150, 200]
    
    for class_counter in tqdm(range(len(classes))):
        all_frames_in_class = class_dict['augmented'].get(float(classes[class_counter]/100), []) + class_dict['original'].get(float(classes[class_counter]/100), [])
        frame_name = np.random.choice(all_frames_in_class)
        image = decode_binary_image(frames[frame_name]['image'][...])
        image, bbox = rotate_and_crop_to_bbox(image, frames[frame_name]['annotations']['bbox_chute'][...])
        image = cv2.resize(image, (204, 1370))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[0, class_counter].imshow(image)
        ax[0, class_counter].axis('off')
        ax[0, class_counter].set_title(f'Class {classes[class_counter]}')
        for i, threshold1, threshold2 in (zip(range(4), threshold1s, threshold2s)):
            edges = edge_detection(image, threshold1, threshold2)
            ax[i+1, class_counter].imshow(edges, cmap='gray')
            ax[i+1, class_counter].axis('off')
        
    plt.suptitle(f'Edge Detection Results (thresh1={threshold1}, thresh2={threshold2}) ({frames.filename})')
    plt.tight_layout()
    plt.savefig('reports/figures/straw_analysis/data/edge_detection.png', dpi=300)
    plt.show()

def plot_pca(class_dict: dict, frames: h5py.File) -> None:
    """Plot PCA of each image in each class.
    """
    
    print(" ---- Plotting PCA ---- ")
    
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))
    
    num_augmented_images = len([frame for class_key in class_dict['augmented'].keys() for frame in class_dict['augmented'][class_key]])
    num_original_images = len([frame for class_key in class_dict['original'].keys() for frame in class_dict['original'][class_key]])
    total_images = num_augmented_images + num_original_images
    
    def load_image(frames, frame_name):
        image = decode_binary_image(frames[frame_name]['image'][...])
        image, bbox = rotate_and_crop_to_bbox(image, frames[frame_name]['annotations']['bbox_chute'][...])
        image = cv2.resize(image, (204, 1370))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    # Calculate PCA for each image in each class
    print('Collecting all images...')
    all_images = np.empty((total_images, 204*1370*1))
    iter_counter = 0
    all_frames = get_all_frames_by_class_order(frames, class_dict)
    for frame_name in tqdm(all_frames):
        image = load_image(frames, frame_name)
        all_images[iter_counter] = image.flatten()
        iter_counter += 1
    
    print('Fitting PCA...')
    pca = PCA(n_components=0.8)
    coords = pca.fit_transform(all_images)
    
    # Normalize
    # coords = (coords - np.mean(coords, axis=0)) / np.std(coords, axis=0)
    
    print("Collecting image coordinates...")
    coords_by_class = {}
    for class_counter in range(len(classes)):
        class_idx = get_idx_of_class(frames, class_dict, class_counter)
        for idx in class_idx:
            coord = coords[idx]
            coords_by_class[class_counter] = coords_by_class.get(class_counter, []) + [coord]
    
    # Plot the PCA of each image in each class as a scatter plot
    print('Plotting PCA of each image in each class...')
    fig = plt.figure(figsize=(10, 8))
    plt.grid()
    for class_counter in range(len(classes)):
        coords = np.array(coords_by_class[class_counter])
        plt.scatter(coords[:, 0], coords[:, 1], label=f'{classes[class_counter]}', alpha=0.5)
    plt.title('PCA of Each Image in Each Class')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(map(int,labels))[k])] )
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    plt.savefig('reports/figures/straw_analysis/data/pca.png', dpi=300)
    plt.show()

def plot_tsne(frames: h5py.File, class_dict: dict) -> None:
    """Plot t-SNE of each image in each class.
    """
    
    print(" ---- Plotting t-SNE ---- ")
    
    classes_original = list([int(x*100) for x in class_dict['original'].keys()])
    classes_augmented = list(int(x*100) for x in class_dict['augmented'].keys())
    classes = list(set(classes_original) | set(classes_augmented))
    
    def load_image(frames, frame_name):
        image = decode_binary_image(frames[frame_name]['image'][...])
        image, bbox = rotate_and_crop_to_bbox(image, frames[frame_name]['annotations']['bbox_chute'][...])
        image = cv2.resize(image, (204, 1370))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    # Calculate PCA for each image in each class
    print('Collecting all images...')
    all_frames = get_all_frames_by_class_order(frames, class_dict)
    total_images = len(all_frames)
    all_images = np.empty((total_images, 204*1370*1))
    iter_counter = 0
    for frame_name in tqdm(all_frames):
        image = load_image(frames, frame_name)
        all_images[iter_counter] = image.flatten()
        iter_counter += 1
    
    print('Fitting t-SNE...')
    model = TSNE(n_components=2, random_state=0)
    coords = model.fit_transform(all_images)
    
    # Normalize
    # coords = (coords - np.mean(coords, axis=0)) / np.std(coords, axis=0)
    
    print("Collecting image coordinates...")
    coords_by_class = {}
    for class_counter in range(len(classes)):
        class_idx = get_idx_of_class(frames, class_dict, class_counter)
        for idx in class_idx:
            coord = coords[idx]
            coords_by_class[class_counter] = coords_by_class.get(class_counter, []) + [coord]
    
    # Plot the t-SNE of each image in each class as a scatter plot
    print('Plotting t-SNE of each image in each class...')
    fig = plt.figure(figsize=(10, 8))
    plt.grid()
    for class_counter in range(len(classes)):
        coords = np.array(coords_by_class[class_counter])
        plt.scatter(coords[:, 0], coords[:, 1], label=f'{classes[class_counter]}', alpha=0.5)
    plt.title('t-SNE of Each Image in Each Class')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(map(int,labels))[k])] )
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    plt.savefig('reports/figures/straw_analysis/data/tsne.png', dpi=300)
    plt.show()
    
    
# TODO:
# - Look into some feature engineering
#    - Histogram of Oriented Gradients (HOG)
#    - Local Binary Patterns (LBP)
#    - Fourier Transform
# - Per-Class Heatmaps
#    - Heatmap of feature importance for each class e.g. using Grad-CAM
# - Shannon Entroyp
# - Mutual Information
# - Some of the stuff is hardcoded for 21 classes currently, might need to change that if we decide to go with less classes
#   - PCA and t-SNE are hardcoded for 21 classes (get_idx_of_class)


if __name__ == '__main__':
    frames = h5py.File('data/processed/augmented/chute_detection.hdf5', 'r')
    class_dictionary = get_frames_by_class(frames)
    
    ## These functions create plots of the dataset for the straw level monitoring model ##
    # plot_class_distribution(class_dictionary, frames)
    # plot_pixel_intensities(class_dictionary, frames)
    # plot_pixel_means_and_variance(class_dictionary, frames)
    # mean_images = plot_mean_image_per_class(class_dictionary, frames)
    # plot_variance_image_per_class(class_dictionary, frames, mean_images)
    # plot_mse_matrix(class_dictionary, frames, mean_images)
    # plot_edge_detection(class_dictionary, frames)
    # plot_pca(class_dictionary, frames)
    # plot_tsne(frames, class_dictionary)
    







