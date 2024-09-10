from __init__ import *
import torch
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from strawml.data.make_dataset import decode_binary_image
from torchvision import transforms
import psutil

class Platoon(torch.utils.data.Dataset):
    def __init__(self, data_path: str = 'data/processed/annotated_images.hdf5', data_type: str = 'train', inc_heatmap: bool = True,
                 random_state: int = 42) -> None:
                
        self.data_path = data_path
        self.data_type = data_type
        self.inc_heatmap = inc_heatmap
        self.epsilon = 1e-6 # Small number to avoid division by zero
        # Load the data file
        self.frames = h5py.File(self.data_path, 'r')

        # Unfold the data to (segment, second)
        frame_names = list(self.frames.keys())

        # Create indices for train, test and validation
        self.train_indices, self.test_indices, _, _ = train_test_split(frame_names, frame_names, test_size=0.15, random_state=random_state)
        
        # Set the indices based on the data type
        if data_type == 'train':
            self.indices = self.train_indices
        elif data_type == 'test':
            self.indices = self.test_indices
        else:
            raise ValueError('data_type must be either "train" or "test"')
        
        # Define the transformation to apply to the data
        self.transform = transforms.Compose([transforms.ToTensor()]) 

        # Store the mean and std of the training data for normalization
        if self.inc_heatmap:
            self.train_mean, self.train_std, self.train_hm_mean, self.train_hm_std = self.extract_means_and_stds()
        else:
            self.train_mean, self.train_std = self.extract_means_and_stds()
        # Store the mean and std of the training data for normalization       
        self.print_arguments()

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Fetches the data sample and its corresponding labels for the given index.

        Parameters:
        idx (int): The index of the data sample to fetch.

        Returns:

        """
        # Define variable to contain the current segment data
        frame = self.frames[self.indices[idx]]
        frame_data = decode_binary_image(frame['image'][...])
        if self.inc_heatmap:
            heatmap = decode_binary_image(frame['image_diff'][...])
        
        # Standardize image wrt. training data
        frame_data = (frame_data - self.train_mean) / (self.train_std + + self.epsilon)
        frame_data = self.transform(frame_data)
        if self.inc_heatmap:
            # Standardize heatmap wrt. training data
            heatmap = (heatmap - self.train_hm_mean) / (self.train_hm_std + self.epsilon)
            heatmap = self.transform(heatmap)
            frame_data = (frame_data, heatmap)

        # Extract the labels
        anno = frame['annotations']
        # Ensure that the annotations are present
        try:
            bbox_chute = anno['bbox_chute'][...]
            bbox_straw = anno['bbox_straw'][...]
            fullness = anno['fullness'][...]
            obstructed = anno['obstructed'][...]
        except KeyError as e:
            # If the annotations are not present, print the error and the keys of the frame
            print(f'\nKeyError: {e} in frame {self.indices[idx]}')
            print(frame['annotations'].keys(), "\n")

        labels = (bbox_chute, bbox_straw, obstructed, fullness)
        return frame_data, labels



    def extract_means_and_stds(self):
        """Extracts the mean and standard deviation of the training data for normalization.

        Returns:
        -------
        tuple: mean, std
        """
        # Define variable to contain the current segment data
        all_data = np.array([])
        if self.inc_heatmap:
            all_hm_data = np.array([])
        pbar = tqdm(range(len(self.train_indices)), desc='Extracting mean and std', leave=True)
        for idx in pbar:
            frame = self.frames[self.train_indices[idx]]
            image = decode_binary_image(frame['image'][...])
            if len(all_data) == 0:
                all_data = image
            else:
                all_data = np.vstack((all_data, image))

            if self.inc_heatmap:
                heatmap = decode_binary_image(frame['image_diff'][...])
                if len(all_hm_data) == 0:
                    all_hm_data = heatmap
                else:
                    all_hm_data = np.vstack((all_hm_data, heatmap))
                pbar.set_postfix_str(f"Total RAM Usage (GB): {np.round(psutil.virtual_memory().used / 1e9, 2)}")

        im_mean = np.mean(all_data, axis=0)
        im_std = np.std(all_data, axis=0)
        hm_mean = np.mean(all_hm_data, axis=0)
        hm_std = np.std(all_hm_data, axis=0)

        if self.inc_heatmap:
            return im_mean, im_std, hm_mean, hm_std
        return im_mean, im_std

    def plot_data(self):
        """Plots the histograms of the GM data before and after transformation.
        """
        ...
    
    def get_data_shape(self):
        """Returns the shape of the data and target tensors.

        Returns:
        -------
            If feature_extraction is False:
            tuple: features.shape, targets.shape
            Else:
            tuple: features.shape, segment_nr, second_nr
        """
        if self.inc_heatmap:
            frame_data, labels = self.__getitem__(0)
            # frame_data is a tuple of (image, heatmap). Extract the image shape
            return frame_data[0].shape
        else:
            frame_data, labels = self.__getitem__(0)
            return frame_data.shape
    
    def print_arguments(self):
        print(f'Arguments: \n \
                    Data Path:          {self.data_path}\n \
                    Include Heatmaps:   {self.inc_heatmap} \n \
                    Data size:           \n \
                        - Train:        {len(self.train_indices)} \n \
                        - Test:         {len(self.test_indices)} \n \
                    ')

if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    trainset = Platoon(data_type='train')
    # trainset.plot_data()
    # test_set = Platoon(data_type='test', pm_windowsize=2)
    # test_set.plot_data()
    # val_set = Platoon(data_type='val', pm_windowsize=2)
    # val_set.plot_data()
    
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

    start = time.time()
    i = 0
    durations = []
    for data, target in train_loader:
        end = time.time()
        # asses the shape of the data and target
        duration = end-start
        durations += [duration]
        print(f'Index: {i}, Time: {duration}.')
        i+= 1
        start = time.time()
    print(f'Mean duration: {np.mean(durations)}')
    # Print example statistics of the last batch
    print(f'Last data shape: {data[0].shape}')