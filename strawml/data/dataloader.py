from __init__ import *
import torch
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors
import cv2

from strawml.data.make_dataset import decode_binary_image
from strawml.models.straw_classifier import chute_cropper as cc

class Chute(torch.utils.data.Dataset):
    def __init__(self, data_path: str = 'data/processed/augmented/chute_detection.hdf5', data_type: str = 'train', inc_heatmap: bool = True, inc_edges: bool = False,
                 random_state: int = 42, force_update_statistics: bool = False, data_purpose: str = "chute", image_size=(448, 448)) -> torch.utils.data.Dataset:
        
        self.image_size = image_size
        self.data_purpose = data_purpose
        self.data_path = data_path
        self.data_type = data_type
        self.inc_heatmap = inc_heatmap
        self.inc_edges = inc_edges
        self.epsilon = 1e-6 # Small number to avoid division by zero
        # Load the data file
        self.frames = h5py.File(self.data_path, 'a')
        
        # Unfold the data to (segment, second)
        frame_names = list(self.frames.keys())

        # If data purpose is straw, we remove all cropped images from the dataset
        if self.data_purpose == "straw":
            # Remove all images that have been cropped
            banned_augmentations = ['cropping', 'translation', 'rotation', 'color'] # TODO: TBD if we want to keep color
            print(f"Removing {banned_augmentations} images from the dataset (data_purpose='straw')")
            for frame_name in list(self.frames.keys()):
                attributes = self.frames[frame_name].attrs
                if 'augmented' in attributes:
                    augmentations = self.frames[frame_name].attrs['augmented']
                    if any(x in augmentations for x in banned_augmentations):
                        frame_names.remove(frame_name)

        
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
        self.transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]) # transforms.Resize((224, 224)) 

        # Store the mean and std of the training data for normalization
        if self.inc_heatmap:
            self.train_mean, self.train_std, self.train_min, self.train_max, self.train_hm_mean, self.train_hm_std, self.train_hm_min, self.train_hm_max = self.extract_means_and_stds(force_update_statistics)
        else:
            self.train_mean, self.train_std, self.train_min, self.train_max = self.extract_means_and_stds(force_update_statistics)
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
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        
        if self.inc_heatmap:
            heatmap = decode_binary_image(frame['image_diff'][...])
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
        
        # print(" ----------- LOADING NEW ITEM ------------ ")
        # print("1. Original images")
        # self.plot_data(frame_data=(frame_data, heatmap), labels=None)
        
        # Extract the labels
        anno = frame['annotations']
        # Ensure that the annotations are present
        try:
            bbox_chute = anno['bbox_chute'][...]
            # bbox_straw = anno['bbox_straw'][...]
            fullness = anno['fullness'][...]
            obstructed = anno['obstructed'][...]
            
            fullness = self.convert_fullness_to_class(fullness)
        except KeyError as e:
            # If the annotations are not present, print the error and the keys of the frame
            print(f'\nKeyError: {e} in frame {self.indices[idx]}')
            print(frame['annotations'].keys(), "\n")
        
        # Rotate and crop the image to the bounding box if we are training on the straw dataset
        if self.data_purpose == "straw":
            frame_data, bbox_chute = cc.rotate_and_crop_to_bbox(frame_data, bbox_chute)
            if self.inc_heatmap:
                heatmap, _ = cc.rotate_and_crop_to_bbox(heatmap, bbox_chute)

            # print("1.5 Rotation and cropping to bbox")
            # self.plot_data(frame_data=(frame_data, heatmap), labels = [bbox_chute])
        
        if self.inc_edges:
            edges = self.get_edge_features(frame_data)
        
        if self.inc_heatmap: frame_data = (frame_data, heatmap)
        # Transform to tensor images
        if self.inc_heatmap:
            img = self.transform(frame_data[0])
            heatmap = self.transform(frame_data[1])
            frame_data = (img, heatmap)
        else:
            frame_data = self.transform(frame_data)
        
        # print("2. Transform")
        # self.plot_data(frame_data=frame_data, labels = [bbox_chute])
        
        # Standardize image wrt. training data
        img_normalize = transforms.Normalize(mean=self.train_mean, std=self.train_std)
        
        if self.inc_heatmap:
            # Standardize heatmap wrt. training data
            img = img_normalize(frame_data[0])
            heatmap_normalize = transforms.Normalize(mean=self.train_hm_mean, std=self.train_hm_std)
            heatmap = heatmap_normalize(frame_data[1])
            frame_data = (img, heatmap)
        else:
            frame_data = img_normalize(frame_data)
        
        # print("3. Normalized")
        # self.plot_data(frame_data=frame_data, labels = [bbox_chute])
        
        
        # bboxes_all = np.array([bbox_chute, bbox_straw])
        bboxes_all = np.array([bbox_chute])
        # bboxes_all = tv_tensors.BoundingBoxes(bboxes_all, format="XYXY", canvas_size = frame_data[0].shape[-2:])
        
        
        if self.data_purpose == "straw" and self.image_size is not None:
            # Resize the image to specified size
            self.resize = transforms.Resize(self.image_size)
            if self.inc_heatmap:
                frame_data = (self.resize(frame_data[0]), self.resize(frame_data[1]))
            else:
                frame_data = self.resize(frame_data)
            
            if self.inc_edges:
                edges = self.resize(edges)
            
            # print("4. Resize")
            # self.plot_data(frame_data=frame_data, labels=[bbox_chute])
            
        bbox_chute = bboxes_all[0]
        # bbox_straw = bboxes_all[1]
        # labels = (bbox_chute, bbox_straw, obstructed, fullness)
        fullness = torch.Tensor(fullness)
        labels = (bbox_chute, obstructed, fullness)
        
        if self.inc_heatmap:
            frame_data = torch.vstack([frame_data[0], frame_data[1]])
        
        if self.inc_edges:
            frame_data = torch.vstack([frame_data, edges])
        
        if self.data_purpose == "straw":
            return frame_data, fullness
        
        return frame_data, labels
            
    def convert_fullness_to_class(self, fullness: float) -> list[int]:
        """Converts the fullness value to a class label.

        Parameters:
        fullness (float): The fullness value to convert.

        Returns:
        list(int): The class label. 
        """
        
        idx = int(fullness/0.05)
        label = [0] * 21
        label[idx] = 1
        
        return label
        
    
    def convert_class_to_fullness(self, label: list[int]) -> float:
        """Converts the class label to a fullness value.

        Parameters:
        label (list(int)): The class label to convert.

        Returns:
        float: The fullness value. 
        """
        
        idx = np.argmax(label)
        fullness = (idx * 0.5) / 10
        
        return fullness


    def extract_means_and_stds(self, force_update_statistics: bool = False):
        """Extracts the mean and standard deviation of the training data for normalization.

        Returns:
        -------
        tuple: mean, std
        """
        if self.frames is None:
            return
        
        if "mean" in list(self.frames.attrs.keys()) and not force_update_statistics:
            print(f"Statistics already extracted, loading from file: {self.data_path}")
            if self.inc_heatmap:
                return self.frames.attrs['mean'], self.frames.attrs['std'], self.frames.attrs['min'], self.frames.attrs['max'], self.frames.attrs['mean_hm'], self.frames.attrs['std_hm'], self.frames.attrs['min_hm'], self.frames.attrs['max_hm']
            return self.frames.attrs['mean'], self.frames.attrs['std'], self.frames.attrs['min'], self.frames.attrs['max']
        
        import strawml.data.extract_statistics as es
        
        running_mean = None
        running_min = None
        running_max = None
        zeros_array = None
        n = 0
        
        running_mean_hm = None
        running_min_hm = None
        running_max_hm = None
        zeros_array_hm = None
        n_hm = 0
        
        pbar = tqdm(range(len(self.train_indices)), desc='Extracting statistics', leave=True)
        for idx in pbar:
            new_frame = self.frames[self.train_indices[idx]]
            
            image = decode_binary_image(new_frame['image'][...])
            image = image.reshape(image.shape[2], image.shape[0], image.shape[1])
            if self.inc_heatmap:
                image_diff = decode_binary_image(new_frame['image_diff'][...])
                image_diff = image_diff.reshape(image_diff.shape[2], image_diff.shape[0], image_diff.shape[1])
            
            if running_mean is None:
                if self.inc_heatmap:
                    existing_aggregate_hm = (0, np.zeros(image.shape[0]), np.zeros(image.shape[0]))
                    running_min_hm = np.min(image_diff, axis=(1, 2))
                    running_max_hm = np.max(image_diff, axis=(1, 2))
                    
                existing_aggregate = (0, np.zeros(image.shape[0]), np.zeros(image.shape[0]))
                running_min = np.min(image, axis=(1, 2))
                running_max = np.max(image, axis=(1, 2))
            else:
                if self.inc_heatmap:
                    existing_aggregate_hm = (n_hm, running_mean_hm, running_s_hm)
                existing_aggregate = (n, running_mean, running_s)

            if self.inc_heatmap:
                new_hm = np.mean(image_diff, axis=(1, 2))
                n_hm, running_mean_hm, running_s_hm = es.update(existing_aggregate=existing_aggregate_hm, new_value=new_hm)
                running_min_hm = np.minimum(running_min_hm, np.min(image_diff, axis=(1, 2)))
                running_max_hm = np.maximum(running_max_hm, np.max(image_diff, axis=(1, 2)))
                
            new_image = np.mean(image, axis=(1, 2))
            n, running_mean, running_s = es.update(existing_aggregate=existing_aggregate, new_value=new_image)
            running_min = np.minimum(running_min, np.min(image, axis=(1, 2)))
            running_max = np.maximum(running_max, np.max(image, axis=(1, 2)))

        
        # Finalize the statistics
        if self.inc_heatmap:
            running_mean_hm, _, running_std_hm = es.finalize((n_hm, running_mean_hm, running_s_hm))
            self.frames.attrs['mean_hm'] = running_mean_hm
            self.frames.attrs['std_hm'] = running_std_hm
            self.frames.attrs['min_hm'] = running_min_hm
            self.frames.attrs['max_hm'] = running_max_hm
            
        running_mean, _, running_std = es.finalize((n, running_mean, running_s))
        self.frames.attrs['mean'] = running_mean 
        self.frames.attrs['std'] = running_std
        self.frames.attrs['min'] = running_min
        self.frames.attrs['max'] = running_max
        
        print("Statistics extracted:")
        if self.inc_heatmap:
            print(f'Mean HM: {running_mean_hm}, Std HM: {running_std_hm}, Min HM: {running_min_hm}, Max HM: {running_max_hm}')
        else:
            print(f'Mean: {running_mean}, Std: {running_std}, Min: {running_min}, Max: {running_max}')
        
        if self.inc_heatmap:
            return running_mean, running_std, running_min, running_max, running_mean_hm, running_std_hm, running_min_hm, running_max_hm
        else:
            return running_mean, running_std, running_min, running_max
        

    def plot_data(self, frame_idx: int = 0, frame_data=None, labels=None):
        """Plots the data and labels for the first frame in the dataset."""
        
        if frame_data is None:
            frame_data, labels = self.__getitem__(frame_idx)
        
        if self.inc_heatmap:
            image = frame_data[:3, :, :]
            heatmap = frame_data[3:, :, :]
            
            if type(image) in [torch.Tensor, tv_tensors._image.Image]:
                if len(image.shape) == 4: 
                    image = image.squeeze()
                    heatmap = heatmap.squeeze()
                if torch.min(image) < 0:
                    img_unnormalize = transforms.Normalize(mean = [-m/s for m, s in zip(self.train_mean, self.train_std)],
                                                        std = [1/s for s in self.train_std])
                    hm_unnormalize = transforms.Normalize(mean = [-m/s for m, s in zip(self.train_hm_mean, self.train_hm_std)],
                                                        std = [1/s for s in self.train_hm_std])
                    image = img_unnormalize(image)
                    heatmap = hm_unnormalize(heatmap)
                image = image.permute(1, 2, 0)
                heatmap = heatmap.permute(1, 2, 0)
                image = image.detach().numpy()
                heatmap = heatmap.detach().numpy()

                
            
            # 1x2 grid for image and heatmap with bounding boxes
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image)
            ax[0].set_title('Image')
            ax[1].imshow(heatmap)
            ax[1].set_title('Heatmap')
            ax[0].axis('off')
            ax[1].axis('off')
            if self.data_purpose == "straw":
                plt.suptitle("Straw Dataset, Fullness: " + str(np.round(100*self.convert_class_to_fullness(labels).item())) +"%")
            else:
                plt.suptitle("Chute Dataset")
        else:
            image = frame_data
            
            # Display the image
            plt.imshow(image.squeeze().permute(1, 2, 0))
            if self.data_purpose == "straw":
                plt.title("Straw Dataset, Fullness: " + str(np.round(100*self.convert_class_to_fullness(labels).item())) +"%")
            else:
                plt.title("Chute Dataset")
            plt.axis('off')
            
        
        if labels is not None:
            # Display the bounding boxes
            if self.data_purpose == "straw":
                bbox = labels
            else:
                bbox = labels[0]
            
            import matplotlib.patches as patches
            # delete gradient information
            # bbox = bbox.detach().numpy()
            rect = patches.Polygon([[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]], [bbox[6], bbox[7]]], edgecolor='g', facecolor='none')
            ax[0].add_patch(rect)
        
        plt.show()
        
    
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
    
    def get_edge_features(self, frame_data: np.ndarray) -> torch.Tensor:
        """Extracts the edges from the frame data.
        
        Parameters:
        frame_data (np.ndarray): The frame data to extract the edges from. Shape: (H, W, C)
        
        Returns:
        torch.Tensor: The edges of the frame data. Shape: (C, H, W)
        """
        edges = cv2.Canny(frame_data, 100, 200)
        edges = edges.reshape(1, edges.shape[0], edges.shape[1])
        edges = torch.from_numpy(edges)/255
        # TODO: Figure out normalization, might need to do a calculate statistics on the edge images as well, mean is probably 0.0x and std is probably ~1
        return edges
    
    def print_arguments(self):
        dataset_size = len(self.train_indices) if self.data_type == 'train' else len(self.test_indices)
        print(f'Parameters: \n \
                    Data Path:          {self.data_path}\n \
                    Include Heatmaps:   {self.inc_heatmap} \n \
                    Data Type:          {self.data_type}\n \
                    Data size:          {dataset_size}\n \
                    ')

if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    print("---- CHUTE DETECTION DATASET ----")
    train_set = Chute(data_type='train', inc_heatmap=True, force_update_statistics=False)
    
    # trainset.plot_data()
    # test_set = Platoon(data_type='test', pm_windowsize=2)
    # test_set.plot_data()
    # val_set = Platoon(data_type='val', pm_windowsize=2)
    # val_set.plot_data()
    
    print("Measuring time taken to load a batch")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

    start = time.time()
    i = 0
    durations = []
    pbar = tqdm(train_loader, unit="batch", position=0, leave=False)
    for data, target in train_loader:
        end = time.time()
        # asses the shape of the data and target
        duration = end-start
        durations += [duration]
        pbar.set_description(f'Batch {i+1}/{len(train_loader)} Avg. duration: {np.mean(durations):.2f}s')
        pbar.update(1)
        i+= 1
        start = time.time()
        
        
        # # Display last image, bboxes and labels
        # images = data
        
        # frame = images[:,0,:, :]
        # heatmap = images[:, 1, :, :]
        # bbox_chute = target[0]
        # # bbox_straw = target[1]
        # obstructed = target[1]
        # fullness = target[2]
        
        # Skip timing dataloader
        if i > 0:
            break
    
    print(f'\nTotal time taken: {np.sum(durations):.2f}')
    print(f'Max duration: {np.max(durations):.2f} at index {np.argmax(durations)}')
    print(f'Min duration: {np.min(durations):.2f} at index {np.argmin(durations)}')
    print(f'Mean duration: {np.mean(durations):.2f}')
    # Print example statistics of the last batch
    print(f'Last data shape: {data[0].shape}')
    
    # train_set.plot_data(frame_idx=9)
    
    print("---- STRAW DETECTION DATASET ----")
    train_set = Chute(data_type='train', inc_heatmap=True, inc_edges=True, force_update_statistics=False, data_purpose="straw", image_size=(1370//2, 204//2))
    
    train_set.plot_data(frame_idx=0)
    
    
    
    
    
    