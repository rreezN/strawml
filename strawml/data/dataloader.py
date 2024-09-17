from __init__ import *
import torch
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from strawml.data.make_dataset import decode_binary_image
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors
import psutil

class Chute(torch.utils.data.Dataset):
    def __init__(self, data_path: str = 'data/processed/chute_detection.hdf5', data_type: str = 'train', inc_heatmap: bool = True,
                 random_state: int = 42, force_update_statistics: bool = False) -> None:
                
        self.data_path = data_path
        self.data_type = data_type
        self.inc_heatmap = inc_heatmap
        self.epsilon = 1e-6 # Small number to avoid division by zero
        # Load the data file
        self.frames = h5py.File(self.data_path, 'a')

        # Unfold the data to (segment, second)
        frame_names = list(self.frames.keys())

        # Create indices for train, test and validation
        # TODO: Does this cause overlap between the sets?
        # When we do:
        # train = Chute(data_type='train')
        # test = Chute(data_type='test')
        # are we certain that the indices are disjoint?
        self.train_indices, self.test_indices, _, _ = train_test_split(frame_names, frame_names, test_size=0.15, random_state=random_state)
        
        # Set the indices based on the data type
        if data_type == 'train':
            self.indices = self.train_indices
        elif data_type == 'test':
            self.indices = self.test_indices
        else:
            raise ValueError('data_type must be either "train" or "test"')
        
        # Define the transformation to apply to the data
        self.transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Resize((224, 224))]) 

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
        if self.inc_heatmap:
            heatmap = decode_binary_image(frame['image_diff'][...])
        
        # Standardize image wrt. training data
        img_normalize = transforms.Normalize(mean=self.train_mean, std=self.train_std)
        frame_data = img_normalize(frame_data)
        
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

        if self.inc_heatmap:
            # Standardize heatmap wrt. training data
            heatmap_normalize = transforms.Normalize(mean=self.train_hm_mean, std=self.train_hm_std)
            heatmap = heatmap_normalize(heatmap)
            frame_data = (frame_data, heatmap)
        
        # bboxes_all = np.array([bbox_chute, bbox_straw])
        bboxes_all = np.array([bbox_chute])
        bboxes_all = tv_tensors.BoundingBoxes(bboxes_all, format="XYXY", canvas_size = frame_data[0].shape[-2:])
        frame_data, bboxes_all = self.transform(frame_data, bboxes_all)
        bbox_chute = bboxes_all[0]
        # bbox_straw = bboxes_all[1]
        # labels = (bbox_chute, bbox_straw, obstructed, fullness)
        fullness = torch.Tensor(fullness)
        labels = (bbox_chute, obstructed, fullness)
            
        return frame_data, labels
            
    def convert_fullness_to_class(self, fullness: float) -> list[int]:
        """Converts the fullness value to a class label.

        Parameters:
        fullness (float): The fullness value to convert.

        Returns:
        list(int): The class label. 
        """
        
        idx = int((fullness * 10)/0.5)
        label = [0] * 20
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
            if self.inc_heatmap:
                print(f"Statistics already extracted, loading from file: {self.data_path}")
                return self.frames.attrs['mean'], self.frames.attrs['std'], self.frames.attrs['min'], self.frames.attrs['max'], self.frames.attrs['mean_hm'], self.frames.attrs['std_hm'], self.frames.attrs['min_hm'], self.frames.attrs['max_hm']
            return self.frames.attrs['mean'], self.frames['std'], self.frames.attrs['min'], self.frames.attrs['max']
        
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
        print(f'Mean: {running_mean}, Std: {running_std}, Min: {running_min}, Max: {running_max}')
        print(f'Mean HM: {running_mean_hm}, Std HM: {running_std_hm}, Min HM: {running_min_hm}, Max HM: {running_max_hm}')
        
        if self.inc_heatmap:
            return running_mean, running_std, running_min, running_max, running_mean_hm, running_std_hm, running_min_hm, running_max_hm
        else:
            return running_mean, running_std, running_mean_hm, running_std_hm
        

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

    trainset = Chute(data_type='train', inc_heatmap=True, force_update_statistics=True)
    # trainset.plot_data()
    # test_set = Platoon(data_type='test', pm_windowsize=2)
    # test_set.plot_data()
    # val_set = Platoon(data_type='val', pm_windowsize=2)
    # val_set.plot_data()
    
    print("Measuring time taken to load a batch")
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

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
        
        
        # Display last image, bboxes and labels
        images = data
        
        frame = images[0]
        heatmap = images[1]
        bbox_chute = target[0]
        # bbox_straw = target[1]
        obstructed = target[1]
        fullness = target[2]
    
    print(f'\nTotal time taken: {np.sum(durations):.2f}')
    print(f'Max duration: {np.max(durations):.2f} at index {np.argmax(durations)}')
    print(f'Min duration: {np.min(durations):.2f} at index {np.argmin(durations)}')
    print(f'Mean duration: {np.mean(durations):.2f}')
    # Print example statistics of the last batch
    print(f'Last data shape: {data[0].shape}')
    
    # convert back from normalized values
    # means = [-trainset.train_mean[0], -trainset.train_mean[1], -trainset.train_mean[2]]
    # stds = [1/(trainset.train_std[0]+trainset.epsilon), 1/(trainset.train_std[1]+trainset.epsilon), 1/(trainset.train_std[2]+trainset.epsilon)]
    # img_normalize = transforms.Compose([
    #                                    transforms.Normalize(mean=[0, 0, 0], std=stds),
    #                                    transforms.Normalize(mean=means, std=[1, 1, 1]),
    #                                    transforms.ToPILImage()
    #                                    ])
    # means = [-trainset.train_hm_mean[0], -trainset.train_hm_mean[1], -trainset.train_hm_mean[2]]
    # stds = [1/(trainset.train_hm_std[0]+trainset.epsilon), 1/(trainset.train_hm_std[1]+trainset.epsilon), 1/(trainset.train_hm_std[2]+trainset.epsilon)]
    # heatmap_normalize = transforms.Compose([
    #                                        transforms.Normalize(mean=[0, 0, 0], std=stds),
    #                                        transforms.Normalize(mean=means, std=[1, 1, 1]),
    #                                        transforms.ToPILImage()
    #                                        ])
    frame = frame.squeeze().permute(1, 2, 0)
    heatmap = heatmap.squeeze().permute(1, 2, 0)
    # frame = img_normalize(frame)
    # heatmap = heatmap_normalize(heatmap)
    
    # Display the image, heatmap and bounding boxes in a 1x2 grid
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(frame)
    ax[0].set_title('Image')
    ax[1].imshow(heatmap)
    ax[1].set_title('Heatmap')
    ax[0].axis('off')
    ax[1].axis('off')
    
    # Display the bounding boxes
    for bbox in bbox_chute:
        # delete gradient information
        bbox = bbox.detach().numpy()
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='g', facecolor='none')
        ax[0].add_patch(rect)
    
    # for bbox in bbox_straw:
    #     bbox = bbox.detach().numpy()
    #     rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='r', facecolor='none')
    #     ax[0].add_patch(rect)
    
    # Set suptitle to the fullness and obstructed labels
    plt.suptitle(f'Fullness: {trainset.convert_class_to_fullness(fullness)}, Obstructed: {obstructed.item()}')
    
    plt.show()
    
    
    