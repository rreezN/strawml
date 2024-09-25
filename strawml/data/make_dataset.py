from __init__ import *
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm 
import datetime
import time
from argparse import ArgumentParser, Namespace
import psutil
from sklearn.model_selection import train_test_split
import shutil
import json


def decode_binary_image(image: bytes) -> np.ndarray:
    """
    Decodes a binary image and returns it as a NumPy array.

    ...

    Parameters
    ----------
    image : bytes
        The binary image data.
    
    Returns
    -------
    np.ndarray
        The decoded image as a NumPy array.
    """
    # Convert the binary data to a NumPy array
    image_array = np.frombuffer(image, np.uint8)
    # Decode the image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def extract_frames_from_video(video_name: str, 
                              video_path: str, 
                              current_video_nr: int, 
                              total_nr_videos: int, 
                              hdf5_file: str,
                              video_id: str,
                              frame_nr: int,
                              save_individual_images: bool = False,
                              image_id = None,
                              fbf: int = 30
                              ) -> int:
    """
    This function takes a video file and extracts the frames from it. The frames are then saved to an HDF5 file.
    Depending on the value of the fbf (frames between frames) parameter, only every fbf-th frame is saved to the HDF5 file.
    The function also has the option to save the individual images to the temp_images folder for validation purposes.
    ...

    Parameters
    ----------
    video_name  :   str
        The name of the video file.
    video_path  :   str
        The path to the video file.
    current_video_nr    :   int
        The current video number being processed.
    total_nr_videos :   int
        The total number of videos in the folder.
    hdf5_file   :   str
        The path to the HDF5 file where the frames are saved.
    video_id    :   str
        The unique ID of the video.
    frame_nr    :   int
        The current frame number.
    save_individual_images   :   bool
        Whether to save the individual images to the temp_images folder.
    image_id    :   int
        The current image number.
    fbf :   int
        The number of frames between frames to extract.
    
    Returns
    -------
    int
        The number of frames extracted from the video
    """

    print("Frame extraction started...")
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    # Get the total number of frames in the video
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the last 9 letters/digits of the video name
    # Initialise the progress bar
    pbar = tqdm(total=video_length, desc=f"Extracting frames from '[...]{video_name[-9:]}' ({current_video_nr}/{total_nr_videos}) (0 saved)", position=0)
    # Save the current frame as the previous frame for the next iteration
    prev_frame = None
    # A counter to keep track of the number of frames processed so far in the video -> thereby allowing for fpf (frames per frame) extraction
    count = -1
    # A counter to keep track of the number of frames saved to the HDF5 file so far -> allowing for continous numbering of the frames
    saved_frame_count = 0

    # Loop through the frames in the video
    while True:
        # Update the progress bar with the current frame number and the total RAM usage
        pbar.set_postfix_str(f"Total RAM Usage (GB): {np.round(psutil.virtual_memory().used / 1e9, 2)}")

        # Skip the first frame as it will be used as the previous frame and skip it if the count is not a multiple of fbf
        if count == -1: #or count % fbf != 0:
            # Read the current frame from the video
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if no more frames are left
            prev_frame = frame  # Handle the unique part of the first if block
            pbar.update(1)
            count += 1
            continue
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, saved_frame_count*fbf+1) # we say +1 because we want to skip the first frame as it is the previous frame (for the first iteration)
            # Read the current frame from the video
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if no more frames are left


        # save frame to temp_images folder if save_individual_images is True
        if save_individual_images:
            cv2.imwrite(f'data/raw/temp_images/frame_{image_id}.jpg', frame)

        # Calculate the difference between two consecutive frames 
        frame_diff = cv2.absdiff(frame, prev_frame)
        
        # Save the current frame as the previous frame for the next iteration
        prev_frame = frame.copy()

        # Save the frames to the HDF5 file
        success, frame = cv2.imencode('.jpg', frame)
        if success:
            success, frame_diff = cv2.imencode('.jpg', frame_diff)
            if success:
                save_frames_to_hdf5(frame, frame_diff, hdf5_file, video_id, frame_nr+saved_frame_count, count)
        # Increment the image ID if save_individual_images is True
        if save_individual_images:
            image_id += 1

        # Increment the frame count and the total frame count
        count += 1
        saved_frame_count += 1
        pbar.set_description(f"Extracting frames from '{video_name[-9:]}' ({current_video_nr}/{total_nr_videos}) ({saved_frame_count} saved)")
        pbar.update(fbf)


    pbar.close() # Close the progress bar
    cap.release() # Release the video capture object
    return frame_nr + saved_frame_count

def save_frames_to_hdf5(frame: np.ndarray,
                        frame_diff: np.ndarray,
                        hdf5_file: str,
                        video_id: str,
                        frame_nr: int,
                        count: int) -> None:
    """
    Given a frame and its difference with the previous frame, this function saves 
    the frames to an existing HDF5 file. The frames are saved as datasets in a 
    group named after the frame number, and the video ID is saved as an attribute 
    allowing for easy retrieval upon debugging or analysis.

    ...

    Parameters
    ----------
    frame   :   np.ndarray
        The frame to save.
    frame_diff  :   np.ndarray
        The difference between the current frame and the previous frame.
    hdf5_file   :   str
        The path to the HDF5 file where the frames are saved.
    video_id    :   str
        The unique ID of the video.
    frame_nr    :   int
        The frame number used to name the group in the HDF5 file.
    count   :   int
        The frame nr count in the video.  
    """
    with h5py.File(hdf5_file, 'a') as hf:
        # create a group for the video
        group_name = f'frame_{frame_nr}'
        if group_name in hf:
            print(f"Warning: The group {group_name} already exists. Skipping...")
            return
        # Create a group for the video
        group = hf.create_group(group_name)

        # add image as datasets to the group
        dataset_name_image = 'image'
        image = np.asarray(frame)
        group.create_dataset(dataset_name_image, data=image)

        # add image_diff as datasets to the group
        dataset_name_diff = 'image_diff'
        image_diff = np.asarray(frame_diff)
        group.create_dataset(dataset_name_diff, data=image_diff)

        # Add the video ID as an attribute to the dataset
        group.attrs['video ID'] = video_id
        group.attrs['og_frame_nr'] = count
        group.attrs['augmented'] = "None"
        # print(f"\nog_frame_nr: {count} saved to {group_name}")

def image_extractor(video_folder: str, 
                    hdf5_file: str, 
                    dataset_name: str = "straw-chute", 
                    description: str = "Dataset made for a master's project at Meliora Bio.",
                    overwrite_seconds: int = 3,
                    save_individual_images: bool = False,
                    fbf: int = 24
                    ) -> None:
    """
    Works as a wrapper function for the extract_frames_from_video function. This function
    loops through the video files in the video folder and extracts the frames from each video.
    The frames are then saved to an HDF5 file.
    ...

    Parameters
    ----------
    video_folder    :   str
        The path to the folder containing the videos.
    hdf5_file   :   str
        The path to the HDF5 file where the frames are saved.
    dataset_name    :   str
        The name of the dataset.
    description :   str
        The description of the dataset.
    overwrite_seconds   :   int
        The number of seconds to wait before overwriting any existing HDF5 file.
    save_individual_images   :   bool
        Whether to save the individual images to the temp_images folder.
    fbf :   int
        The number of frames between frames to extract.
    """
    # check if the video folder is a file or a folder
    if not os.path.isdir(video_folder):
        video_files = [video_folder]
    else:
        # Get the video files in the video folder
        video_files = os.listdir(video_folder) # Get the paths inside of the video folder
    # Get the unique video IDs from the video files
    unique_video_ids = [video_file.split('.')[0] for video_file in video_files]
    # check if the hdf5 file already exists
    if os.path.exists(hdf5_file):
        print(f"Warning: The file {hdf5_file} already exists and will be overwritten in {overwrite_seconds} seconds.")
        for i in range(overwrite_seconds):
            print(f"Overwriting in {overwrite_seconds-i} second(s)...")
            time.sleep(1)
        # remove the file
        os.remove(hdf5_file)
    
    # create the hdf5 file
    with h5py.File(hdf5_file, 'a') as hf:
        # add global attributes
        hf.attrs['dataset_name'] = dataset_name
        hf.attrs['description'] = description
        hf.attrs['date_created'] = np.bytes_(str(datetime.datetime.now()))

    # Initialise the image ID according to the save_individual_images parameter
    if save_individual_images:
        image_id = 0
    else:
        image_id = None

    # Loop through the video files and extract the frames
    frame_nr = 0
    for file_nr, video_name in enumerate(video_files):
        frame_nr = extract_frames_from_video(video_name=video_name, 
                                             video_path=os.path.join(video_folder, video_name), 
                                             current_video_nr=file_nr+1,
                                             total_nr_videos=len(video_files),
                                             save_individual_images=save_individual_images,
                                             image_id=image_id,
                                             hdf5_file=hdf5_file,
                                             video_id=unique_video_ids[file_nr], 
                                             frame_nr=frame_nr,
                                             fbf=fbf)

def hdf5_to_yolo(hdf5_file: str, 
                 with_augmentation: bool = True, 
                 sizes: list[float] = [0.8,0.1,0.1]) -> None:
    """
    Accomodates the conversion of the frames in the HDF5 file to the YOLO format. The function
    loops through the frames in the HDF5 file and saves the images and labels in the YOLO format.
    The images are saved to the 'data/processed/yolo_format' folder, and the labels are saved in
    the same folder with the same name as the image file but with a .txt extension. 

    The most important things are:
        # image name and label file should have same name
        # image name should be in the format: frame_{frame_nr}.jpg
        # label file should be in the format: frame_{frame_nr}.txt
        # The label should contain the following information with the following syntax,
            class_index x1 y1 x2 y2 x3 y3 x4 y4

    ...

    Parameters
    ----------
    hdf5_file  :   str
        The path to the HDF5 file containing the frames.
    with_augmentation   :   bool
        Whether to include augmented images in the YOLO format.
    
    Raises
    ------
    FileNotFoundError
        If the HDF5 file does not exist.
    """
    # First ensure the file exists
    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"The file {hdf5_file} does not exist.")
    
    save_path = 'data/processed/yolo_format'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        # remove the folder and create a new one
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    
    # Load json file
    with open('data/processed/augmented/frame_to_augment.json', 'r') as f:
        frame_to_augment_frame = json.load(f)
    frame_names = list(frame_to_augment_frame.keys())
    
    # split frames in train, test and validation with 70%, 20% and 10% respectively
    # random state = 42
    train_size, test_size, val_size = sizes
    train_indices, test_indices, _, _ = train_test_split(frame_names, frame_names, test_size=test_size+val_size, random_state=42)
    test_indices, val_indices, _, _ = train_test_split(test_indices, test_indices, test_size=test_size/(test_size+val_size), random_state=42)
    # create a dictionary to store the indices
    index_dict = {'train': train_indices, 'test': test_indices, 'val': val_indices}

    with h5py.File(hdf5_file, 'r') as hf:        
        # create a progress bar
        pbar = tqdm(total = len(frame_names), desc="Converting HDF5 to YOLO format", position=0)
        
        # loop through the datasets and save the images and labels
        for key, val in index_dict.items():
            pbar.set_description(f"Converting HDF5 to YOLO format: {key} set")
            if not os.path.exists(f'{save_path}/{key}'):
                os.makedirs(f'{save_path}/{key}')
            for frame in val:
                if not with_augmentation:
                    if hf[frame].attrs['augmented'] != "None":
                        continue
                fs_ = [frame] + [f"frame_{i}" for i in frame_to_augment_frame[frame]]
                for fs in fs_:
                    image_bytes = hf[fs]['image'][...]
                    # decode the binary image
                    image = decode_binary_image(image_bytes)
                    # Save the image to a file
                    cv2.imwrite(f'{save_path}/{key}/{fs}.jpg', image)

                    # Now we load the bounding boxes
                    bbox = hf[fs]['annotations']['bbox_chute'][...]
                    # normalize the bounding box coordinates to values between 0 and 1 
                    for i in range(len(bbox)):
                        bbox[i] /= image.shape[1] if i % 2 == 0 else image.shape[0]
                    # add the class index to the label
                    label = np.insert(bbox, 0, 0)
                    # Save the label to a file
                    with open( f'{save_path}/{key}/{fs}.txt', 'w') as f:
                        f.write(' '.join(map(str, label)))
                pbar.update(1)

def validate_image_extraction(hdf5_file: str,
                              save_individual_images) -> None:
    """
    This function validates the image extraction process by comparing the original 
    image with the decoded image to check for any differences. The function also
    checks if the frames are continuous and prints a warning if any frames are missing.

    ...

    Parameters
    ----------
    hdf5_file   :   str
        The path to the HDF5 file containing the frames.
    save_individual_images   :   bool
        Whether to save the individual images to the temp_images folder.
    """
    # First ensure that all frames are continuous
    with h5py.File(hdf5_file, 'r') as hf:
        all_frames = hf.keys()
        
        # sort frames by number
        all_frames = sorted([int(frame.split('_')[1]) for frame in all_frames])
        
        for i, frame in enumerate(all_frames):
            # Skip first frame
            if i == 0:
                continue
            # check if frames are continuous
            if frame - all_frames[i-1] != 1:
                print(f"Warning: Frame {frame-1} is missing. Frame {frame} is not continuous with the previous frame {all_frames[i-1]}")
            
        # If the save_individual_images is False, return 
        if not save_individual_images:
            return
        # make sure the temp_images folder exists
        if not os.path.exists('data/raw/temp_images') and save_individual_images:
            raise FileNotFoundError("The temp_images folder does not exist. Please run the image_extractor function with the save_individual_images argument set to True.")
        # randomly select a frame
        frame = np.random.choice(list(hf.keys()))
        image_bytes = hf[frame]['image'][...]
        # decode the binary image
        image = decode_binary_image(image_bytes)
        # Load the original image from the temp_images folder
        original_image = cv2.imread(f'data/raw/temp_images/{frame}.jpg')
        # calculate the absolute difference between the two images
        diff = cv2.absdiff(original_image, image)
        # show the images side by side in the same window
        cv2.imshow(f'{frame} - Original Image vs Extracted Image (absdiff={np.sum(diff)})', np.hstack([original_image, image]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def print_hdf5_tree(hdf5_file: str, 
                    group_name: str = "/", 
                    indent: int = 0) -> None:
    """
    Recursively prints the structure of an HDF5 file.

    ...

    Parameters
    ----------
    hdf5_file   :   str
        The path to the HDF5 file.
    group_name  :   str
        The name of the group (default is root '/').
    indent  :   int
        The current level of indentation (used for recursive calls).
    """
    # Open the HDF5 file
    with h5py.File(hdf5_file, 'r') as f:
        def print_group(group, indent):
            # Print attributes of the current group/dataset
            if isinstance(group, h5py.Group):
                print(' ' * indent + f"Group: {group.name}")
            else:
                print(' ' * indent + f"Dataset: {group.name}")
                print(' ' * (indent + 4) + f"├── shape: {group.shape}")
                print(' ' * (indent + 4) + f"├── dtype: {group.dtype}")
            
            if group.attrs:
                print(' ' * (indent + 4) + "└── Attributes:")
                for attr_name, attr_value in group.attrs.items():
                    print(' ' * (indent + 8) + f"└── {attr_name}: {attr_value}")
            
            # Iterate over sub-groups and datasets
            if isinstance(group, h5py.Group):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        print_group(item, indent + 4)
                    elif isinstance(item, h5py.Dataset):
                        print_group(item, indent + 4)
        
        # Start printing from the root group or a specific group
        print_group(f[group_name], indent)

def place_digits_on_chute_images(hdf5_file: str) -> None:
    """
    HWD+ Data: https://link.springer.com/article/10.1007/s42979-022-01494-2#Sec5
               https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or

    """
    def internal_image_operations(image: np.ndarray) -> np.ndarray:
        """
        Perform random operations on the image to simulate the chute environment. It crops the 
        image to 500 x 500 pixels and randomly rotates the image between -180 and 180 degrees, while ensuring
        that the cropped image is inside the original image.
        """
        # sample a random angle between -45 and 45 degrees
        angle = np.random.randint(-180, 181)
        # rotate the image
        M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        # randomly crop the image to 254 x 254 pixels
        x = np.random.randint(0, image.shape[1] - 500)
        y = np.random.randint(0, image.shape[0] - 500)
        image = image[y:y+500, x:x+500]
        return image 

    digit_images = np.load("D:/HCAI/msc/strawml/data/interim/digits/Images(28x28).npy")
    digit_info = np.load("D:/HCAI/msc/strawml/data/interim/digits/WriterInfo.npy")

    # We sort the digit images by the digit labels
    digit_images = digit_images[np.argsort(digit_info[:,0])]
    digit_info = digit_info[np.argsort(digit_info[:,0])]
    digit_labels = digit_info[:,0]
    frame_nr = 0
    X_train, X_val, y_train, y_val = train_test_split(digit_images, digit_labels, test_size=0.2, random_state=42, stratify=digit_labels)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify=y_val)

    data_dict = {'train': X_train, 'val': X_val, 'test': X_test}
    label_dict = {'train': y_train, 'val': y_val, 'test': y_test}

    # let us create a suitable background image for the digits by loading a chute image and taking image[:1440, :1250]
    img = cv2.imread("D:/HCAI/msc/strawml/data/processed/chute_image_for_backgroud_creation.jpg")[:1440, :1250]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    import matplotlib.pyplot as plt
    for data_type in data_dict.keys():
        # create data_type folder if it does not exist
        if not os.path.exists(f'data/processed/digits_on_chute_images/{data_type}'):
            os.makedirs(f'data/processed/digits_on_chute_images/{data_type}')
        for digit, label in tqdm(zip(data_dict[data_type], label_dict[data_type]), total=len(data_dict[data_type]), desc=f"Placing digits on chute images ({data_type})"):

            image = img.copy().astype(np.uint8)
            # perform random operation on the image
            image = internal_image_operations(image)
            # To ensure that when rotating the background stays the same color
            digit_image = digit.astype(np.uint8)
            digit_image = 255 - digit_image
            # sample a random location on the chute image
            x = np.random.randint(0, image.shape[1] - (digit_image.shape[1]+25))
            y = np.random.randint(0, image.shape[0] - (digit_image.shape[0]+25))
            # Rotate the digit_image
            # sample a random angle between -45 and 45 degrees
            angle = np.random.randint(-20, 21)
            (h, w) = digit_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_digit_image = cv2.warpAffine(digit_image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            # Create a mask for the rotated digit_image
            _, mask = cv2.threshold(rotated_digit_image, 254, 255, cv2.THRESH_BINARY_INV)

            # Convert mask to uint8 if necessary
            mask = mask.astype(np.uint8)

            # expand the rotated_digit_image to 3 channels
            rotated_digit_image = cv2.cvtColor(rotated_digit_image, cv2.COLOR_GRAY2RGB)
            # Convert rotated_digit_image to uint8 if necessary
            rotated_digit_image = rotated_digit_image.astype(np.uint8)

            # Sample a random location on the image
            x = np.random.randint(0, image.shape[1] - (rotated_digit_image.shape[1] + 25))
            y = np.random.randint(0, image.shape[0] - (rotated_digit_image.shape[0] + 25))

            # Extract the region of interest (ROI) from the image
            roi = image[y:y+rotated_digit_image.shape[0], x:x+rotated_digit_image.shape[1]]

            # Blend the rotated digit_image with the ROI using the mask
            roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            roi_fg = cv2.bitwise_and(rotated_digit_image, rotated_digit_image, mask=mask)
            blended = cv2.add(roi_bg, roi_fg)
            blended = 255 - blended
            # Place the blended result back into the image
            image[y:y+rotated_digit_image.shape[0], x:x+rotated_digit_image.shape[1]] = blended
            # save the image to a file
            cv2.imwrite(f'data/processed/digits_on_chute_images/{data_type}/frame_{frame_nr}.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # get the bounding box coordinates of the digit image on the chute image, x1, y1, x2, y2, x3, y3, x4, y4        
            bbox = np.array([x/image.shape[1], y/image.shape[0], 
                            (x+rotated_digit_image.shape[1])/image.shape[1], y/image.shape[0], 
                            (x+rotated_digit_image.shape[1])/image.shape[1], (y+rotated_digit_image.shape[0])/image.shape[0], 
                            x/image.shape[1], (y+rotated_digit_image.shape[0])/image.shape[0]])
            
            # add the class index to the 
            label = [label] + bbox.tolist()
            # Save the label to a file
            with open( f'data/processed/digits_on_chute_images/{data_type}/frame_{frame_nr}.txt', 'w') as f:
                f.write(' '.join(map(str, label)))
            frame_nr += 1



def main(args: Namespace) -> None:
    """
    The main function that runs the script based on the arguments provided.

    ...

    Parameters
    ----------
    args    :   Namespace
        The arguments parsed by the ArgumentParser.
    """
    if args.mode == 'extract':
        image_extractor(video_folder=args.video_folder, 
                        hdf5_file=args.hdf5_file, 
                        dataset_name=args.dataset_name, 
                        description=args.description, 
                        overwrite_seconds=args.overwrite_seconds, 
                        save_individual_images=args.save_individual_images,
                        fbf=args.fbf)
    elif args.mode == 'validate':
        validate_image_extraction(args.hdf5_file, args.save_individual_images)
    elif args.mode == 'tree':
        print_hdf5_tree(args.hdf5_file)
    elif args.mode == 'h5_to_yolo':
        hdf5_to_yolo(args.hdf5_file, args.with_augmentation)
    elif args.mode == 'place_digits':
        place_digits_on_chute_images(args.hdf5_file)

def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('mode', type=str, choices=['extract', 'validate', 'tree', 'h5_to_yolo', 'place_digits'], help='Mode to run the script in (extracts images from videos and saves them to an hdf5 file, validate shows the difference between the original and extracted images, and tree prints the tree structure of the hdf5 file).')
    parser.add_argument('--video_folder', type=str, default='data/raw/videos', help='The folder containing the videos.')
    parser.add_argument('--hdf5_file', type=str, default='data/raw/images/images.hdf5', help='The hdf5 file to save the images to.')
    parser.add_argument('--save_individual_images', type=bool, default=False, help='Whether to save the individual images to the temp_images folder.')
    parser.add_argument('--overwrite_seconds', type=int, default=3, help='The number of seconds to wait before overwriting the hdf5 file.')
    parser.add_argument('--description', type=str, default='Dataset created for a master\'s project at Meliora Bio.', help='The description of the dataset.')
    parser.add_argument('--dataset_name', type=str, default='straw-chute', help='The name of the dataset.')
    parser.add_argument('--fbf', type=int, default=30, help='Number of frames between frames to extract.')
    parser.add_argument('--with_augmentation', type=bool, default=True, help='Whether to include augmented images in the YOLO format.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
