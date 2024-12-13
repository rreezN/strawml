from __init__ import *
import os
import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm 
import datetime
import time
from argparse import ArgumentParser, Namespace
import psutil
from sklearn.model_selection import train_test_split
import shutil
import json
import threading

# Create a lock object
lock = threading.Lock()

def decode_binary_image(image: bytes) -> np.ndarray:
    """

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
            frame_position = saved_frame_count * fbf + 1
            if frame_position >= video_length:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            # Read the current frame from the video
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if no more frames are left


        # save frame to temp_images folder if save_individual_images is True
        if save_individual_images:
            cv2.imwrite(f'data/calibration/images/frame_{image_id}.jpg', frame)

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
                 sizes: list[float] = [0.8,0.1,0.1],
                 annotation_types = ['bbox_chute', 'bbox_straw', 'bbox_straw', 'whole']
) -> None:
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

    with h5py.File(hdf5_file, 'r') as hf:
        frame_names = list(np.arange(hf.__len__()))
        # check if 'bbox_straw' is in the annotations
        hf_annotations = list(hf['frame_0']['annotations'].keys())
        anno_set = set(annotation_types) - set(['whole'])
        if not anno_set.issubset(hf_annotations):
            raise ValueError(f"The HDF5 file does not contain the annotation types {annotation_types}.")
        
    save_path = 'data/processed/yolo_format_' + "_".join(annotation_types)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        # remove the folder and create a new one
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    # split frames in train, test and validation with 80%, 10% and 10% respectively
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
            save_path = 'data/processed/yolo_format_' + "_".join(annotation_types)
            pbar.set_postfix_str(f"Saving to ..._{'_'.join(annotation_types)}/{key}")
            if not os.path.exists(f'{save_path}/{key}'):
                os.makedirs(f'{save_path}/{key}')
            for frame in val:
                labels = []
                image_bytes = hf[f"frame_{frame}"]['image'][...]
                # decode the binary image
                image = decode_binary_image(image_bytes)
                ############################
                # NOTE CASE 1: bbox_straw only, use the bbox_chute to crop the image and correct the bbox_straw coordinates
                ############################
                if annotation_types == ['bbox_straw']:
                    desired_res = (180, 1350)
                    # crop the image to the chute area
                    x1, y1, x2, y2, x3, y3, x4, y4 = hf[f"frame_{frame}"]['annotations']["bbox_chute"][...]
                    pts_src = np.array([[x4, y4], [x1, y1], [x3, y3], [x2, y2]], dtype='float32')

                    # Ensure the output is a rectangular
                    width = int(max(abs(x2 - x1), abs(x4 - x3)))  # Largest horizontal difference
                    height = int(max(abs(y3 - y1), abs(y4 - y2))) # Largest vertical difference
                    # Define the destination points for the square output
                    pts_dst = np.array([[0, 0], 
                                        [width - 1, 0], 
                                        [0, height - 1], 
                                        [width - 1, height - 1]]
                                        , dtype='float32')
                    # Compute the perspective transformation matrix
                    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                    # Apply the perspective warp to create a square cutout
                    cutout = cv2.warpPerspective(image, M, (width, height))
                    # Resize the cutout to the desired resolution
                    cutout_resized = cv2.resize(cutout, desired_res)
                    # Save the resized cutout
                    cv2.imwrite(f'{save_path}/{key}/{frame}.jpg', cutout_resized)

                    # Now we load the respective bounding boxes
                    straw_bbox = hf[f"frame_{frame}"]['annotations']['bbox_straw'][...]
                    # If the bbox_straw is empty, meaning the chute is empty, we dont save a *.txt file
                    if len(straw_bbox) == 0:
                        # create empty txt file
                        with open( f'{save_path}/{key}/{frame}.txt', 'w') as f:
                            pass
                        pbar.update(1)
                        continue
                    straw_bbox = straw_bbox.reshape(-1, 1, 2).astype('float32')
                    straw_bbox = cv2.perspectiveTransform(straw_bbox, M).flatten()
                    # Scale the straw_bbox to match the resized cutout
                    scale_x = desired_res[0] / width
                    scale_y = desired_res[1] / height
                    straw_bbox[::2] *= scale_x
                    straw_bbox[1::2] *= scale_y
                    for i in range(len(straw_bbox)):
                        straw_bbox[i] /= cutout_resized.shape[1] if i % 2 == 0 else cutout_resized.shape[0]
                    label = np.insert(straw_bbox, 0, 0)
                    # Save the label to a file
                    with open( f'{save_path}/{key}/{frame}.txt', 'w') as f:
                        f.write(' '.join(map(str, label)) + "\n")
                ############################
                # NOTE CASE 2: bbox_straw but with the whole image
                ############################
                elif annotation_types[-1] == 'whole':
                    # Save the image to a file
                    cv2.imwrite(f'{save_path}/{key}/{frame}.jpg', image)
                    bbox = hf[f"frame_{frame}"]['annotations'][annotation_types[0]][...]
                    if len(bbox) == 0:
                        continue
                    # normalize the bounding box coordinates to values between 0 and 1 
                    for j in range(len(bbox)):
                        bbox[j] /= image.shape[1] if j % 2 == 0 else image.shape[0]
                    # add the class index to the label
                    labels += [np.insert(bbox, 0, 0)]
                    # Save the label to a file
                    with open( f'{save_path}/{key}/{frame}.txt', 'w') as f:
                        for label in labels:
                            f.write(' '.join(map(str, label)) + "\n")
                ############################
                # NOTE CASE 3: Either bbox_chute only or both labels are saved
                ############################
                else:
                    # Save the image to a file
                    cv2.imwrite(f'{save_path}/{key}/{frame}.jpg', image)
                    for i, annotation_type in enumerate(annotation_types):
                        # Now we load the respective bounding boxes
                        bbox = hf[f"frame_{frame}"]['annotations'][annotation_type][...]
                        if len(bbox) == 0:
                            continue
                        # normalize the bounding box coordinates to values between 0 and 1 
                        for j in range(len(bbox)):
                            bbox[j] /= image.shape[1] if j % 2 == 0 else image.shape[0]
                        # add the class index to the label
                        labels += [np.insert(bbox, 0, i)]
                    # Save the label to a file
                    with open( f'{save_path}/{key}/{frame}.txt', 'w') as f:
                        for label in labels:
                            f.write(' '.join(map(str, label)) + "\n")
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

def place_digits_on_chute_images() -> None:
    """
    HWD+ Data: https://link.springer.com/article/10.1007/s42979-022-01494-2#Sec5
               https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or

    """
    from strawml.data.image_utils import overlay_image, SpecialRotate, create_random_permutations_with_repeats, internal_image_operations
    from strawml.data.augment_chute_data import color_image
    from skimage import exposure

    # digit_images = np.load("D:/HCAI/msc/strawml/data/interim/digits/Images(500x500).npy")
    digit_images = np.load("data/interim/digits/Images(28x28).npy")
    digit_info = np.load("data/interim/digits/WriterInfo.npy")

    # We sort the digit images by the digit labels
    digit_images = digit_images[np.argsort(digit_info[:,0])]
    digit_info = digit_info[np.argsort(digit_info[:,0])]
    digit_labels = digit_info[:,0]
    frame_nr = 0
    X_train, X_val, y_train, y_val = train_test_split(digit_images, digit_labels, test_size=0.2, random_state=42, stratify=digit_labels)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify=y_val)

    data_dict = {'train': X_train, 'val': X_val, 'test': X_test}
    label_dict = {'train': y_train, 'val': y_val, 'test': y_test}
    total_permutations = {'train': 8000, 'val': 1000, 'test': 1000}
    # let us create a suitable background image for the digits by loading a chute image and taking image[:1440, :1250]
    img = cv2.imread("data/processed/chute_image_for_backgroud_creation.jpg")[:1440, :1250]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for data_type in data_dict.keys():
        # create data_type folder if it does not exist
        if not os.path.exists(f'data/processed/digits_on_chute_images/{data_type}'):
            os.makedirs(f'data/processed/digits_on_chute_images/{data_type}')
        else:
            shutil.rmtree(f'data/processed/digits_on_chute_images/{data_type}')
            os.makedirs(f'data/processed/digits_on_chute_images/{data_type}')

        d = data_dict[data_type]
        l = label_dict[data_type]

        permutations = create_random_permutations_with_repeats(data=d, labels=l, total_permutations=total_permutations[data_type], min_size=3, max_size=5)

        for digits, labels in tqdm(permutations, total=len(data_dict[data_type]), desc=f"Placing digits on chute images ({data_type})"):
            label_ = []
            image = img.copy()
            # perform random operation on the image
            image = internal_image_operations(image)
            # perform random gamma operation on the and add noise
            if np.random.uniform() > 0.5:
                gamma = np.random.uniform(0.5, 2.5)
                image = exposure.adjust_gamma(image, gamma)
                gaussian_noise = np.random.uniform(0, 1)
                image = cv2.add(image, np.random.normal(0, gaussian_noise, image.shape).astype(np.uint8))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            for digit, label in zip(digits, labels):

                # To ensure that when rotating the background stays the same color
                digit_image = digit.astype(np.uint8)
                # We change the white background to a random off-white color to make the digits stand out
                # the complete white color might be too specific of a color and thus when it comes to finding the digits in the chute image, it might be difficult
                digit_image[digit_image == 255] = np.random.randint(200, 256)
                # draw a hyphen on the digit image to replicate the number images it is supposed to find in the chute image
                cv2.line(digit_image, (22, 14), (27, 14), (0, 0, 0), 2)
                # randomly resize the image to be between 14x14 and 28x28
                digit_image = cv2.resize(digit_image, (np.random.randint(14, 29), np.random.randint(14, 29)))
                digit_image_width, digit_image_height = digit_image.shape
                # sample a random angle between -20 and 20 degrees
                angle = np.random.randint(-20, 21)
                # rotate the digit image
                rotated_digit_image, default_rot = SpecialRotate(digit_image, angle)
                # expand the digit image to 3 channels
                rotated_digit_image = cv2.cvtColor(rotated_digit_image, cv2.COLOR_RGB2RGBA)
                # add the digit image to the chute image
                image, bbox = overlay_image(image, rotated_digit_image, default_rot, (digit_image_width, digit_image_height))
                # draw the bounding box on the image
                # x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                # x1, y1, x2, y2, x3, y3, x4, y4 = int(x1*image.shape[1]), int(y1*image.shape[0]), int(x2*image.shape[1]), int(y2*image.shape[0]), int(x3*image.shape[1]), int(y3*image.shape[0]), int(x4*image.shape[1]), int(y4*image.shape[0])
                # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.line(image, (x2, y2), (x3, y3), (255, 0, 0), 2)
                # cv2.line(image, (x3, y3), (x4, y4), (255, 0, 0), 2)
                # cv2.line(image, (x4, y4), (x1, y1), (255, 0, 0), 2)
                label_ += [[0] + bbox.tolist()]
            # save the image to a file
            # write image to rgb format
            cv2.imwrite(f'data/processed/digits_on_chute_images/{data_type}/frame_{frame_nr}.jpg', cv2.cvtColor(image, cv2.COLOR_RGBA2BGR))

            # get the bounding box coordinates of the digit image on the chute image, x1, y1, x2, y2, x3, y3, x4, y4        
            # add the class index to the 
            # Save the label to a file
            with open( f'data/processed/digits_on_chute_images/{data_type}/frame_{frame_nr}.txt', 'w') as f:
                for inner_label in label_:
                    # Write each inner list to the file
                    f.write(' '.join(map(str, inner_label)) + '\n')
            frame_nr += 1

def prepare_data_for_timm(hdf5_file: str, sizes: list[float] = [0.8,0.1,0.1]) -> None:
    """
    This function prepares the data for the timm library by creating a folder structure similar to ImageNet.
    The folder structure is as follows:
    - data
        - train
            - class_0
            - class_1
            - ...
        - val
            - class_0
            - class_1
            - ...
        - test
            - class_0
            - class_1
            - ...
    """ 
    
    # First ensure the file exists
    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"The file {hdf5_file} does not exist.")
    
    save_path = 'data/processed/timm_format'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        # remove the folder and create a new one
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    
    frames = h5py.File(hdf5_file, 'r')
    frame_names = list(frames.keys())
    
    # split frames in train, test and validation with 70%, 20% and 10% respectively
    # random state = 42
    train_size, test_size, val_size = sizes
    train_indices, test_indices, _, _ = train_test_split(frame_names, frame_names, test_size=test_size+val_size, random_state=42)
    test_indices, val_indices, _, _ = train_test_split(test_indices, test_indices, test_size=test_size/(test_size+val_size), random_state=42)
    # create a dictionary to store the indices
    index_dict = {'train': train_indices, 'test': test_indices, 'val': val_indices}
    
    with h5py.File(hdf5_file, 'r') as hf:        
        # create a progress bar
        pbar = tqdm(total = len(frame_names), desc="Converting HDF5 to timm format", position=0)
        
        # loop through the datasets and save the images and labels
        for key, val in index_dict.items():
            pbar.set_description(f"Converting HDF5 to timm format: {key} set")
            if not os.path.exists(f'{save_path}/{key}'):
                os.makedirs(f'{save_path}/{key}')
            for frame in val:
                image_bytes = hf[frame]['image'][...]
                # decode the binary image
                image = decode_binary_image(image_bytes)
                # Find the fullness (label)
                label = hf[frame]['annotations']['fullness'][...]
                # Save the image to a file
                if not os.path.exists(f'{save_path}/{key}/{label}'):
                    os.makedirs(f'{save_path}/{key}/{label}')
                cv2.imwrite(f'{save_path}/{key}/{label}/{frame}.jpg', image)
                pbar.update(1)


def make_sensor_data(hdf5_file_path: str, sensor_file_path: str) -> None:
    # First ensure the file exists
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f"The file {hdf5_file_path} does not exist.")

    if not os.path.exists(sensor_file_path):
        raise FileNotFoundError(f"The file {sensor_file_path} does not exist.")
    
    save_path = 'data/raw/sensor'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Load the sensor data from the sensor file
    sensor_df = pd.read_excel(sensor_file_path)
    
    # Remove the first three rows and first column
    sensor_df = sensor_df.iloc[3:, 1:]
    
    timestamp_col = sensor_df.columns[0]
    sensor_col = sensor_df.columns[1]
    sensor_data = sensor_df[sensor_col].values
    time_stamp = sensor_df[timestamp_col].values
    
    # convert the time stamp to a datetime object
    time_stamp = pd.to_datetime(time_stamp)
    
    # if len(sensor_data) != len(frame_names):
    #     raise ValueError(f"The number of frames in the HDF5 file ({len(frame_names)}) does not match the number of sensor data entries ({len(sensor_data)}).")
    
    sensor_hdf5_file = 'data/raw/sensor/sensor_data.hdf5'
    # create a new hdf5 file to store the images and annotations from sensor data
    if os.path.exists(sensor_hdf5_file):
        os.remove(sensor_hdf5_file)
    
    new_hf = h5py.File(sensor_hdf5_file, 'a') # Open the HDF5 file in write mode
    old_hf = h5py.File(hdf5_file_path, 'r') # Open the original HDF5 file in read mode
    
    # Copy the attributes from the original HDF5 file to the new HDF5 file
    new_hf.attrs['dataset_name'] = old_hf.attrs['dataset_name']
    new_hf.attrs['description'] = old_hf.attrs['description']
    new_hf.attrs['date_created'] = np.bytes_(str(datetime.datetime.now()))
    
    # Sort the frame_names by the frame number
    frame_names = list(old_hf.keys())
    frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1]))
    
    pbar = tqdm(total = min(len(sensor_data), len(old_hf.keys())), desc="Adding sensor data to the HDF5 file", position=0)
    for i, frame in enumerate(frame_names):
        if i >= len(sensor_data):
            break
        pbar.set_description(f"Adding sensor data to the HDF5 file: {frame}")
        group = new_hf.create_group(frame)
        # Copy the original image and image_diff to the new HDF5 file
        old_hf.copy(old_hf[frame]['image'], group)
        old_hf.copy(old_hf[frame]['image_diff'], group) 
        group.attrs['video ID'] = old_hf[frame].attrs['video ID']
        
        # Create a new group for the annotations
        annotation_group = group.create_group('annotations')
        
        # annotate the sensor data to the frame
        fullness = sensor_data[i]/100
        annotation_group.create_dataset('sensor_fullness', data=fullness)
        
        # print(f"{new_hf[frame].keys(), new_hf[frame]['annotations']['sensor_fullness'][...]}")
        
        pbar.update(1)
    
    old_hf.close()  # close the original hdf5 file
    new_hf.close()  # close the hdf5 file
    

def validate_sensor_data(hdf5_file: str = 'data/processed/sensor_data/sensor_data.hdf5') -> None:
    import matplotlib.pyplot as plt
    hf = h5py.File(hdf5_file, 'r')
    
    # Sort the frame_names by the frame number
    frame_names = list(hf.keys())
    frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1]))
    
    for frame in frame_names:
        print(f"Frame: {frame}")
        image = decode_binary_image(hf[frame]['image'][...])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fullness = hf[frame]['annotations']['sensor_fullness'][...]
        plt.imshow(image)
        plt.title(f"{frame} Fullness: {fullness}")
        plt.show()



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
        hdf5_to_yolo(hdf5_file=args.hdf5_file, annotation_types = ['bbox_straw', 'whole'])
    elif args.mode == 'place_digits':
        place_digits_on_chute_images()
    elif args.mode == 'timm':
        prepare_data_for_timm(args.hdf5_file)
    elif args.mode == 'sensor':
        make_sensor_data(args.hdf5_file, args.sensor_file)
    elif args.mode == 'validate_sensor':
        validate_sensor_data(args.hdf5_file)

def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('mode', type=str, choices=['extract', 'validate', 'tree', 'h5_to_yolo', 'place_digits', 'timm', 'sensor', 'validate_sensor'], help='Mode to run the script in (extracts images from videos and saves them to an hdf5 file, validate shows the difference between the original and extracted images, and tree prints the tree structure of the hdf5 file).')
    parser.add_argument('--video_folder', type=str, default='data/raw/videos', help='The folder containing the videos.')
    parser.add_argument('--hdf5_file', type=str, default='data/raw/images/images.hdf5', help='The hdf5 file to save the images to.')
    parser.add_argument('--sensor_file', type=str, default='data/raw/sensor/studentexport.xlsx', help='The sensor data file to save the sensor data to.')
    parser.add_argument('--save_individual_images', type=bool, default=False, help='Whether to save the individual images to the temp_images folder.')
    parser.add_argument('--overwrite_seconds', type=int, default=3, help='The number of seconds to wait before overwriting the hdf5 file.')
    parser.add_argument('--description', type=str, default='Dataset created for a master\'s project at Meliora Bio.', help='The description of the dataset.')
    parser.add_argument('--dataset_name', type=str, default='straw-chute', help='The name of the dataset.')
    parser.add_argument('--fbf', type=int, default=30, help='Number of frames between frames to extract.')
    parser.add_argument('--with_augmentation',  type=bool, default=True, help='Whether to include augmented images in the YOLO format.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
