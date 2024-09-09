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


def decode_binary_image(image: bytes) -> np.ndarray:
    """
    Decodes a binary image and returns it as a NumPy array.
    :param image: The binary image data.
    :return: The decoded image as a NumPy array.
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
                              image_id: int = None,
                              fbf: int = 30 # frames between frames
                              ) -> list:
    print("Frame extraction started...")
    cap = cv2.VideoCapture(video_path)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    # Get the total number of frames in the video
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total = video_length // fbf, desc=f"Extracting frames from '{video_name}' ({current_video_nr}/{total_nr_videos})", position=0) # Create a progress bar
    # save the current frame as the previous frame for the next iteration
    prev_frame = None
    # A counter to keep track of the number of frames processed so far in the video -> thereby allowing for fpf (frames per frame) extraction
    count = -1
    # A counter to keep track of the number of frames saved to the HDF5 file so far -> allowing for continous numbering of the frames
    frame_count = 0
    while True:
        # Skip the first frame as it will be used as the previous frame
        if (count == 0) or (count % fbf != 0):
            count += 1
            continue
        pbar.set_postfix_str(f"RAM Usage (GB): {np.round(psutil.virtual_memory().used / 1e9, 2)}")
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames are left
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # save frame to temp_images folder
        if save_individual_images:
            cv2.imwrite(f'data/raw/temp_images/frame_{image_id}.jpg', frame)

        # Save the current frame as the previous frame for the next iteration
        prev_frame = frame    

        # Calculate the difference between consecutive frames
        frame_diff = cv2.absdiff(frame, prev_frame)
        
        # Save the frames to the HDF5 file
        success, frame = cv2.imencode('.jpg', frame)
        if success:
            success, frame_diff = cv2.imencode('.jpg', frame_diff)
            if success:
                save_frames_to_hdf5(frame, frame_diff, hdf5_file, video_id, frame_nr+frame_count)
        if save_individual_images:
            image_id += 1
        
        count += 1
        frame_count += 1
        pbar.update(1)

        if frame_count == video_length // fbf:
            break

    pbar.close() # Close the progress bar
    cap.release() # Release the video capture object
    return frame_nr + frame_count

def save_frames_to_hdf5(frame: np.ndarray,
                        frame_diff: np.ndarray,
                        hdf5_file: str,
                        video_id: str,
                        frame_nr: int):
    # print(frame_nr)
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

def image_extractor(video_folder: str, 
                    hdf5_file: str, 
                    dataset_name: str = "straw-chute", 
                    description: str = "Dataset create for a master's project at Meliora Bio.",
                    overwrite_seconds: int = 3,
                    save_individual_images: bool = False,
                    fbf: int = 24
                    ) -> None:

    video_files = os.listdir(video_folder) # Get the paths inside of the video folder

    unique_video_ids = [video_file.split('.')[0] for video_file in video_files] # Get the video ID from the name of the file

    # check if the hdf5 file already exists
    if os.path.exists(hdf5_file):
        print(f"Warning: The file {hdf5_file} already exists and will be overwritten in {overwrite_seconds} seconds.")
        for i in range(overwrite_seconds):
            print(f"Overwriting in {overwrite_seconds-i} second(s)...")
            time.sleep(1)
        # remove the file
        os.remove(hdf5_file)

    with h5py.File(hdf5_file, 'a') as hf:
        # add global features
        hf.attrs['dataset_name'] = dataset_name
        hf.attrs['description'] = description
        hf.attrs['date_created'] = np.bytes_(str(datetime.datetime.now()))

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

def validate_image_extraction(hdf5_file: str,
                              save_individual_images) -> None:
    # make sure the temp_images folder exists
    if not os.path.exists('data/raw/temp_images') and save_individual_images:
        FileNotFoundError("The temp_images folder does not exist. Please run the image_extractor function with the save_individual_images argument set to True.")

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
            
        
        if not save_individual_images:
            return
        
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
    
    :param hdf5_file: Path to the HDF5 file or a group/dataset to print.
    :param group_name: The name of the group (default is root '/').
    :param indent: The current level of indentation (used for recursive calls).
    """
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

def main(args: Namespace) -> None:
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
        
def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('mode', type=str, choices=['extract', 'validate', 'tree'], help='Mode to run the script in (extract extracts images from videos and saves them to an hdf5 file, validate shows the difference between the original and extracted images, and tree prints the tree structure of the hdf5 file).')
    parser.add_argument('--video_folder', type=str, default='data/raw/videos', help='The folder containing the videos.')
    parser.add_argument('--hdf5_file', type=str, default='data/raw/images/images.hdf5', help='The hdf5 file to save the images to.')
    parser.add_argument('--save_individual_images', type=bool, default=False, help='Whether to save the individual images to the temp_images folder.')
    parser.add_argument('--overwrite_seconds', type=int, default=3, help='The number of seconds to wait before overwriting the hdf5 file.')
    parser.add_argument('--description', type=str, default='Dataset created for a master\'s project at Meliora Bio.', help='The description of the dataset.')
    parser.add_argument('--dataset_name', type=str, default='straw-chute', help='The name of the dataset.')
    parser.add_argument('--fbf', type=int, default=30, help='Number of frames between frames to extract.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
