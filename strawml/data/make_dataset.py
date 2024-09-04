from __init__ import *
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm 
import datetime
import time
from argparse import ArgumentParser, Namespace


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
                              binary_format: bool = True, 
                              save_individual_images: bool = False, 
                              image_id: int = None) -> list:
    cap = cv2.VideoCapture(video_path)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    # List to store frames
    frames = []
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total = video_length, desc=f"Extracting frames from {video_name} ({current_video_nr}/{total_nr_videos})", position=0) # Create a progress bar

    # Read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames are left
        # save frame to temp_images folder
        if save_individual_images:
            cv2.imwrite(f'data/raw/temp_images/frame_{image_id}.jpg', frame)
        if not binary_format:
            frames.append(frame)
        else:
            success, frame = cv2.imencode('.jpg', frame)
            if success:
                frames.append(frame.tobytes())
        image_id += 1
        pbar.update(1)

    pbar.close() # Close the progress bar
    cap.release() # Release the video capture object

    # frames now contain all the frames from the video   
    return frames, image_id

def save_frames_to_hdf5(frames: list, 
                        hf,
                        video_id: str,
                        frame_nr: int):
    
    for i, image in tqdm(enumerate(frames), total=len(frames), desc=f"Saving frames to HDF5 file", position=0):
        # Save the image data as a binary dataset directly at the root
        dataset_name = f'frame_{i+frame_nr}'
        # dataset = hf.create_dataset(dataset_name, data=image)
        image = np.asarray(image)
        dataset = hf.create_dataset(dataset_name, data=image)

        # Add the video ID as an attribute to the dataset
        dataset.attrs['video ID'] = video_id
    return frame_nr + len(frames)

def image_extractor(video_folder: str, 
         hdf5_file: str, 
         dataset_name: str = "straw-chute", 
         description: str = "Dataset create for a master's project at Meliora Bio.",
         overwrite_seconds: int = 3,
         save_individual_images: bool = False
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

    hf = h5py.File(hdf5_file, 'w') # Open the HDF5 file in write mode

    # add global features
    hf.attrs['dataset_name'] = dataset_name
    hf.attrs['description'] = description
    hf.attrs['date_created'] = np.bytes_(str(datetime.datetime.now()))

    if save_individual_images:
        image_id = 0

    # Loop through the video files and extract the frames
    frame_nr = 0
    for file_nr, video_name in enumerate(video_files):
        frames, image_id = extract_frames_from_video(video_name=video_name, 
                                           video_path=os.path.join(video_folder, video_name), 
                                           current_video_nr=file_nr+1,
                                           total_nr_videos=len(video_files),
                                           save_individual_images=save_individual_images,
                                           image_id=image_id)
        # Save the frames to the HDF5 file
        frame_nr = save_frames_to_hdf5(frames, hf, unique_video_ids[file_nr], frame_nr)
    hf.close()  # close the hdf5 file

def validate_image_extraction(hdf5_file: str) -> None:
    with h5py.File(hdf5_file, 'r') as hf:
        # randomly select a frame
        frame = np.random.choice(list(hf.keys()))
        image_bytes = hf[frame][...]
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
                    indent: int = 0):
    """
    Recursively prints the structure of an HDF5 file.
    :param hdf5_file: The HDF5 file object or group to print.
    :param indent: The current level of indentation (used for recursive calls).
    """
    with h5py.File(hdf5_file, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            item = hdf5_file[key]
            
            # Print the key name (group or dataset)
            print(' ' * indent + f"├── {key}  ({'Group' if isinstance(item, h5py.Group) else 'Dataset'})")
            
            # If the item is a dataset, print its shape, dtype, and attributes
            if isinstance(item, h5py.Dataset):
                print(' ' * (indent + 4) + f"├── shape: {item.shape}")
                print(' ' * (indent + 4) + f"├── dtype: {item.dtype}")
                for attr_name, attr_value in item.attrs.items():
                    print(' ' * (indent + 4) + f"└── attrs")
                    print(' ' * (indent + 8) + f"└── {attr_name} : {attr_value}")
            
            # If the item is a group, recursively print its contents
            if isinstance(item, h5py.Group):
                print_hdf5_tree(item, indent + 4)
        
        # Print attributes of the current group (or file root)
        if isinstance(hdf5_file, h5py.Group):
            print(' ' * indent + f"└── attrs")
            for attr_name, attr_value in hdf5_file.attrs.items():
                print(' ' * (indent + 4) + f"└── {attr_name} : {attr_value}")

def main(args: Namespace) -> None:

    if args.mode == 'extract':
        image_extractor(video_folder=args.video_folder, 
                        hdf5_file=args.hdf5_file, 
                        dataset_name=args.dataset_name, 
                        description=args.description, 
                        overwrite_seconds=args.overwrite_seconds, 
                        save_individual_images=args.save_individual_images)
    elif args.mode == 'validate':
        validate_image_extraction(args.hdf5_file)
    elif args.mode == 'tree':
        print_hdf5_tree(args.hdf5_file)

def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('mode', type=str, choices=['extract', 'validate', 'tree'], help='Mode to run the script in (extract extracts images from videos and saves them to an hdf5 file, validate shows the difference between the original and extracted images, and tree prints the tree structure of the hdf5 file).')
    parser.add_argument('--video_folder', type=str, default='data/raw/videos', help='The folder containing the videos.')
    parser.add_argument('--hdf5_file', type=str, default='data/raw/images/images.hdf5', help='The hdf5 file to save the images to.')
    parser.add_argument('--save_individual_images', type=bool, default=True, help='Whether to save the individual images to the temp_images folder.')
    parser.add_argument('--overwrite_seconds', type=int, default=3, help='The number of seconds to wait before overwriting the hdf5 file.')
    parser.add_argument('--description', type=str, default='Dataset create for a master\'s project at Meliora Bio.', help='The description of the dataset.')
    parser.add_argument('--dataset_name', type=str, default='straw-chute', help='The name of the dataset.')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
