from __init__ import *
import h5py
import os
from argparse import ArgumentParser, Namespace
import cv2
from tqdm import tqdm
from strawml.data.make_dataset import decode_binary_image

def combine_hdf5(data_path: str, file1: str, file2: str, output_file:str):
    """
    Combines the contents of two HDF5 files into a new file.

    ...

    Parameters
    ----------
    data_path : str
        The path to the directory containing the HDF5 files.
    file1 : str
        The path to the first HDF5 file.
    file2 : str
        The path to the second HDF5 file.
    output_file : str
        The name of the new HDF5 file that will contain the combined contents.

    Raises
    ------
    ValueError
        If either file does not exist or if the files are not in HDF5 format.
    """
    if not os.path.exists(data_path + file1):
        raise ValueError(f'{data_path + file1} does not exist.')
    if not os.path.exists(data_path + file2):
        raise ValueError(f'{data_path + file2} does not exist.')
    if not file1.endswith('.hdf5') or not file2.endswith('.hdf5'):
        raise ValueError('Files must be in HDF5 format.')

    with h5py.File(data_path + file1, 'r') as f1:
        # we find the highets index name in f1 and add to the names of f2
        # we then add the contents of f2 to f1
        # we then save f1 to the output file
        k = list(f1.keys()) 
        values = [int(key.split(".")[0].split("_")[-1]) for key in k]
        max_value = max(values)
        # Change the name of the keys in f2
        with h5py.File(data_path + file2, 'r') as f2:
            with h5py.File('temp.hdf5', 'w') as f_new:
                for name in list(f2.keys()):
                    frame_nr = int(name.split(".")[0].split("_")[-1]) + max_value
                    new_name = f"frame_{frame_nr}"
                    # Copy the dataset to the new file with the new key
                    f2.copy(name, f_new, new_name)
        # Delete the old file and rename the new file
        with h5py.File("temp.hdf5", 'r') as f2:
            # Create a new file
            with h5py.File(data_path + output_file, 'w') as f_combined:
                # Copy the contents of the first file
                for name in f1:
                    f1.copy(name, f_combined)
                
                # Append the contents of the second file
                for name in f2:
                    if name in f_combined:
                        print(name)
                        continue
                    else:
                        f2.copy(name, f_combined)
        print('Files combined successfully!')
        os.remove('temp.hdf5')
    check_validity(data_path, output_file)

def check_validity(data_path: str, file: str):
    """
    Checks the validity of an HDF5 file.

    ...

    Parameters
    ----------
    data_path : str
        The path to the directory containing the HDF5 file.
    file : str
        The path to the HDF5 file.

    Raises
    ------
    ValueError
        If the file does not exist or if the file is not in HDF5 format.
    """
    if not os.path.exists(data_path + file):
        raise ValueError(f'{data_path + file} does not exist.')
    if not file.endswith('.hdf5'):
        raise ValueError('File must be in HDF5 format.')
    with h5py.File(data_path + file, 'r') as f:
        print(f'{file} is a valid HDF5 file.')

    errored = False
    with h5py.File(data_path + file, 'r') as f:
        for group in f:
            # ensure that each group has 'annotations', 'image', and 'image_diff' datasets
            if 'annotations' not in f[group]:
                print(f'{file} is missing the "annotations" dataset in the "{group}" group.')
                errored = True
            if 'image' not in f[group]:
                print(f'{file} is missing the "image" dataset in the "{group}" group.')
                errored = True
            if 'image_diff' not in f[group]:
                print(f'{file} is missing the "image_diff" dataset in the "{group}" group.')
                errored = True
            
            # Check if image actually contains an image
            if f[group]['image'].shape[0] == 0:
                print(f'The "image" dataset in the "{group}" group is empty.')
                errored = True
            if f[group]['image_diff'].shape[0] == 0:
                print(f'The "image_diff" dataset in the "{group}" group is empty.')
                errored = True
            
            # ensure that 'annotations' has a fullness score 
            if 'fullness' not in f[group]['annotations'] and not 'straw' in file:
                print(f'The "annotations" dataset in the "{group}" group is missing the "fullness" attribute.')
                errored = True
            
    if errored:
        print(f'{file} is not a valid dataset, i.e. missing values.')
    else:
        print(f'{file} is a valid dataset, i.e. no missing values.')

def combine_and_correct_hdf5(data_path: str, 
                             file1: str, 
                             file2: str, 
                             annotations_to_merge: list = ['bbox_chute', 'bbox_straw'], 
                             desired_resolution: tuple = (2560, 1440)):
    frame_nrs = []
    nr_of_resized_images = 0
    case1, case2, case3, case4 = [], [], [], []
    with h5py.File(data_path + file1, 'r+') as f1:
        with h5py.File(data_path + file2, 'r') as f2:
            go_to_outer_loop = False
            pbar = tqdm(f1)
            for frame_nr in pbar:
                # first check if the frame_nr exists in f2
                if not frame_nr in f2:
                    tqdm.write(f"[!!!] Warning: {frame_nr} not found in {file2}. Can't merge annotations.")
                    tqdm.write(" --- Dropping frame...")
                    del f1[frame_nr]
                    continue
                tqdm.write(f"\nCURRENT: {frame_nr}")
                frame_nrs.append(int(frame_nr.split(".")[0].split("_")[-1]))
                ## FIRST: Add annotations from f2 to f1
                # Get the annotations of the current frame for each file
                f1_annotations = list(f1[frame_nr]['annotations'].keys())
                f2_annotations = list(f2[frame_nr]['annotations'].keys())
                # Loop through the annotations to merge
                for annotation in annotations_to_merge:
                    # Check if the annotation is missing from f1 
                    if annotation not in f1_annotations:
                        # Check if the annotation is present in f2
                        if annotation in f2_annotations:
                            # then copy the annotation from f2 to f1
                            tqdm.write(f"--- Copying {annotation}")
                            f1[frame_nr]['annotations'].create_dataset(annotation, data=f2[frame_nr]['annotations'][annotation][()])
                            case1 += [frame_nr]
                        else: # If the annotation is missing from both files, print a warning
                            tqdm.write(f"[!!!] Warning: {annotation} not found in {file1} and {file2}.")
                            tqdm.write(f" ---  Dropping {frame_nr}")
                            del f1[frame_nr]
                            go_to_outer_loop = True
                            case2 += [frame_nr]
                            break
                    else: # Check if the annotation exists in both files
                        # If the annotation is empty in f1 but not in f2, copy the annotation from f2 to f1
                        if len(f1[frame_nr]['annotations'][annotation][()]) == 0:
                            if len(f2[frame_nr]['annotations'][annotation][()]) != 0:
                                tqdm.write(f" --- Copying {annotation} from {file2} to {file1}")
                                # drop the old annotation and replace with the new one
                                del f1[frame_nr]['annotations'][annotation]
                                f1[frame_nr]['annotations'].create_dataset(annotation, data=f2[frame_nr]['annotations'][annotation][()])
                                case3 += [frame_nr]
                            else: # If the annotation is empty in both files, print a warning and drop the frame
                                tqdm.write(f"[!!!] Warning: {annotation} empty in {file1} and {file2}. Ignore if this is expected e.g. empty chute for straw level, then the straw_bbox is expected to be an empty list.")
                                case4 += [frame_nr]
                if go_to_outer_loop:
                    go_to_outer_loop = False
                    continue
                image = decode_binary_image(f1[frame_nr]['image'][()])            
                if image.shape[0] != desired_resolution[1] or image.shape[1] != desired_resolution[0]:
                    tqdm.write(f'[!!!] Warning: The resolution of the image "{frame_nr}" is incorrect. Performing resize...')
                    image_diff = decode_binary_image(f1[frame_nr]['image_diff'][()])
                    image = cv2.resize(image, desired_resolution)
                    image_diff = cv2.resize(image_diff, desired_resolution)
                    # drop the old image and image_diff and replace with the new ones
                    del f1[frame_nr]['image']
                    del f1[frame_nr]['image_diff']
                    f1[frame_nr].create_dataset('image', data=cv2.imencode('.jpg', image)[1])
                    f1[frame_nr].create_dataset('image_diff', data=cv2.imencode('.jpg', image_diff)[1])
                    bbox_chute, bbox_straw = fix_bbox(image.shape[:2], desired_resolution, f1[frame_nr]['annotations']['bbox_chute'][()], f1[frame_nr]['annotations']['bbox_straw'][()])
                    try:
                        del f1[frame_nr]['annotations']['bbox_chute']
                        f1[frame_nr]['annotations'].create_dataset('bbox_chute', data=bbox_chute)
                    except KeyError:
                        tqdm.write(f'No bbox_chute in {frame_nr}')
                    if bbox_straw != []:
                        try:
                            del f1[frame_nr]['annotations']['bbox_straw']
                            f1[frame_nr]['annotations'].create_dataset('bbox_straw', data=bbox_straw)
                        except KeyError:
                            tqdm.write(f'No bbox_straw in {frame_nr}')
                    tqdm.write(f'Resized the image "{frame_nr}" to the correct resolution.')
                    nr_of_resized_images += 1
                
    frame_nrs = sorted(frame_nrs)
    missing_frames = [i for i in range(frame_nrs[0], frame_nrs[-1]+1) if i not in frame_nrs]
    if missing_frames:
        print(f'\nMissing frames: {missing_frames}\n')
    else:
        print('All frames are present.')
    print(f'Number of resized images: {nr_of_resized_images}')
    print(f'Direct copy from f2 to f1 (f1 has no annotation): {len(case1)}')
    print(f'Missing annotation in both files: {len(case2)}')
    print(f'Copy to f1 from f2 (f1 has empty value) {len(case3)}')
    print(f'Both files have empty value for the annotation: {len(case4)}')


def check_missing_frames(annotated_file: str, original_file: str):
    if not os.path.exists(annotated_file):
        raise ValueError(f'{annotated_file} does not exist.')
    if not annotated_file.endswith('.hdf5'):
        raise ValueError('Annotated File must be in HDF5 format.')
    if not os.path.exists(original_file):
        raise ValueError(f'{original_file} does not exist.')
    if not original_file.endswith('.hdf5'):
        raise ValueError('Original File must be in HDF5 format.')

    with h5py.File(annotated_file, 'r') as anno:
        with h5py.File(original_file, 'r') as original:
            for group in original:
                if group not in anno:
                    print(f'{group} is missing in {annotated_file}.')
    
    print("Check complete. If no output, then all frames are present.")

def correct_hdf5(data_path: str, file: str, resolution: tuple = (2560, 1440)):
    """
    File that ensures that the HDF5 file is correct, meaning we account for
        - resolution of the image
        - Missing images i.e. frame_nr is not continuous from 0 to n, then report the missing frames

    Parameters
    ----------
    data_path : str
        The path to the directory containing the HDF5 file.
    file : str
        The hdf5 filename
    resolution : tuple
        The resolution of the images in the hdf5 file, i.e. (width, height) e.g. (1920, 1080) or (2560, 1440)
    """
    frame_nrs = []
    nr_of_resized_images = 0
    with h5py.File(data_path + file, 'r+') as f:
        pbar = tqdm(f)
        pbar.set_postfix_str(f'resized images: {nr_of_resized_images}')
        for group in pbar:
            frame_nrs.append(int(group.split(".")[0].split("_")[-1]))
            image = decode_binary_image(f[group]['image'][()])            
            if image.shape[0] != resolution[1] or image.shape[1] != resolution[0]:
                print(f'The resolution of the image "{group}" is incorrect. Performing resize...')
                image_diff = decode_binary_image(f[group]['image_diff'][()])
                image = cv2.resize(image, resolution)
                image_diff = cv2.resize(image_diff, resolution)
                # drop the old image and image_diff and replace with the new ones
                del f[group]['image']
                del f[group]['image_diff']
                f[group].create_dataset('image', data=cv2.imencode('.jpg', image)[1])
                f[group].create_dataset('image_diff', data=cv2.imencode('.jpg', image_diff)[1])
                bbox_chute, bbox_straw = fix_bbox(image.shape[:2], resolution, f[group]['annotations']['bbox_chute'][()], f[group]['annotations']['bbox_straw'][()])
                
                try:
                    del f[group]['annotations']['bbox_chute']
                    f[group]['annotations'].create_dataset('bbox_chute', data=bbox_chute)
                except KeyError:
                    print(f'No bbox_chute in {group}')
                
                try:
                    del f[group]['annotations']['bbox_straw']
                    f[group]['annotations'].create_dataset('bbox_straw', data=bbox_straw)
                except KeyError:
                    print(f'No bbox_straw in {group}')
                print(f'Resized the image "{group}" to the correct resolution.')
                nr_of_resized_images += 1
                pbar.set_postfix_str(f'resized images: {nr_of_resized_images}')

    frame_nrs = sorted(frame_nrs)
    missing_frames = [i for i in range(frame_nrs[0], frame_nrs[-1]+1) if i not in frame_nrs]
    if missing_frames:
        print(f'Missing frames: {missing_frames}')
    else:
        print('All frames are present.')
    print(f'Number of resized images: {nr_of_resized_images}')

def fix_bbox(old_resolution: tuple, new_resolution: tuple, bbox_chute: list, bbox_straw: list):
    """
    Scales the bounding boxes to the new resolution, based on the old resolution.

    Parameters
    ----------
    old_resolution : tuple
        The resolution of the images in the hdf5 file, i.e. (height, width) e.g. (1080, 1920) or (1440, 2560)
    new_resolution : tuple
        The resolution of the images in the hdf5 file, i.e. (width, height) e.g. (1920, 1080) or (2560, 1440)
    bbox_chute : list
        The bounding box coordinates for the chute. The format is [x1, y1, x2, y2, x3, y3, x4, y4].
    bbox_straw : list
        The bounding box coordinates for the straw. The format is [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns
    -------

    """
    x_scale = new_resolution[0] / old_resolution[1]
    y_scale = new_resolution[1] / old_resolution[0]
    
    bbox_chute = [int(val * x_scale) if i % 2 == 0 else int(val * y_scale) for i, val in enumerate(bbox_chute)]
    bbox_straw = [int(val * x_scale) if i % 2 == 0 else int(val * y_scale) for i, val in enumerate(bbox_straw)]
    
    return bbox_chute, bbox_straw


def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('mode', type=str, choices=['combine', 'validate', 'annotate_combine', 'check_missing'], help='Mode to run the script in (extracts images from videos and saves them to an hdf5 file, validate shows the difference between the original and extracted images, and tree prints the tree structure of the hdf5 file).')
    parser.add_argument('--data_path', type=str, default='D:/HCAI/msc/strawml/data/interim/', help='The folder containing the files.')
    parser.add_argument('--file1', type=str, help='The first file to combine.')
    parser.add_argument('--file2', type=str, help='The second file to combine.')
    parser.add_argument('--output_file', type=str, default='chute_detection_combined.hdf5', help='The name of the output file.')
    parser.add_argument('-force', action="store_true", help='Whether to force the combination of the files.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    data_path = args.data_path
    file1 = args.file1
    file2 = args.file2
    output_file = args.output_file
    force = args.force
    if args.mode == 'validate':
        check_validity(data_path, file1)
    elif args.mode == 'combine':
        combine_hdf5(data_path, file1, file2, output_file, force)
    elif args.mode == 'annotate_combine':
        combine_and_correct_hdf5(data_path, file1, file2, annotations_to_merge=['bbox_chute', 'bbox_straw'], desired_resolution=(2560, 1440))
    elif args.mode == 'check_missing':
        check_missing_frames(file1, file2)
