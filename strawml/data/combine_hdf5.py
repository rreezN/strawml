import h5py
import os
from argparse import ArgumentParser, Namespace

def combine_hdf5(data_path: str, file1: str, file2: str, output_file:str, force=True):
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
        if force:
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

    with h5py.File(data_path + file, 'r') as f:
        for group in f:
            # ensure that each group has 'annotations', 'image', and 'image_diff' datasets
            if 'annotations' not in f[group]:
                raise ValueError(f'{file} is missing the "annotations" dataset in the "{group}" group.')
            if 'image' not in f[group]:
                raise ValueError(f'{file} is missing the "image" dataset in the "{group}" group.')
            if 'image_diff' not in f[group]:
                raise ValueError(f'{file} is missing the "image_diff" dataset in the "{group}" group.')
            
            # Check if image actually contains an image
            if f[group]['image'].shape[0] == 0:
                raise ValueError(f'The "image" dataset in the "{group}" group is empty.')
            if f[group]['image_diff'].shape[0] == 0:
                raise ValueError(f'The "image_diff" dataset in the "{group}" group is empty.')
            
            # ensure that 'annotations' has a fullness score 
            if 'fullness' not in f[group]['annotations']:
                raise ValueError(f'The "annotations" dataset in the "{group}" group is missing the "fullness" attribute.')
    print(f'{file} is a valid dataset, i.e. no missing values.')

def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('mode', type=str, choices=['combine', 'validate'], help='Mode to run the script in (extracts images from videos and saves them to an hdf5 file, validate shows the difference between the original and extracted images, and tree prints the tree structure of the hdf5 file).')
    parser.add_argument('--data_path', type=str, default='D:/HCAI/msc/strawml/data/interim/', help='The folder containing the files.')
    parser.add_argument('--file1', type=str, help='The first file to combine.')
    parser.add_argument('--file2', type=str, help='The second file to combine.')
    parser.add_argument('--output_file', type=str, default='chute_detection_combined.hdf5', help='The name of the output file.')
    parser.add_argument('--force', type=bool, default=True, help='Whether to force the combination of the files.')
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
