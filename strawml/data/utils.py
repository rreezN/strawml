import h5py
import os


def combine_hdf5(file1: str, file2: str):
    """
    Combines the contents of two HDF5 files into a new file.

    ...

    Parameters
    ----------
    file1 : str
        The path to the first HDF5 file.
    file2 : str
        The path to the second HDF5 file.

    Raises
    ------
    ValueError
        If either file does not exist or if the files are not in HDF5 format.
    """
    if not os.path.exists(file1):
        raise ValueError(f'{file1} does not exist.')
    if not os.path.exists(file2):
        raise ValueError(f'{file2} does not exist.')
    if not file1.endswith('.hdf5') or not file2.endswith('.hdf5'):
        raise ValueError('Files must be in HDF5 format.')

    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        # Create a new file
        with h5py.File('combined.hdf5', 'w') as f_combined:
            # Copy the contents of the first file
            for name in f1:
                f1.copy(name, f_combined)
            
            # Append the contents of the second file
            for name in f2:
                if name in f_combined:
                    # Assuming datasets are 1D and can be concatenated
                    data1 = f_combined[name][:]
                    data2 = f2[name][:]
                    combined_data = np.concatenate((data1, data2))
                    del f_combined[name]  # Delete the existing dataset
                    f_combined.create_dataset(name, data=combined_data)
                else:
                    f2.copy(name, f_combined)
            
            print('Files combined successfully!')