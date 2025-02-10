import h5py
from tqdm import tqdm


def combine(file1_path, file2_path, output_file_path):
    chute_file = h5py.File(file1_path, 'r')
    straw_file = h5py.File(file2_path, 'r')
    
    new_file = h5py.File(output_file_path, 'w')
    
    # Copy entire structure of first file to new file
    for key in tqdm(chute_file.keys()):
        chute_file.copy(key, new_file)
        
    # Copy all datasets from second file to new file
    for key in tqdm(straw_file.keys()):
        straw_file.copy(key, new_file)
        
    chute_file.close()
    straw_file.close()
    new_file.close()

def check_file(file_path):
    f = h5py.File(file_path, 'r')
    print(f'Total keys: {len(f.keys())}')
    f.close()


if __name__ == "__main__": 
    file_1 = 'data/processed/recording_rotated_all_frames_processed_combined.hdf5'
    file_2 = 'data/processed/recording_vertical_all_frames_processed_combined.hdf5'
    
    output_file = 'data/processed/recording_combined_all_frames_processed.hdf5'
    combine(file_1, file_2, output_file)
    check_file(output_file)