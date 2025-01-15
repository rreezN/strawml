from __init__ import *
import h5py

set_type = "rotated" 
file1_path = f"data/predictions/recording_{set_type}_all_frames_processed.hdf5"
file2_path = f"data/predictions/recording_{set_type}_all_frames_processed_chute_annotated.hdf5"
file3_path = f"data/predictions/recording_{set_type}_all_frames_processed_straw_annotated.hdf5"


chute_file = h5py.File(file2_path, 'r')
straw_file = h5py.File(file3_path, 'r')

with h5py.File(file1_path, 'r+') as original_file:
    timestamps = list(original_file.keys())

    # Sort timestamps
    if "frame" == timestamps[0].split("_")[0]:
        timestamps = sorted(timestamps, key=lambda x: int(x.split('_')[1]))
    else:
        timestamps = sorted(timestamps, key=lambda x: float(x))

    for timestamp in timestamps:
        try:
            original_frame = original_file[timestamp]

            # Copy all datasets from chute annotations
            try:
                if 'annotations' in chute_file[timestamp]:
                    for dataset_name, dataset in chute_file[timestamp]['annotations'].items():
                        target_path = f"{timestamp}/annotations/{dataset_name}"
                        if target_path not in original_file:
                            original_file.create_dataset(target_path, data=dataset[...])
            except KeyError as e:
                print(f"1, no chute annotations for {timestamp}, {e}")

            # Copy all datasets from straw annotations
            try:
                if 'annotations' in straw_file[timestamp]:
                    for dataset_name, dataset in straw_file[timestamp]['annotations'].items():
                        target_path = f"{timestamp}/annotations/{dataset_name}"
                        if target_path not in original_file:
                            original_file.create_dataset(target_path, data=dataset[...])
            except KeyError as e:
                print(f"2, no straw annotations for {timestamp}, {e}")

        except KeyError as e:
            print(f"Error accessing timestamp {timestamp}, {e}")

# Close the read-only files
chute_file.close()
straw_file.close()