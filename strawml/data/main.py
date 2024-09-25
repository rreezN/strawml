"""
The run guide for new data.

1. Run the following command to extract frames from the video:
    python strawml/data/make_dataset.py extract --fbf 1000 --video_folder path/to/video

2. Run the following command to label the extracted frames:
    python strawml/annotate/annotate_gui.py

3. Augment the dataset:
    python strawml/data/augment_chute_data.py --data_folder path/to/extracted_frames

4. (optional) Combine the augmented dataset with any previously augmented datasets
    python strawml/data/utils.py combine --data_path path/to/hdf5files --file1 name_of_file1 --file2 name_of_file2 --output_file path/to/output_file

5. Extract to yolo format:
    python strawml/data/make_dataset.py h5_to_yolo --hdf5_file path/to/hdf5file

6. Train the model:
    
"""


"""
Create a dataset that consists of the chute data and HWD+ data, with the digits resized to small size and placed on top of the chute images.

-   The numbers should be black digits on a white background.
-   The numbers should be resized to 28x28 pixels.
-   We can add the numbers after we have augmented the chute data.
-   The numbers should be placed at a random location on the chute image.
-   The numbers should be placed at a random angle between -45 and 45 degrees.
"""