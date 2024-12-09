"""
This file intends to show the recording of the data, when recorded with the stream.py file.

The data is recorded in the following format:

ID 
ID / attr.
ID / attr. / interpolated
ID / attr. /tags
ID / image
ID / scada
ID / scada / percent
ID / scada / pixel
ID / yolo
ID / yolo / percent
ID / yolo / pixel

"""
from __init__ import *
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from strawml.data.make_dataset import decode_binary_image


def _validate_data(file_path:str):
    missing_keys = {'scada': [], 'yolo': []}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if 'scada' not in f[key].keys():
                missing_keys['scada'].append(key)
            if 'yolo' not in f[key].keys():
                missing_keys['yolo'].append(key)
    return missing_keys

def _retreive_data(file_path: str):
    """
    Load the data from the file path
    :param file_path: str: The file path to the data
    :return: np.ndarray: The data
    """
    missing_keys = _validate_data(file_path)

    # Initialize the dataframes
    sensor_data = np.array([])
    model_data = np.array([])

    # We then load the data from the file path
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            sensor_data = np.append(sensor_data, f[key]['scada']['percent'][...])            
            model_data = np.append(model_data, f[key]['yolo']['percent'][...])
    
    # We then return the data
    return sensor_data, model_data

def _smooth_data(sensor_data, model_data):
    """
    This function takes the data and smooths it out. It smooths by taking the average of the previous 5 data points.
    :param sensor_data: np.ndarray: The sensor data
    :param model_data: np.ndarray: The model data
    :return: np.ndarray: The smoothed sensor data
    :return: np.ndarray: The smoothed model data
    """
    # We first initialize the smoothed data
    smoothed_sensor_data = np.array([])
    smoothed_model_data = np.array([])

    # Define weights for previous points (current has highest weight)
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # Must sum to 1
    weights = weights / weights.sum()  # Normalize (in case they don't already sum to 1)

    # We then iterate through the data and smooth it out
    for i in range(len(sensor_data)):
        smoothed_sensor_data.append(np.convolve(sensor_data[i], weights[::-1], mode='valid')) # We use valid here to ensure that 
                                                                                              # the kernel is aligned to consider only past data.
        smoothed_model_data.append(np.convolve(model_data[i], weights[::-1], mode='valid'))

    return smoothed_sensor_data, smoothed_model_data

def _plot_recording(ax, sensor_data, model_data, labels=['a', 'b'], c=['b', 'r'], linestyle='-'):
    # We the iteratre through the data and plot it on the figure with lines in between each data point
    for i in range(len(sensor_data)):
        ax.plot(sensor_data[i], label=labels[0], c=c[0], linestyle=linestyle)
        ax.plot(model_data[i], label=labels[1], c=c[1], linestyle=linestyle)
    return ax

def main(file_path):  
    # We first define the figure on which we wish to plot the data
    fig, ax = plt.subplots()
    
    # We then load the data from the file path
    sensor_data, model_data = _retreive_data(file_path)
    # Plot the data on top of the figure
    ax = _plot_recording(ax, sensor_data, model_data, labels=['Sensor Data', 'Model Data'], c=['b', 'r'], linestyle='-')

    # Calculate a smoothed version of the data.
    smooth_sensor_data, smooth_model_data = _smooth_data(sensor_data, model_data)
    # Plot the data on top of the figure
    ax = _plot_recording(ax, smooth_sensor_data, smooth_model_data, labels=['Sensor Data (Smooth)', 'Model Data (Smooth)'], c=['b', 'r'], linestyle='--')

    ax.set_xlabel('Time')
    ax.set_ylabel('Data')
    ax.set_title('Recording')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    file_path = "D:/HCAI/msc/strawml/data/predictions/recording.hdf5"
    main(file_path)