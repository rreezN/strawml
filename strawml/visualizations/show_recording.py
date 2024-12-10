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
    # missing_keys = _validate_data(file_path)

    # Initialize the dataframes
    sensor_data = np.array([])
    model_data = np.array([])

    # We then load the data from the file path
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            sensor_data = np.append(sensor_data, f[key]['scada']['percent'][...])
            if 'yolo' not in f[key].keys():
                model_data = np.append(model_data, np.array([0.0]))
            else:
                model_data = np.append(model_data, f[key]['yolo']['percent'][...])
    
    x_axis = np.arange(len(sensor_data))
    # We then return the data
    return sensor_data, model_data, x_axis

def _smooth_data(sensor_data, model_data):
    """
    This function takes the data and smooths it out. It smooths by taking the average of the previous 5 data points.
    :param sensor_data: np.ndarray: The sensor data
    :param model_data: np.ndarray: The model data
    :return: np.ndarray: The smoothed sensor data
    :return: np.ndarray: The smoothed model data
    """
    # Define weights for previous points (current has highest weight)
    weights = np.array([0.4, 0.3, 0.1, 0.1, 0.1])  # Must sum to 1
    weights = weights / weights.sum()  # Normalize (in case they don't already sum to 1)


    smoothed_sensor_data = np.convolve(sensor_data, weights[::-1], mode='valid')
    smoothed_model_data = np.convolve(model_data, weights[::-1], mode='valid')
    
    # Now we wish to get the x-axis for the smoothed data accounting for the kernel size and therefore the loss of data points
    x_axis = np.arange(len(sensor_data))
    x_axis = x_axis[len(weights)-1:]  # We remove the first 3 data points as they are lost due to the kernel size of 4

    return smoothed_sensor_data, smoothed_model_data, x_axis

def _plot_recording(ax, sensor_data, model_data, x_axis, labels=['a', 'b'], c=['b', 'r'], linestyle='-'):
    ax.plot(x_axis, sensor_data, label=labels[0], c=c[0], linestyle=linestyle)
    ax.plot(x_axis, model_data, label=labels[1], c=c[1], linestyle=linestyle)

    # draw confidence intervals of +- 5%
    ax.fill_between(x_axis, sensor_data - 5, sensor_data + 5, color='lightblue', alpha=0.5)
    ax.fill_between(x_axis, model_data - 5, model_data + 5, color='lightcoral', alpha=0.5)    
    return ax

def main(file_path:str, time_step:int = 5):  
    # We first define the figure on which we wish to plot the data
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    titles = ['Raw Data', 'Smoothed Data']
    cs = ['royalblue', 'indianred']

    # We then load the data from the file path
    sensor_data, model_data, x_axis = _retreive_data(file_path)
    x_axis = x_axis * time_step
    # Plot the data on top of the figure
    axes[0] = _plot_recording(axes[0], sensor_data, model_data, x_axis, 
                              labels=['Sensor Data', 'Model Data'], 
                              c=cs, 
                              linestyle='-')

    # Calculate a smoothed version of the data.
    smooth_sensor_data, smooth_model_data, x_axis = _smooth_data(sensor_data, model_data)
    x_axis = x_axis * time_step
    # Plot the data on top of the figure
    axes[1] = _plot_recording(axes[1], smooth_sensor_data, smooth_model_data, x_axis, 
                              labels=['Sensor Data (Smooth)', 'Model Data (Smooth)'], 
                              c=cs, 
                              linestyle='--')

    for i, ax in enumerate(axes):
        ax.grid()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Data')
        ax.set_title(f'Recording: {titles[i]}')
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = "D:/HCAI/msc/strawml/data/predictions/recording.hdf5"
    main(file_path)