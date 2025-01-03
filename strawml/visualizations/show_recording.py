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
    # add the confidence intervals to the legend
    ax.plot([], [], color='lightblue', alpha=0.5, label='Data Threshold (+-5%)')
    ax.plot([], [], color='lightcoral', alpha=0.5)

    # Highlight the areas where the sensor and model data overlap when within 5% of each other
    overlap = np.where((sensor_data >= model_data - 5) & (sensor_data <= model_data + 5))
    # Highlight the entire vertical area where the sensor and model data overlap
    for i in overlap[0]:
        ax.axvline(x=x_axis[i], color='goldenrod', linestyle='-', alpha=0.2)
    # add the highlighted area to the legend
    ax.plot([], [], color='goldenrod', linestyle='-', alpha=0.2, label='Overlap points (w. Threshold)')
    
    return ax

def _print_summary_statistics(sensor_data, model_data, smooth_sensor_data, smooth_model_data):
    print(f"\nSummary Statistics:")
    print(f"  -- Sensor Data: Mean:          {np.mean(sensor_data):.2f}, STD: {np.std(sensor_data):.2f}")
    print(f"  -- Model Data: Mean:           {np.mean(model_data):.2f}, STD: {np.std(model_data):.2f}")
    print(f"  -- Delta: Mean:                {np.mean(sensor_data - model_data):.2f}, STD: {np.std(sensor_data - model_data):.2f}")
    print("\n")
    print(f"  -- Smoothed Sensor Data: Mean: {np.mean(smooth_sensor_data):.2f}, STD: {np.std(smooth_sensor_data):.2f}")
    print(f"  -- Smoothed Model Data: Mean:  {np.mean(smooth_model_data):.2f}, STD: {np.std(smooth_model_data):.2f}")
    print(f"  -- Delta: Mean:                {np.mean(smooth_sensor_data - smooth_model_data):.2f}, STD: {np.std(smooth_sensor_data - smooth_model_data):.2f}")
    print("NOTE: Negative values in the delta indicate that the model's predictions are higher than the sensor data.")

def main(file_path:str, time_step:int = 5, delta:bool = True):  
    # We first define the figure on which we wish to plot the data
    if delta:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        titles = ['Raw Data', 'Smoothed Data', 'Delta']
        cs = ['royalblue', 'indianred']
    else:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        titles = ['Raw Data', 'Smoothed Data']
        cs = ['royalblue', 'indianred']

    # We then load the data from the file path
    sensor_data, model_data, x_axis = _retreive_data(file_path)
    x_axis_data = x_axis * time_step
    # Plot the data on top of the figure
    axes[0] = _plot_recording(axes[0], sensor_data, model_data, x_axis_data, 
                              labels=['Sensor Data', 'Model Data'], 
                              c=cs, 
                              linestyle='-')

    # Calculate a smoothed version of the data.
    smooth_sensor_data, smooth_model_data, x_axis = _smooth_data(sensor_data, model_data)
    x_axis_smooth = x_axis * time_step
    # Plot the data on top of the figure
    axes[1] = _plot_recording(axes[1], smooth_sensor_data, smooth_model_data, x_axis_smooth, 
                              labels=['Sensor Data (Smooth)', 'Model Data (Smooth)'], 
                              c=cs, 
                              linestyle='--')

    if delta:
        # draw line through 0 to indicate when the model is over or under predicting
        axes[2].axhline(0, color='black', linestyle='--')
        # axes[2].plot(x_axis_data, sensor_data - model_data, label='Delta', c=cs[0], linestyle='-')
        axes[2].plot(x_axis_smooth, smooth_sensor_data - smooth_model_data, label='Delta (Smooth)', c="mediumseagreen", linestyle='-')
        # Highlight the areas where the sensor and model data overlap when within 5% of each other, meaning where the delta is within 5% of 0
        overlap = np.where((smooth_sensor_data - smooth_model_data >= -5) & (smooth_sensor_data - smooth_model_data <= 5))
        # Highlight the entire vertical area where the sensor and model data overlap
        for i in overlap[0]:
            axes[2].axvline(x=x_axis_smooth[i], color='goldenrod', linestyle='-', alpha=0.2)
        axes[2].plot([], [], color='goldenrod', linestyle='-', alpha=0.2, label='Overlap points (w. Threshold)')


    _print_summary_statistics(sensor_data, model_data, smooth_sensor_data, smooth_model_data)

    for i, ax in enumerate(axes):
        ax.grid()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Data')
        ax.set_title(f'Recording: {titles[i]}')
        
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                fancybox=True, shadow=True, ncol=5)

        # ax.legend()

    fig.tight_layout(pad=5.0)
    plt.show()


if __name__ == '__main__':
    file_path = "D:/HCAI/msc/strawml/data/predictions/recording.hdf5"
    main(file_path)