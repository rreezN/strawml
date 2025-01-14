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
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib.collections import PatchCollection

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors, alphas):
        self.colors = colors
        self.alphas = alphas
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='none', alpha=orig_handle.alphas))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch

class JointPlot:
    def __init__(self, x, label_data, name1_data, name2_data, name1, name2, marginal_x=True, marginal_y=True, plot_data=True, use_label=False):
        """
        Initializes the JointPlot object.
        
        :param x: Array-like, data for the x-axis.
        :param y: Array-like, data for the y-axis.
        :param marginal_x: Boolean, whether to show the marginal distribution for x.
        :param marginal_y: Boolean, whether to show the marginal distribution for y.
        """
        self.x = x
        self.label_data = label_data
        self.name1_data = name1_data
        self.name2_data = name2_data
        self.marginal_x = marginal_x
        self.marginal_y = marginal_y
        self.plot_data = plot_data
        self.name1 = name1
        self.name2 = name2
        self.use_label=use_label
        if name1 == 'scada':
            self.c1 = 'goldenrod'
        elif name1 == 'yolo':
            self.c1 = 'royalblue'
        else:
            self.c1 = 'indianred'

        if name2 == 'scada':
            self.c2 = 'goldenrod'
        elif name2 == 'yolo':
            self.c2 = 'royalblue'
        else:
            self.c2 = 'indianred'

    def plot(self, ax=None):
        """
        Plots the jointplot with optional marginal distributions.

        :param ax: Matplotlib axis object to use for plotting. If None, creates a new figure.
        """
        if ax is None:
            # Create a new figure if no axis is provided
            fig = plt.figure(figsize=(8, 8))
            gs = GridSpec(4, 4, figure=fig)
            ax_joint = fig.add_subplot(gs[1:4, 0:3])
            ax_marginal_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint) if self.marginal_x else None
            ax_marginal_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint) if self.marginal_y else None
        else:
            # Use the provided axis
            fig = ax.figure
            bbox = ax.get_position()
            fig.delaxes(ax)  # Remove the placeholder axis
            max_x, max_y = 12, 30
            gs = GridSpec(max_x, max_y, figure=fig, left=bbox.x0, right=bbox.x1, bottom=bbox.y0, top=bbox.y1)
            
            # Joint plot occupies the larger section
            ax_joint = fig.add_subplot(gs[1:max_x, 0:max_y-3])
            
            # Marginal x plot occupies the top, with the same width
            ax_marginal_x = fig.add_subplot(gs[0, 0:max_y-3], sharex=ax_joint) if self.marginal_x else None
            
            # Marginal y plot occupies the right side, with the same height as ax_joint
            ax_marginal_y = fig.add_subplot(gs[1:max_x-1, max_y-2], sharey=ax_joint) if self.marginal_y else None

        if self.plot_data:
            # Main scatter plot
            if self.use_label:
                ax_joint.plot(self.x, self.label_data, label=f"Annotated Data", c='darkslategray', linestyle='--')
            ax_joint.plot(self.x, self.name1_data, label=f"{self.name1} Data", c=self.c1, linestyle='-')
            ax_joint.plot(self.x, self.name2_data, label=f"{self.name2} Data", c=self.c2, linestyle='-')
            ax_joint.yaxis.tick_right()
            # draw confidence intervals of +- 5%
            ax_joint.fill_between(self.x, self.name1_data - 10, self.name1_data + 10, color=self.c1, alpha=0.5)
            ax_joint.fill_between(self.x, self.name2_data - 10, self.name2_data + 10, color=self.c2, alpha=0.5)
            # add the confidence intervals to the legend
            # ax_joint.plot([], [], color=['lightblue', 'lightcoral'], alpha=0.5, label='Data Threshold (+-5%)')

            # # Highlight the areas where the sensor and model data overlap when within 5% of each other
            # overlap = np.where((self.name1_data >= self.name2_data - 5) & (self.name1_data <= self.name2_data + 5))
            # # Highlight the entire vertical area where the sensor and model data overlap
            # for i in overlap[0]:
            #     ax.axvline(x=self.x[i], color='goldenrod', linestyle='-', alpha=0.2)
            # # add the highlighted area to the legend
            # ax_joint.plot([], [], color='goldenrod', linestyle='-', alpha=0.2, label='Overlap points (w. Threshold)')
            
            if self.marginal_x and ax_marginal_x:
                ax_marginal_x.grid()
                sns.kdeplot(self.x, ax=ax_marginal_x, color='darkslategray', fill=True, vertical=False)
                ax_marginal_x.axis('off')

            if self.marginal_y and ax_marginal_y:
                ax_marginal_y.grid()
                sns.kdeplot(self.name1_data, ax=ax_marginal_y, color=self.c1, fill=True, vertical=True)
                sns.kdeplot(self.name2_data, ax=ax_marginal_y, color=self.c2, fill=True, vertical=True)
                if self.use_label:
                    sns.kdeplot(self.label_data, ax=ax_marginal_y, color="darkslategray", fill=False, vertical=True, linestyle='--', linewidth=1.5)
                ax_marginal_y.axis('off')

            ax_joint.grid()
            ax_joint.set_xlabel("Time")
            ax_joint.set_ylabel("Data")

            # Shrink current axis's height by 10% on the bottom
            box = ax_joint.get_position()
            ax_joint.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])

            # Get current handles and labels
            handles, labels = ax_joint.get_legend_handles_labels()
            # place the data threshold legend at position 2
            sorted_handles_labels = list(zip(handles, labels))

            # Take the c1 color and add 0.5 alpha to it
            sorted_handles_labels.insert(1, (MulticolorPatch([self.c1, self.c2], 0.5), r'Data Threshold ($\pm$10%)'))
            # calculate accuracy of the model in terms of +- 5% threshold wrt. label data
            if self.use_label:
                accuracy_name1 = np.mean((self.name1_data >= self.label_data - 10) & (self.name1_data <= self.label_data + 10)) * 100
                accuracy_name2 = np.mean((self.name2_data >= self.label_data - 10) & (self.name2_data <= self.label_data + 10)) * 100
                # add the accuracy to the legend
                sorted_handles_labels.append((MulticolorPatch([self.c1, 'darkslategray', self.c1], 1), f'Accuracy {self.name1}: {accuracy_name1:.2f}%'))
                sorted_handles_labels.append((MulticolorPatch([self.c2, 'darkslategray', self.c2], 1), f'Accuracy {self.name2}: {accuracy_name2:.2f}%'))

            sorted_handles_labels = sorted(
                sorted_handles_labels, 
                key=lambda hl: "accuracy" in hl[1]
            )
            
            print(sorted_handles_labels)
            # Unzip sorted handles and labels
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)

            # Put a legend below the current axis
            ax_joint.legend(sorted_handles, sorted_labels, handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc='upper center', 
                            bbox_to_anchor=(0.5, 1.12), fancybox=True, shadow=True, ncol=3)
            
            if ax is None:
                plt.tight_layout()
                plt.show()
        else:
            # draw line through 0 to indicate when the model is over or under predicting
            ax_joint.axhline(0, color='black', linestyle='--')
            # axes[2].plot(x_axis_data, sensor_data - model_data, label='Delta', c=cs[0], linestyle='-')
            ax_joint.plot(self.x, self.name1_data - self.name2_data, label=f'Delta ({self.name1} - {self.name2})', c="mediumseagreen", linestyle='-')
            # Highlight the areas where the sensor and model data overlap when within 5% of each other, meaning where the delta is within 5% of 0
            overlap = np.where((self.name1_data - self.name2_data >= -5) & (self.name1_data - self.name2_data <= 5))
            # Highlight the entire vertical area where the sensor and model data overlap
            for i in overlap[0]:
                ax_joint.axvline(x=self.x[i], color='goldenrod', linestyle='-', alpha=0.2)
            ax_joint.plot([], [], color='goldenrod', linestyle='-', alpha=0.2, label='Overlap points (w. Threshold)')
            
            ax_joint.grid()
            ax_joint.set_xlabel("Time")
            ax_joint.set_ylabel("Delta")
            ax_joint.yaxis.tick_right()
            
            if self.marginal_y and ax_marginal_y:
                ax_marginal_y.grid()
                sns.kdeplot(self.name1_data - self.name2_data, ax=ax_marginal_y, color='mediumseagreen', fill=True, vertical=True)
                ax_marginal_y.axis('off')

            # Shrink current axis's height by 10% on the bottom
            box = ax_joint.get_position()
            ax_joint.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])

            # Put a legend below current axis
            ax_joint.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
                    fancybox=True, shadow=True, ncol=5)

def _validate_data(file_path:str):
    missing_keys = {'scada': [], 'yolo': []}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if 'scada' not in f[key].keys():
                missing_keys['scada'].append(key)
            if 'yolo' not in f[key].keys():
                missing_keys['yolo'].append(key)
    return missing_keys

def _retreive_data(file_path: str, name1: str = 'scada', name2: str = 'convnextv2', use_label=False):
    """
    Load the data from the file path
    :param file_path: str: The file path to the data
    :return: np.ndarray: The data
    """
    # missing_keys = _validate_data(file_path)

    # Initialize the dataframes
    label_data = np.array([])
    name1_data = np.array([])
    name2_data = np.array([])

    # We then load the data from the file path
    errors = 0
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        if "frame" == keys[0].split("_")[0]:
            keys = sorted(keys, key=lambda x: int(x.split('_')[1]))
        else:
            keys = sorted(keys, key=lambda x: float(x))
        for key in keys:
            try:
                if use_label:
                    label_data = np.append(label_data, f[key]['straw_percent'][...])
                if name1 not in f[key].keys():
                    name1_data = np.append(name1_data, np.array([0.0]))
                else:
                    name1_data = np.append(name1_data, f[key][name1]['percent'][...])
                if name2 not in f[key].keys():
                    name2_data = np.append(name2_data, np.array([0.0]))
                else:
                    name2_data = np.append(name2_data, f[key][name2]['percent'][...])
            except Exception as e:
                errors += 1
                print(f"Error in loading data from key: {key}")
    
    print(f"Errors in loading data: {errors}")
    x_axis = np.arange(len(name1_data))
    # We then return the data
    if use_label:
        return label_data, name1_data, name2_data, x_axis
    return None, name1_data, name2_data, x_axis
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

def _print_summary_statistics(name1, name2, name1_data, name2_data, label_data):
    print(f"\nSummary Statistics:")
    print(f"  -- Data Length:        #1:  {len(name1_data)}, #2: {len(name2_data)}")
    print(f"  -- {name1} Data:    Mean:  {np.mean(name1_data):.2f}, STD: {np.std(name1_data):.2f}")
    print(f"  -- {name2} Data:    Mean:  {np.mean(name2_data):.2f}, STD: {np.std(name2_data):.2f}")
    print(f"  -- Delta:           Mean:  {np.mean(name1_data - name2_data):.2f}, STD: {np.std(name1_data - name2_data):.2f}")

    if label_data is not None:
        # Print accuracies with different thresholds, for all labels, labels below 50% and labels above 50%
        percentages = [2.5, 5, 10]
        data = [name1_data, name2_data]
        for percentage in percentages:
            for i, name in enumerate([name1, name2]):
                print(f"\nAccuracy (+-{percentage}%) for {name}:")
                accuracy = np.mean((data[i] >= label_data - percentage) & (data[i] <= label_data + percentage)) * 100
                print(f"  -- Accuracy:                      {accuracy:.2f}%")
                # accuracy_below_50 = np.mean((label_data < 50) & (data[i] >= label_data - percentage) & (data[i] <= label_data + percentage)) * 100
                mask = label_data < 50
                accuracy_below_50 = np.mean((data[i][mask] >= label_data[mask] - percentage) & (data[i][mask] <= label_data[mask] + percentage)) * 100
                print(f"  -- Accuracy for labels below 50%: {accuracy_below_50:.2f}%")
                mask = label_data >= 50
                accuracy_above_50 = np.mean((data[i][mask] >= label_data[mask] - percentage) & (data[i][mask] <= label_data[mask] + percentage)) * 100
                print(f"  -- Accuracy for labels above 50%: {accuracy_above_50:.2f}%")

def main(file_path:str, name:str="Recording", name1='yolo', name2='convnextv2', time_step:int = 5, delta:bool = True, use_label=False):  
    # We first define the figure on which we wish to plot the data
    if delta:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    # We then load the data from the file path
    label_data, name1_data, name2_data, x_axis = _retreive_data(file_path, name1=name1, name2=name2, use_label=use_label)
    x_axis_data = x_axis * time_step
    # Plot the data on top of the figure
    if delta:
        JointPlot(x_axis_data, label_data, name1_data, name2_data, name1=name1, name2=name2, marginal_x=False, marginal_y=True, use_label=False).plot(axes[0])
    else:
        JointPlot(x_axis_data, label_data, name1_data, name2_data, name1=name1, name2=name2, marginal_x=False, marginal_y=True, use_label=False).plot(axes)
    if delta:
        JointPlot(x_axis_data, label_data, name1_data, name2_data, name1=name1, name2=name2, marginal_x=False, marginal_y=True, plot_data=False, use_label=False).plot(axes[1])

    _print_summary_statistics(name1, name2, name1_data, name2_data, label_data)
    fig.suptitle(f"Recording: {name}", y=0.92, fontsize=20)

    # Adjust vertical spacing between subplots
    plt.subplots_adjust(hspace=0.2)  # Reduce hspace as needed
    # plt.tight_layout(pad=1.0)  # Adjust padding as necessary
    plt.savefig(f"reports/recording_{name.lower()}_{name1}_{name2}.pdf")
    plt.show()

if __name__ == '__main__':
    file_path = "D:/HCAI/msc/strawml/data/predictions/recording_vertical_all_frames_processed.hdf5"
    # file_path = "D:/HCAI/msc/strawml/data/predictions/recording_vertical_all_frames_processed.hdf5"
    main(file_path, name="sensors", name1='scada', name2='convnextv2', time_step=5, delta=False, use_label=False)