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
import matplotlib.lines as mlines


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
                           edgecolor='none', alpha=orig_handle.alphas[i]))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch

class JointPlot:
    def __init__(self, x, label_data, name1_data, name2_data, name3_data, name4_data, name1, name2, name3, name4, marginal_x=True, marginal_y=True, plot_data=True, use_label=False, label_as="scada", with_threshold=True, changed_index=None):
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
        self.name3_data = name3_data
        self.name4_data = name4_data
        self.marginal_x = marginal_x
        self.marginal_y = marginal_y
        self.plot_data = plot_data
        self.name1 = name1
        self.name2 = name2
        self.name3 = name3
        self.name4 = name4
        self.use_label = use_label
        self.label_as = label_as
        self.with_threshold=with_threshold
        self.changed_index=changed_index
        self.change_color = 'mediumseagreen'
        
        if name1 == 'scada':
            self.c1 = 'goldenrod'
        elif name1 == 'yolo' or name1 == 'yolo_smoothing':
            self.c1 = 'royalblue'
        else:
            self.c1 = 'indianred'

        if name2 == 'scada':
            self.c2 = 'goldenrod'
        elif name2 == 'yolo' or name2 == 'yolo_smoothing':
            self.c2 = 'royalblue'
        else:
            self.c2 = 'indianred'

        if name3 == 'scada':
            self.c3 = 'goldenrod'
        elif name3 == 'yolo' or name3 == 'yolo_smoothing':
            self.c3 = 'royalblue'
        else:
            self.c3 = 'indianred'

        if name4 == 'scada':
            self.c4 = 'goldenrod'
        elif name4 == 'yolo' or name4 == 'yolo_smoothing':
            self.c4 = 'royalblue'
        else:
            self.c4 = 'indianred'
        
        # if none of the data is None then we specify special colors for the data
        # if name3 == 'convnext_apriltag' and name4 is None:
        self.c1 = 'goldenrod'
        self.c2 = 'royalblue'
        self.c3 = 'indianred'
        self.c4 = 'purple'

        # if name1_data is not None and name2_data is not None and name3_data is not None and name4_data is not None:
        #     self.c1 = 'royalblue'
        #     self.c2 = '#182e6f'
        #     self.c3 = 'indianred'
        #     self.c4 = '#892A2A'

        self.n_cols = 3 if name4_data is not None else 3 if name3_data is not None else 2 if name2_data is not None else 2
        # self.n_cols = 2

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
            ax_marginal_y = fig.add_subplot(gs[1:max_x, max_y-2], sharey=ax_joint) if self.marginal_y else None


        if self.plot_data:
            # Main scatter plot
            if self.use_label:
                ax_joint.plot(self.x, self.label_data['straw_percent_bbox'], label=f"label data", c='darkslategray', linestyle='--')
            if self.name1_data is not None:
                ax_joint.plot(self.x, self.name1_data, label=f"{self.name1.upper()} data", c=self.c1, linestyle='-', alpha=1)
            if self.name2_data is not None:
                ax_joint.plot(self.x, self.name2_data, label=f"{self.name2.upper()} data", c=self.c2, linestyle='-', alpha=1)
            # ax_joint.scatter(self.x[8], self.name2_data[8], color=self.change_color, s=50, marker='o', zorder=10000, label='Spike extrema')
            # ax_joint.scatter(self.x[20], self.name2_data[20], color=self.change_color, s=50, marker='o', zorder=10000)
            # ax_joint.scatter(self.x[21], self.name3_data[21], color=self.change_color, s=50, marker='o', zorder=10000)
            if self.name3_data is not None:
                ax_joint.plot(self.x, self.name3_data, label=f"{self.name3.upper()} data", c=self.c3, linestyle='-', alpha=1)
            if self.name4_data is not None:
                ax_joint.plot(self.x, self.name4_data, label=f"{self.name4.upper()} data", c=self.c4, linestyle='-', alpha=1)

            ax_joint.yaxis.tick_right()
            # draw confidence intervals of +- 5%
            if self.with_threshold:
                ax_joint.fill_between(self.x, self.label_data['straw_percent_bbox'] - 10, self.label_data['straw_percent_bbox'] + 10, color="darkslategray", alpha=0.2)
                # ax_joint.fill_between(self.x, self.name1_data - 10, self.name1_data + 10, color=self.c1, alpha=0.5)
                # ax_joint.fill_between(self.x, self.name2_data - 10, self.name2_data + 10, color=self.c2, alpha=0.5)
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
                sns.kdeplot(self.x, ax=ax_marginal_x, color='darkslategray', fill=False, vertical=False)
                ax_marginal_x.axis('off')

            if self.marginal_y and ax_marginal_y:
                ax_marginal_y.grid()
                if self.name1_data is not None:
                    sns.kdeplot(self.name1_data, ax=ax_marginal_y, color=self.c1, fill=False, vertical=True, clip_on=False, linestyle='-')
                if self.name2_data is not None:
                    sns.kdeplot(self.name2_data, ax=ax_marginal_y, color=self.c2, fill=False, vertical=True, clip_on=False, linestyle='-')
                if self.name3_data is not None:
                    sns.kdeplot(self.name3_data, ax=ax_marginal_y, color=self.c3, fill=False, vertical=True, clip_on=False, linestyle='-')
                if self.name4_data is not None:
                    sns.kdeplot(self.name4_data, ax=ax_marginal_y, color=self.c4, fill=False, vertical=True, clip_on=False, linestyle='-')
                if self.use_label:
                    sns.kdeplot(self.label_data['straw_percent_bbox'], ax=ax_marginal_y, color="darkslategray", fill=False, vertical=True, linestyle='--', linewidth=1.5, clip_on=False)
                # turn off the label data axis
                ax_marginal_y.axis('off')

            if self.changed_index is not None:
                ax_joint.axvline(x=self.x[self.changed_index], color=self.change_color, linestyle='--')
            # plot circle at x=28 and self.name1_data[28] to indicate the change in the model
            ax_joint.grid()
            ax_joint.set_xlabel("Frame", fontsize=15)
            ax_joint.set_ylabel("Straw level (%)", fontsize=15)
            ax_joint.set_yticks(np.arange(0, 101, 10))
            ax_joint.set_yticklabels(np.arange(0, 101, 10))
            # fix fontsize of y axis
            ax_joint.yaxis.set_tick_params(labelsize=12)

            # set xtiks with 90 degree rotation for every 5th tick
            # ax_joint.set_xticks(np.arange(np.min(self.x), np.min(self.x) + len(self.x), 5))
            # ax_joint.set_xticklabels(np.arange(np.min(self.x), np.min(self.x) + len(self.x), 5), rotation=90)
            # set font size
            ax_joint.xaxis.set_tick_params(labelsize=12)
            # ax_joint.xaxis.set_tick_params(labelsize=12)
            # Shrink current axis's height by 10% on the bottom
            box = ax_joint.get_position()
            ax_joint.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])

            # Get current handles and labels
            # handles, labels = ax_joint.get_legend_handles_labels()
            # place the data threshold legend at position 2
            # sorted_handles_labels = list(zip(handles, labels))
            # create empty list to store the sorted handles and labels
            sorted_handles_labels = []

            # first insert label data as dashed line
            if self.use_label:
                label_name = 'Annotations'
                label = plt.Line2D([], [], color='darkslategray', label=label_name)
                sorted_handles_labels.append((label, label_name))

            # Take the c1 color and add 0.5 alpha to it
            if self.with_threshold:
                sorted_handles_labels.insert(1,(MulticolorPatch(['darkslategray', 'darkslategray', 'darkslategray'], [0.2, 1, 0.2]), r'Threshold ($\pm$10%)'))

            # calculate accuracy of the model in terms of +- 5% threshold wrt. label data
            if self.use_label and self.label_as != 'scada':
                if self.name1_data is not None:
                    if self.name1 == 'convnext' or self.name1 == 'convnext_apriltag' or self.name1 == 'convnext_smooth':
                        label_data = self.label_data['straw_percent_fullness']
                    else:
                        label_data = self.label_data['straw_percent_bbox']
                    accuracy_name1 = np.mean((self.name1_data >= label_data - 10) & (self.name1_data <= label_data + 10)) * 100
                    mae_name1 = np.mean(np.abs(self.name1_data - label_data))
                    # add the accuracy to the legend
                    label_name = self.name1.upper()
                    # sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c1, 'darkslategray'], [0.2, 1, 0.2]), f'{label_name}, Accuracy: {accuracy_name1:.2f}%, MAE: {mae_name1:.2f}'))
                    sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c1, 'darkslategray'], [0.2, 1, 0.2]), f'{label_name}'))

                if self.name2_data is not None:
                    if self.name2 == 'convnext' or self.name2 == 'convnext_apriltag' or self.name2 == 'convnext_smooth':
                        label_data = self.label_data['straw_percent_fullness']
                    else:
                        label_data = self.label_data['straw_percent_bbox']
                    accuracy_name2 = np.mean((self.name2_data >= label_data - 10) & (self.name2_data <= label_data + 10)) * 100
                    mae_name2 = np.mean(np.abs(self.name2_data - label_data))
                    label_name = self.name2.upper() + "-S"
                    label_name = 'YOLO-S C.'
                    # sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c2, 'darkslategray'], [0.2, 1, 0.2]), f'{label_name}, Accuracy: {accuracy_name2:.2f}%, MAE: {mae_name2:.2f}'))
                    sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c2, 'darkslategray'], [0.2, 1, 0.2]), f'{label_name}'))

                if self.name3_data is not None :
                    if self.name3 == 'convnext' or self.name3 == 'convnext_apriltag' or self.name3 == 'convnext_smooth':
                        label_data = self.label_data['straw_percent_fullness']
                    else:
                        label_data = self.label_data['straw_percent_bbox']
                    accuracy_name3 = np.mean((self.name3_data >= label_data - 10) & (self.name3_data <= label_data + 10)) * 100
                    mae_name3 = np.mean(np.abs(self.name3_data - label_data))
                    label_name = self.name3.upper() + "V1"
                    label_name = 'ConvNeXtV1'
                    # sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c3, 'darkslategray'], [0.2, 1, 0.2]), f'{label_name}, Accuracy: {accuracy_name3:.2f}%, MAE: {mae_name3:.2f}'))
                    sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c3, 'darkslategray'], [0.2, 1, 0.2]), f'{label_name}'))

                if self.name4_data is not None:
                    if self.name4 == 'convnext' or self.name4 == 'convnext_apriltag' or self.name4 == 'convnext_smooth':
                        label_data = self.label_data['straw_percent_fullness']
                    else:
                        label_data = self.label_data['straw_percent_bbox']
                    accuracy_name4 = np.mean((self.name4_data >= label_data - 10) & (self.name4_data <= label_data + 10)) * 100
                    mae_name4 = np.mean(np.abs(self.name4_data - label_data))
                    label_name = self.name4.upper()
                    label_name = 'Ens. Average'
                    # sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c4, 'darkslategray'], [0.2, 1, 0.2]), f'{label_name}, Accuracy: {accuracy_name4:.2f}%, MAE: {mae_name4:.2f}'))
                    sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c4, 'darkslategray'], [0.2, 1, 0.2]), f'{label_name}'))

            if self.changed_index is not None:

                # Define a custom line style for the striped line
                striped_line = mlines.Line2D([], [], color=self.change_color, linestyle='--', linewidth=2, label='Type Change')

                # Append the custom line to the sorted_handles_labels
                sorted_handles_labels.append((striped_line, 'Type Change'))

            # ADD THE GREEN DOT TO THE LEGEND
            # Create a custom legend handler for the green dot
            green_dot = mlines.Line2D([], [], color=self.change_color, marker='o', linestyle='None', markersize=8, label='Spike extrema')
            # Append the green dot to the sorted_handles_labels
            sorted_handles_labels.append((green_dot, 'Spike extrema'))


            sorted_handles_labels = sorted(
                sorted_handles_labels, 
                key=lambda hl: "accuracy" in hl[1]
            )
            
            print(sorted_handles_labels)
            # Unzip sorted handles and labels
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)

            # Put a legend below the current axis
            ax_joint.legend(sorted_handles, sorted_labels, handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc='upper center', 
                            bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=self.n_cols, fontsize=14)
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

def _retrieve_iou_data(file_path: str):

    # Initialize the dataframes
    label_bbox = np.array([]).reshape(0,8)
    yolo_bbox = np.array([]).reshape(0,8)
    # We then load the data from the file path
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        if "frame" == keys[0].split("_")[0]:
            keys = sorted(keys, key=lambda x: int(x.split('_')[1]))
        else:
            keys = sorted(keys, key=lambda x: float(x))
        for key in keys:

            yolo_bbox = np.vstack([yolo_bbox, f[key]['yolo_cutout'][...]])
            label_bbox = np.vstack([label_bbox, f[key]['annotations']['bbox_chute'][...]])

    return yolo_bbox, label_bbox

def _retreive_data(file_path: str, name1: str = 'scada', name2: str = 'convnextv2', name3: str|None = None, name4: str|None = None, use_label=False, label_as='scada'):
    """
    Load the data from the file path
    :param file_path: str: The file path to the data
    :return: np.ndarray: The data
    """
    # missing_keys = _validate_data(file_path)

    # Initialize the dataframes
    label_data = {i: np.array([]) for i in label_as}
    name1_data = np.array([])
    name2_data = np.array([])
    name3_data = np.array([])
    name4_data = np.array([])
    # We then load the data from the file path
    errors = 0

    for path in file_path:
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            print(f"No. of keys: {len(keys)}")
            changed_index = None 
            if "frame" == keys[0].split("_")[0]:
                keys = sorted(keys, key=lambda x: int(x.split('_')[1]))
            else:
                keys = sorted(keys, key=lambda x: float(x))
            if 'type' in f[keys[0]].attrs.keys():
                old_type = f[keys[0]].attrs['type']
            for key in keys:
                try:
                    if use_label:
                        # scada, straw_percent_bbox, straw_percent_fullness, fullness
                        for label in label_as:
                            if label not in f[key].keys():
                                label_data[label] = np.append(label_data[label], np.array([0.0]))
                            else:
                                label_data[label] = np.append(label_data[label], f[key][label]['percent'][...])

                    if name1 is not None:
                        if name1 not in f[key].keys():
                            name1_data = np.append(name1_data, np.array([0.0]))
                        else:
                            name1_data = np.append(name1_data, f[key][name1]['percent'][...])
                    else:
                        name1_data = np.append(name1_data, np.array([0.0]))

                    if name2 is not None:
                        if name2 not in f[key].keys():
                            name2_data = np.append(name2_data, np.array([0.0]))
                        else:
                            name2_data = np.append(name2_data, f[key][name2]['percent'][...])
                    else:
                        name2_data = np.append(name2_data, np.array([0.0]))

                    if name3 is not None:
                        if name3 not in f[key].keys():
                            if name3.lower() == 'average':
                                name3_data = np.append(name3_data, (f[key][name1]['percent'][...] + f[key][name2]['percent'][...])/2)
                            else:
                                name3_data = np.append(name3_data, np.array([0.0]))
                        else:
                            name3_data = np.append(name3_data, f[key][name3]['percent'][...])
                    else:
                        name3_data = np.append(name3_data, np.array([0.0]))

                    if name4 is not None:
                        if name4 not in f[key].keys():
                            if name4.lower() == 'average':
                                name4_data = np.append(name4_data, (f[key][name2]['percent'][...] + f[key][name3]['percent'][...])/2)
                            else:
                                name4_data = np.append(name4_data, np.array([0.0]))
                        else:
                            name4_data = np.append(name4_data, f[key][name4]['percent'][...])
                    else:
                        name4_data = np.append(name4_data, np.array([0.0]))

                    if 'type' in f[key].attrs.keys():
                        if f[key].attrs['type'] != old_type:
                            print(f"Old type: {old_type}, new type: {f[key].attrs['type']}")
                            print(f"Changed index: {key}")
                            changed_index = keys.index(key)
                            old_type = f[key].attrs['type']
                except Exception as e:
                    errors += 1
                    print(f"Error in loading data from key: {key}, {e}")
    
    print(f"Errors in loading data: {errors}")
    x_axis = np.arange(len(name1_data))
    # We then return the data
    if use_label:
        return label_data, name1_data, name2_data, name3_data, name4_data, x_axis, changed_index
    return None, name1_data, name2_data, name3_data, name4_data, x_axis, changed_index

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

def _print_summary_statistics(name1, name2, name3, name4, name1_data, name2_data, name3_data, name4_data, label_data_dict, label_as='scada'):
    print(f"\nSummary Statistics:")

    if label_as != 'scada':
        # Print accuracies with different thresholds, for all labels, labels below 50% and labels above 50%
        percentages = [2.5, 5, 10]
        # create list of data to loop through only if not None
        # create list of names and data to loop through only if not None
        name_data_pairs = [
            (name1, name1_data),
            (name2, name2_data),
            (name3, name3_data),
            (name4, name4_data),
        ]

        names = [name for name, data in name_data_pairs if data is not None]
        data = [data for _, data in name_data_pairs if data is not None]
        
        for percentage in percentages:
            for i, name in enumerate(names):
                if name == 'convnext' or name == 'convnext_apriltag':
                    label_data = label_data_dict['straw_percent_fullness']
                else:
                    label_data = label_data_dict['straw_percent_bbox']

                mask = ~np.isnan(label_data) & ~np.isnan(data[i])
                frame_detection_accuracy  = np.sum(~np.isnan(data[i])) / len(data[i]) * 100
                print(f"\nFrame Detection Accuracy for {name}: {frame_detection_accuracy}%" )
                label_data = label_data[mask]
                prediction_data = data[i][mask]
                print(f"\nAccuracy (+-{percentage}%) for {name}:")
                accuracy = np.mean((prediction_data >= label_data - percentage) & (prediction_data <= label_data + percentage)) * 100
                print(f"  -- Accuracy:                      {accuracy:.2f}%")
                # accuracy_below_50 = np.mean((label_data < 50) & (data[i] >= label_data - percentage) & (data[i] <= label_data + percentage)) * 100
                mask = label_data < 50
                accuracy_below_50 = np.mean((prediction_data[mask] >= label_data[mask] - percentage) & (prediction_data[mask] <= label_data[mask] + percentage)) * 100
                print(f"  -- Accuracy for labels below 50%: {accuracy_below_50:.2f}%, n = {len(prediction_data[mask])}")
                mask = label_data >= 50
                accuracy_above_50 = np.mean((prediction_data[mask] >= label_data[mask] - percentage) & (prediction_data[mask] <= label_data[mask] + percentage)) * 100
                print(f"  -- Accuracy for labels above 50%: {accuracy_above_50:.2f}%, n = {len(prediction_data[mask])}")
                

def main(file_path:str, name:str="Recording", name1='yolo', name2='convnextv2', name3=None, name4=None, time_step:int = 5, delta:bool = True, use_label=False, label_as='scada', with_threshold=False, iou=False):  
    # We first define the figure on which we wish to plot the data
    if delta:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(20,5), sharey=True)

    if iou:
        import cv2
        from shapely.geometry import Polygon
        def calculate_iou_rotated(box1, box2):
            # Define polygons for each box from the corner coordinates
            poly1 = Polygon([(box1[0], box1[1]), (box1[2], box1[3]), 
                            (box1[4], box1[5]), (box1[6], box1[7])])
            poly2 = Polygon([(box2[0], box2[1]), (box2[2], box2[3]), 
                            (box2[4], box2[5]), (box2[6], box2[7])])

            # Calculate the intersection area
            intersection = poly1.intersection(poly2)
            intersection_area = intersection.area

            # Calculate the union area
            union_area = poly1.area + poly2.area - intersection_area

            # Calculate the IoU
            iou = intersection_area / union_area if union_area > 0 else 0

            return iou
        yolo_bbox, label_bbox = _retrieve_iou_data(file_path)
        # calculate the intersection over union between the two bounding boxes for each frame,
        # knowing that the bounding boxes are in the format [x1, y1, x2, y2, x3, y3, x4, y4]
        iou = []
        for i in range(len(yolo_bbox)):
            # get the coordinates of the bounding boxes
            yolo_coords = yolo_bbox[i].reshape(4, 2)
            label_coords = label_bbox[i].reshape(4, 2)
            iou.append(calculate_iou_rotated(yolo_coords.flatten(), label_coords.flatten()))
        print(f"Mean IOU: {np.mean(iou):.2f}")
        print(f"Std IOU: {np.std(iou):.2f}")
        print(f"Max IOU: {np.max(iou):.2f}")
        print(f"Min IOU: {np.min(iou):.2f}")
    else:
        # We then load the data from the file path
        label_data, name1_data, name2_data, name3_data, name4_data, x_axis, changed_index = _retreive_data(file_path, name1=name1, name2=name2, name3=name3, name4=name4, use_label=use_label, label_as=label_as)
        # start_idx = 20
        # end_idx = 45
        start_idx = 0
        end_idx = len(name1_data)
        label_data = {k: v[start_idx:end_idx] for k, v in label_data.items()}
        x_axis = x_axis[start_idx:end_idx]

        if name1 is None:
            name1_data = None
        else:
            name1_data = name1_data[start_idx:end_idx]

        if name2 is None:
            name2_data = None
        else:
            name2_data = name2_data[start_idx:end_idx]

        if name3 is None:
            name3_data = None
        else:
            name3_data = name3_data[start_idx:end_idx]

        if name4 is None:
            name4_data = None
        else:
            name4_data = name4_data[start_idx:end_idx]


        x_axis_data = x_axis * time_step
        # Plot the data on top of the figure
        if delta:
            JointPlot(x_axis_data, label_data, name1_data, name2_data, name3_data, name4_data, name1=name1, name2=name2, name3=name3, name4=name4, marginal_x=False, marginal_y=True, use_label=use_label, label_as=label_as, with_threshold=with_threshold, changed_index=changed_index).plot(axes[0])
        else:
            JointPlot(x_axis_data, label_data, name1_data, name2_data, name3_data, name4_data, name1=name1, name2=name2, name3=name3, name4=name4, marginal_x=False, marginal_y=False, use_label=use_label, label_as=label_as, with_threshold=with_threshold, changed_index=changed_index).plot(axes)

        _print_summary_statistics(name1, name2, name3, name4, name1_data, name2_data, name3_data, name4_data, label_data_dict=label_data, label_as=label_as)
        
        if len(file_path) > 1:
            name = 'Vertical and Rotated combined'
        else:
            file_path = file_path[0]
            name = file_path.split("/")[-1].split(".")[0].split("_")
        if "rotated" in name:
            name = "Rotated"
        elif "vertical" in name:
            name = "Vertical"
        elif "combined" in name:
            name = "Combined of Vertical and Rotated"
        try:
            fig.suptitle(f"{name.replace('_', ' ')}", y=0.97, fontsize=25)
        except Exception as e:
            print(e)
        # fig.suptitle(f"Sensors", y=0.97, fontsize=25)
        # Adjust vertical spacing between subplots
        plt.subplots_adjust(hspace=0.2)  # Reduce hspace as needed
        # plt.tight_layout(pad=1.0)  # Adjust padding as necessary
        plt.savefig(f"reports/recording_{name[0].lower()}_{name1}_{name2}_{name3}.pdf")
        plt.show()

if __name__ == '__main__':
    # file_path = "data/predictions/recording_rotated_all_frames_processed.hdf5"
    # file_path = "data/predictions/recording_rotated_all_frames_processed_combined.hdf5"
    # file_path = "data/predictions/recording_vertical_all_frames_processed_combined.hdf5"
    # file_path = "data/predictions/recording_vertical_all_frames_processed_combined_processed.hdf5"

    # file_path = ["data/predictions/new_run/recording_combined_all_frames_processed.hdf5"]
    file_path = ["D:/HCAI/msc/strawml/data/interim/sensors_with_strawbbox_processed.hdf5"]
    # file_path = 'data/noisy_datasets/noisy_1.hdf5'
    # file_path = ['data/predictions/new_run/recording_vertical_all_frames_processed_combined.hdf5', 'data/predictions/new_run/recording_rotated_all_frames_processed_combined.hdf5']
    # file_path = ['data/predictions/new_run/recording_rotated_all_frames_processed_combined.hdf5']
    # file_path = ['data/predictions/new_run/recording_vertical_all_frames_processed_combined.hdf5']
    # file_path = ["data/predictions/new_run/sensors_with_strawbbox.hdf5"]


    main(file_path, name="sensors", name1='scada', name2="yolo", name3=None, name4=None, time_step=1, delta=False, use_label=True, label_as=['straw_percent_fullness', 'straw_percent_bbox'], with_threshold=True, iou=False)