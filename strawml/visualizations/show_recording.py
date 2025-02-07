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
    def __init__(self, x, label_data, name1_data, name2_data, name1, name2, marginal_x=True, marginal_y=True, plot_data=True, use_label=False, label_as="scada", with_threshold=True, changed_index=None):
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
        self.use_label = use_label
        self.label_as = label_as
        self.with_threshold=with_threshold
        self.changed_index=changed_index
        self.change_color = 'mediumseagreen'
        
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

        self.c2 = 'royalblue'

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
                ax_joint.plot(self.x, self.label_data, label=f"label data", c='darkslategray', linestyle='--')
            ax_joint.plot(self.x, self.name1_data, label=f"{self.name1.upper()} data", c=self.c1, linestyle='-')
            ax_joint.plot(self.x, self.name2_data, label=f"{self.name2.upper()} data", c=self.c2, linestyle='-')

            ax_joint.yaxis.tick_right()
            # draw confidence intervals of +- 5%
            if self.with_threshold:
                ax_joint.fill_between(self.x, self.label_data - 10, self.label_data + 10, color="darkslategray", alpha=0.2)
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
                sns.kdeplot(self.name1_data, ax=ax_marginal_y, color=self.c1, fill=False, vertical=True, clip_on=False)
                sns.kdeplot(self.name2_data, ax=ax_marginal_y, color=self.c2, fill=False, vertical=True, clip_on=False)
                if self.use_label:
                    sns.kdeplot(self.label_data, ax=ax_marginal_y, color="darkslategray", fill=False, vertical=True, linestyle='--', linewidth=1.5, clip_on=False)
                # turn off the label data axis
                ax_marginal_y.axis('off')

            if self.changed_index is not None:
                ax_joint.axvline(x=self.x[self.changed_index], color=self.change_color, linestyle='--')

            ax_joint.grid()
            ax_joint.set_xlabel("Time (s)")
            ax_joint.set_ylabel("Straw level (%)")
            ax_joint.set_yticks(np.arange(0, 101, 10))
            ax_joint.set_yticklabels(np.arange(0, 101, 10))
            # Shrink current axis's height by 10% on the bottom
            box = ax_joint.get_position()
            ax_joint.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])

            # Get current handles and labels
            handles, labels = ax_joint.get_legend_handles_labels()
            # place the data threshold legend at position 2
            sorted_handles_labels = list(zip(handles, labels))

            # Take the c1 color and add 0.5 alpha to it
            if self.with_threshold:
                sorted_handles_labels.insert(1,(MulticolorPatch(['darkslategray', 'darkslategray', 'darkslategray'], [0.2, 1, 0.2]), r'Data Threshold ($\pm$10%)'))

            # calculate accuracy of the model in terms of +- 5% threshold wrt. label data
            if self.use_label and self.label_as != 'scada':
                accuracy_name1 = np.mean((self.name1_data >= self.label_data - 10) & (self.name1_data <= self.label_data + 10)) * 100
                accuracy_name2 = np.mean((self.name2_data >= self.label_data - 10) & (self.name2_data <= self.label_data + 10)) * 100
                # calculate MAE
                mae_name1 = np.mean(np.abs(self.name1_data - self.label_data))
                mae_name2 = np.mean(np.abs(self.name2_data - self.label_data))
                # add the accuracy to the legend
                sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c1, 'darkslategray'], [0.2, 1, 0.2]), f'{self.name1.upper()}, Accuracy: {accuracy_name1:.2f}%, MAE: {mae_name1:.2f}'))
                sorted_handles_labels.append((MulticolorPatch(['darkslategray', self.c2, 'darkslategray'], [0.2, 1, 0.2]), f'{self.name2.upper()}, Accuracy: {accuracy_name2:.2f}%, MAE: {mae_name2:.2f}'))

            if self.changed_index is not None:

                # Define a custom line style for the striped line
                striped_line = mlines.Line2D([], [], color=self.change_color, linestyle='--', linewidth=2, label='Type Change')

                # Append the custom line to the sorted_handles_labels
                sorted_handles_labels.append((striped_line, 'Type Change'))

            sorted_handles_labels = sorted(
                sorted_handles_labels, 
                key=lambda hl: "accuracy" in hl[1]
            )
            
            print(sorted_handles_labels)
            # Unzip sorted handles and labels
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)

            # Put a legend below the current axis
            ax_joint.legend(sorted_handles, sorted_labels, handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc='upper center', 
                            bbox_to_anchor=(0.5, 1.12), fancybox=True, shadow=True, ncol=4)
            
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

def _retreive_data(file_path: str, name1: str = 'scada', name2: str = 'convnextv2', use_label=False, label_as='scada'):
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
                    if label_as == 'scada':
                        label_data = np.append(label_data, f[key]['scada']['percent'][...])
                    elif label_as == 'straw_percent_bbox':
                        label_data = np.append(label_data, f[key]['straw_percent_bbox']['percent'][...])
                    elif label_as == 'straw_percent_fullness':
                        label_data = np.append(label_data, f[key]['straw_percent_fullness']['percent'][...])
                    else:
                        label_data = np.append(label_data, f[key]['annotations']['fullness'][...]*100)
                if name1 not in f[key].keys():
                    name1_data = np.append(name1_data, np.array([0.0]))
                else:
                    name1_data = np.append(name1_data, f[key][name1]['percent'][...])
                if name2 not in f[key].keys():
                    name2_data = np.append(name2_data, np.array([0.0]))
                else:
                    name2_data = np.append(name2_data, f[key][name2]['percent'][...])
                if 'type' in f[key].attrs.keys():
                    if f[key].attrs['type'] != old_type:
                        print(f"Old type: {old_type}, new type: {f[key].attrs['type']}")
                        print(f"Changed index: {key}")
                        changed_index = keys.index(key)
                        old_type = f[key].attrs['type']
            except Exception as e:
                errors += 1
                print(f"Error in loading data from key: {key}")
    
    print(f"Errors in loading data: {errors}")
    x_axis = np.arange(len(name1_data))
    # We then return the data
    if use_label:
        return label_data, name1_data, name2_data, x_axis, changed_index
    return None, name1_data, name2_data, x_axis, changed_index

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

def _print_summary_statistics(name1, name2, name1_data, name2_data, label_data, label_as='scada'):
    print(f"\nSummary Statistics:")
    print(f"  -- Data Length:        #1:  {len(name1_data)}, #2: {len(name2_data)}")
    print(f"  -- {name1} Data:    Mean:  {np.mean(name1_data):.2f}, STD: {np.std(name1_data):.2f}")
    print(f"  -- {name2} Data:    Mean:  {np.mean(name2_data):.2f}, STD: {np.std(name2_data):.2f}")
    print(f"  -- Delta:           Mean:  {np.mean(name1_data - name2_data):.2f}, STD: {np.std(name1_data - name2_data):.2f}")

    if label_data is not None and label_as != 'scada':
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

def main(file_path:str, name:str="Recording", name1='yolo', name2='convnextv2', time_step:int = 5, delta:bool = True, use_label=False, label_as='scada', with_threshold=False, iou=False):  
    # We first define the figure on which we wish to plot the data
    if delta:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(20, 5), sharey=True)

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
        label_data, name1_data, name2_data, x_axis, changed_index = _retreive_data(file_path, name1=name1, name2=name2, use_label=use_label, label_as=label_as)
        x_axis_data = x_axis * time_step
        # Plot the data on top of the figure
        if delta:
            JointPlot(x_axis_data, label_data, name1_data, name2_data, name1=name1, name2=name2, marginal_x=False, marginal_y=True, use_label=use_label, label_as=label_as, with_threshold=with_threshold, changed_index=changed_index).plot(axes[0])
        else:
            JointPlot(x_axis_data, label_data, name1_data, name2_data, name1=name1, name2=name2, marginal_x=False, marginal_y=True, use_label=use_label, label_as=label_as, with_threshold=with_threshold, changed_index=changed_index).plot(axes)
        if delta:
            JointPlot(x_axis_data, label_data, name1_data, name2_data, name1=name1, name2=name2, marginal_x=False, marginal_y=True, plot_data=False, use_label=use_label, label_as=label_as, with_threshold=with_threshold, changed_index=changed_index).plot(axes[1])

        _print_summary_statistics(name1, name2, name1_data, name2_data, label_data, label_as)
        name = file_path.split("/")[-1].split(".")[0].split("_")
        if "rotated" in name:
            name = "Rotated"
        elif "vertical" in name:
            name = "Vertical"
        elif "combined" in name:
            name = "Vertical_and_Rotated_Combined"
        # replace _ with space
        try:
            fig.suptitle(f"{name.replace('_', ' ')}", y=0.96, fontsize=25)
        except Exception as e:
            print(e)
        # Adjust vertical spacing between subplots
        plt.subplots_adjust(hspace=0.2)  # Reduce hspace as needed
        # plt.tight_layout(pad=1.0)  # Adjust padding as necessary
        # plt.savefig(f"reports/recording_{name.lower()}_{name1}_{name2}.pdf")
        plt.show()

if __name__ == '__main__':
    # file_path = "data/predictions/recording_rotated_all_frames_processed.hdf5"
    # file_path = "data/predictions/recording_rotated_all_frames_processed_combined.hdf5"
    # file_path = "data/predictions/recording_vertical_all_frames_processed_combined.hdf5"
    file_path = "data/predictions/recording_vertical_all_frames_processed_combined_processed.hdf5"

    # file_path = "data/predictions/recording_combined_all_frames_processed.hdf5"
    # file_path = "D:/HCAI/msc/strawml/data/interim/sensors_with_strawbbox.hdf5"
    # file_path = 'data/noisy_datasets/noisy_scratches_lens_flare.hdf5'
    # file_path = 'data/predictions/new_run/recording_vertical_all_frames_processed.hdf5'
    main(file_path, name="sensors", name1='convnext', name2='yolo', time_step=5, delta=False, use_label=True, label_as='fullness', with_threshold=True, iou=False)