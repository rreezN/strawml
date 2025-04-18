from __init__ import *
import h5py
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import math

def retrieve_data(path):
    label_bbox_data = np.array([])
    label_fullness_data = np.array([])
    scada_data = np.array([])
    yolo_data = np.array([])
    convnext_data = np.array([])
    convnext_data_at = np.array([])
    errors = 0
    yolo_detection_errors = 0
    conv_detection_errors = 0
    with h5py.File(data_dir + path, 'r') as f:
        keys = list(f.keys())
        if "frame" == keys[0].split("_")[0]:
            keys = sorted(keys, key=lambda x: int(x.split('_')[1]))
        else:
            keys = sorted(keys, key=lambda x: float(x))
        for key in keys:
            try:
                bbox_data = f[key]['straw_percent_bbox']['percent'][...]
                fullness_data = f[key]['straw_percent_fullness']['percent'][...]
                scada_data_ = f[key]['scada']['percent'][...]
                yolo_data_ = f[key]['yolo']['percent'][...]
                convnext_data_ = f[key]['convnext']['percent'][...]
                # convnext_data_at_ = f[key]['convnext_apriltag_alone']['percent'][...]

                label_bbox_data = np.append(label_bbox_data, bbox_data)
                label_fullness_data = np.append(label_fullness_data, fullness_data)
                scada_data = np.append(scada_data, scada_data_)
                yolo_data = np.append(yolo_data, yolo_data_)
                convnext_data = np.append(convnext_data, convnext_data_)
                # convnext_data_at = np.append(convnext_data_at, convnext_data_at_)
            except KeyError as e:
                print(f"KeyError: {path}, {key}: {e}")
                errors += 1
    print(f"Errors: {errors}")
    print(f"Yolo Detection Errors: {yolo_detection_errors}")
    print(f"Conv Detection Errors: {conv_detection_errors}")
    return label_bbox_data, label_fullness_data, scada_data, yolo_data, convnext_data, convnext_data_at

def _get_accuracy_and_mae(label_data, prediction_data, percentage=10):
    # calculate accuracy and mean absolute error only where the label data is not nan and the prediction data is not nan
    mask = ~np.isnan(label_data) & ~np.isnan(prediction_data)
    frame_detection_accuracy  = np.sum(~np.isnan(prediction_data)) / len(prediction_data)
    label_data = label_data[mask]
    prediction_data = prediction_data[mask]
    accuracy = np.mean((prediction_data >= label_data - percentage) & (prediction_data <= label_data + percentage)) * 100
    mae = np.mean(np.abs(label_data - prediction_data))
    # round to 3 decimal places 
    return round(accuracy, 3), round(mae, 3), round(frame_detection_accuracy, 3)

def _run_extraction(data_dir, output):
# Get list of files in directory
    data_files  = os.listdir(data_dir)
    for path in data_files:
        label_bbox_data, label_fullness_data, scada_data, yolo_data, convnextv2_data, convnext_data_at = retrieve_data(path)
        # scada
        scada_bbox_accuracy, scada_bbox_mae, scada_bbox_frame_detection_accuracy = _get_accuracy_and_mae(label_bbox_data, scada_data)
        scada_fullness_accuracy, scada_fullness_mae, scada_fullness_frame_detection_accuracy = _get_accuracy_and_mae(label_fullness_data, scada_data)
        # yolo
        yolo_bbox_accuracy, yolo_bbox_mae, yolo_bbox_frame_detection_accuracy = _get_accuracy_and_mae(label_bbox_data, yolo_data)
        yolo_fullness_accuracy, yolo_fullness_mae, yolo_fullness_frame_detection_accuracy = _get_accuracy_and_mae(label_fullness_data, yolo_data)
        # convnextv2
        convnextv2_bbox_accuracy, convnextv2_bbox_mae, convnextv2_bbox_frame_detection_accuracy = _get_accuracy_and_mae(label_bbox_data, convnextv2_data)
        convnextv2_fullness_accuracy, convnextv2_fullness_mae, convnextv2_fullness_frame_detection_accuracy = _get_accuracy_and_mae(label_fullness_data, convnextv2_data)
        # # convnextv2 apriltag
        # convnextv2_bbox_accuracy_at, convnextv2_bbox_mae_at, convnextv2_bbox_frame_detection_accuracy_at = _get_accuracy_and_mae(label_bbox_data, convnext_data_at)
        # convnextv2_fullness_accuracy_at, convnextv2_fullness_mae_at, convnextv2_fullness_frame_detection_accuracy_at = _get_accuracy_and_mae(label_fullness_data, convnext_data_at)

        print("-------------------")
        print(f"File: {path}")
        print(f"Scada Bbox Accuracy: {scada_bbox_accuracy}, Scada Bbox MAE: {scada_bbox_mae}, Frame Detection Accuracy: {scada_bbox_frame_detection_accuracy}")
        print(f"Scada Fullness Accuracy: {scada_fullness_accuracy}, Scada Fullness MAE: {scada_fullness_mae}, Frame Detection Accuracy: {scada_fullness_frame_detection_accuracy}")
        print("       --:--       ")
        print(f"Yolo Bbox Accuracy: {yolo_bbox_accuracy}, Yolo Bbox MAE: {yolo_bbox_mae}, Frame Detection Accuracy: {yolo_bbox_frame_detection_accuracy}")
        print(f"Yolo Fullness Accuracy: {yolo_fullness_accuracy}, Yolo Fullness MAE: {yolo_fullness_mae}, Frame Detection Accuracy: {yolo_fullness_frame_detection_accuracy}")
        print("       --:--       ")
        print(f"Convnextv2 Bbox Accuracy: {convnextv2_bbox_accuracy}, Convnextv2 Bbox MAE: {convnextv2_bbox_mae}, Frame Detection Accuracy: {convnextv2_bbox_frame_detection_accuracy}")
        print(f"Convnextv2 Fullness Accuracy: {convnextv2_fullness_accuracy}, Convnextv2 Fullness MAE: {convnextv2_fullness_mae}, Frame Detection Accuracy: {convnextv2_fullness_frame_detection_accuracy}")
        # print("       --:--       ")
        # print(f"Convnextv2 Bbox Accuracy (Apriltag): {convnextv2_bbox_accuracy_at}, Convnextv2 Bbox MAE (Apriltag): {convnextv2_bbox_mae_at}, Frame Detection Accuracy (Apriltag): {convnextv2_bbox_frame_detection_accuracy_at}")
        # print(f"Convnextv2 Fullness Accuracy (Apriltag): {convnextv2_fullness_accuracy_at}, Convnextv2 Fullness MAE (Apriltag): {convnextv2_fullness_mae_at}, Frame Detection Accuracy (Apriltag): {convnextv2_fullness_frame_detection_accuracy_at}")
        print("-------------------")

        # append values to existing json file 
        save_dict[path] = {
                "scada_bbox_accuracy": scada_bbox_accuracy,
                "scada_bbox_mae": scada_bbox_mae,
                "scada_bbox_frame_detection_accuracy": scada_bbox_frame_detection_accuracy,
                "scada_fullness_accuracy": scada_fullness_accuracy,
                "scada_fullness_mae": scada_fullness_mae,
                "scada_fullness_frame_detection_accuracy": scada_fullness_frame_detection_accuracy,
                "yolo_bbox_accuracy": yolo_bbox_accuracy,
                "yolo_bbox_mae": yolo_bbox_mae,
                "yolo_bbox_frame_detection_accuracy": yolo_bbox_frame_detection_accuracy,
                "yolo_fullness_accuracy": yolo_fullness_accuracy,
                "yolo_fullness_mae": yolo_fullness_mae,
                "yolo_fullness_frame_detection_accuracy": yolo_fullness_frame_detection_accuracy,
                "convnextv2_bbox_accuracy": convnextv2_bbox_accuracy,
                "convnextv2_bbox_mae": convnextv2_bbox_mae,
                "convnextv2_bbox_frame_detection_accuracy": convnextv2_bbox_frame_detection_accuracy,
                "convnextv2_fullness_accuracy": convnextv2_fullness_accuracy,
                "convnextv2_fullness_mae": convnextv2_fullness_mae,
                "convnextv2_fullness_frame_detection_accuracy": convnextv2_fullness_frame_detection_accuracy,
                # "convnextv2_bbox_accuracy_at": convnextv2_bbox_accuracy_at,
                # "convnextv2_bbox_mae_at": convnextv2_bbox_mae_at,
                # "convnextv2_bbox_frame_detection_accuracy_at": convnextv2_bbox_frame_detection_accuracy_at,
                # "convnextv2_fullness_accuracy_at": convnextv2_fullness_accuracy_at,
                # "convnextv2_fullness_mae_at": convnextv2_fullness_mae_at,
                # "convnextv2_fullness_frame_detection_accuracy_at": convnextv2_fullness_frame_detection_accuracy_at
            }
    with open(output, 'w') as f:
        json.dump(save_dict, f)

def _adjust_text(x_axis, y_axis, string):
    from scipy import interpolate
    texts = []
    for x, y, s in zip(x_axis,y_axis,string):
        texts.append(plt.text(x, y, f'{s:.2f}'))
    f = interpolate.interp1d(x_axis,y_axis)
    x = x_axis
    y = f(x_axis)    
    adjust_text(texts, x=x, y=y, autoalign='y',
                only_move={'points':'y', 'text':'y'}, force_points=0.15,
                arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

def _plot_noisy_data(data, keys):
    accuracy_yolo = []
    mae_yolo = []
    accuracy_cn = []
    mae_cn = []
    accuracy_cn_at = []
    mae_cn_at = []
    for name in keys:
        accuracy_cn.append(data[name]['convnextv2_bbox_accuracy'])
        mae_cn.append(data[name]['convnextv2_bbox_mae'])
        accuracy_yolo.append(data[name]['yolo_bbox_accuracy'])
        mae_yolo.append(data[name]['yolo_bbox_mae'])
        accuracy_cn_at.append(data[name]['convnextv2_bbox_accuracy_at'])
        mae_cn_at.append(data[name]['convnextv2_bbox_mae_at'])
    # add values at the start
    accuracy_cn.insert(0, 92.783)
    mae_cn.insert(0, 4.639)
    accuracy_yolo.insert(0, 94.522)
    mae_yolo.insert(0, 2.919)
    accuracy_cn_at.insert(0, 54.348)
    mae_cn_at.insert(0, 12.804)

    # plot line plots for accuracy and mae for yolo and convnextv2 in the same plot
    # Create a figure and axis
    # plt.style.use('seaborn-darkgrid')

    fig, ax = plt.subplots(figsize=(16, 8))
    title_size = 20
    label_size = 18
    text_size = 16

    # Ensure gridlines are behind plot elements
    ax.set_axisbelow(True)
    ax.grid(zorder=0)
    
    x_axis = np.arange(0, 7)
    # Plot data
    ax.plot(x_axis, accuracy_yolo, label='Yolo Accuracy', color='royalblue', linestyle='-', zorder=2)
    ax.plot(x_axis, mae_yolo, label='Yolo MAE', color='royalblue', linestyle='--', zorder=2)
    ax.plot(x_axis, accuracy_cn, label='ConvNeXtv2 Accuracy', color='indianred', linestyle='-', zorder=2)
    ax.plot(x_axis, mae_cn, label='ConvNeXt MAE', color='indianred', linestyle='--', zorder=2)
    ax.plot(x_axis, accuracy_cn_at, label='ConvNeXt (Apriltag) Accuracy', color='mediumseagreen', linestyle='-', zorder=2)
    ax.plot(x_axis, mae_cn_at, label='ConvNeXt (Apriltag) MAE', color='mediumseagreen', linestyle='--', zorder=2)

    # Scatter points for all values
    ax.scatter(x_axis, accuracy_yolo, color='royalblue', zorder=3, s=50)
    ax.scatter(x_axis, accuracy_cn, color='indianred', zorder=3, s=50)
    ax.scatter(x_axis, mae_yolo, color='royalblue', zorder=3, s=50)
    ax.scatter(x_axis, mae_cn, color='indianred', zorder=3, s=50)
    ax.scatter(x_axis, accuracy_cn_at, color='mediumseagreen', zorder=3, s=50)
    ax.scatter(x_axis, mae_cn_at, color='mediumseagreen', zorder=3, s=50)
    # Add text annotations with dynamic placement logic for accuracy and MAE
    texts = []

    for x, accuracy_y, mae_y, accuracy_c, mae_c, accuracy_c_at, mae_c_at in zip(
        x_axis, accuracy_yolo, mae_yolo, accuracy_cn, mae_cn, accuracy_cn_at, mae_cn_at
    ):
        # Logic for accuracy annotations
        if accuracy_c < accuracy_y:
            accuracy_offset_c = -5  # Move ConvNeXtV2 annotation down
            accuracy_offset_y = 1   # Move YOLO annotation up
        else:
            accuracy_offset_y = -5  # Move YOLO annotation down
            accuracy_offset_c = 1   # Move ConvNeXtV2 annotation up

        if accuracy_c_at < accuracy_y:
            accuracy_offset_c_at = -5  # Move ConvNeXtV2 (AT) annotation down
        else:
            accuracy_offset_c_at = 1   # Move ConvNeXtV2 (AT) annotation up

        # Logic for MAE annotations
        if mae_c < mae_y:
            mae_offset_c = -5  # Move ConvNeXtV2 MAE annotation down
            mae_offset_y = 1   # Move YOLO MAE annotation up
        else:
            mae_offset_y = -5  # Move YOLO MAE annotation down
            mae_offset_c = 1   # Move ConvNeXtV2 MAE annotation up

        if mae_c_at < mae_y:
            mae_offset_c_at = -5  # Move ConvNeXtV2 (AT) MAE annotation down
        else:
            mae_offset_c_at = 1   # Move ConvNeXtV2 (AT) MAE annotation up

        # Add text for YOLO accuracy
        texts.append(plt.text(x, accuracy_y + accuracy_offset_y, f'{accuracy_y:.2f}', fontsize=text_size))
        # Add text for ConvNeXtV2 accuracy
        texts.append(plt.text(x, accuracy_c + accuracy_offset_c, f'{accuracy_c:.2f}', fontsize=text_size))
        # Add text for ConvNeXtV2 (AT) accuracy
        texts.append(plt.text(x, accuracy_c_at + accuracy_offset_c_at, f'{accuracy_c_at:.2f}', fontsize=text_size))

        # Add text for YOLO MAE
        texts.append(plt.text(x, mae_y + mae_offset_y, f'{mae_y:.2f}', fontsize=text_size))
        # Add text for ConvNeXtV2 MAE
        texts.append(plt.text(x, mae_c + mae_offset_c, f'{mae_c:.2f}', fontsize=text_size))
        # Add text for ConvNeXtV2 (AT) MAE
        texts.append(plt.text(x, mae_c_at + mae_offset_c_at, f'{mae_c_at:.2f}', fontsize=text_size))

    # Adjust text positions to avoid overlapping
    adjust_text(texts, ax=ax)


    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07),
            fancybox=True, shadow=True, ncol=3, fontsize=text_size)

    # Add labels and title
    ax.set_xlabel('#Random Augments', fontsize=label_size)
    ax.set_ylabel('Accuracy (%), MAE', fontsize=label_size)
    # ax.set_title("No. Random Augmentations vs. Accuracy", y=0.96, fontsize=title_size)
    fig.suptitle("No. Random Augmentations vs. Accuracy & MAE", y=0.98, fontsize=title_size)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_axis, fontsize=text_size)

    # Set the tick labels font size
    ax.set_yticklabels(ax.get_yticks(), fontsize=text_size)
    # Display the plot
    plt.tight_layout()
    plt.savefig('reports/figures/robustness_plot_n1_6.pdf')
    plt.show()

def _load_json(stat_path):
    # load data
    with open(stat_path, 'r') as f:
        data = json.load(f)
    
    # get all keys
    keys = list(data.keys())    
    _plot_noisy_data(data, keys[:6])


def _run_plotting(stat_path):
    _load_json(stat_path)

def find_largest_pair(A):
    """
    Find two integers x and y such that x * y >= A and both x and y are as large as possible.

    Parameters:
        A (int): The target value.

    Returns:
        tuple: A pair of integers (x, y).
    """
    # Start from the square root of A
    x = math.ceil(math.sqrt(A))
    # Calculate y as the smallest integer such that x * y >= A
    y = math.ceil(A / x)
    
    return x, y

def _load_csv_dir(directory):
    import pandas as pd
    files = os.listdir(directory)
    files =sorted(files, reverse=True)
    # create square figure as many as the number of files
    # divide the number of files with 2 to get the number of rows
    # check if the number of files is odd
    # if odd, add 1 to the number of rows and columns

    # Make sqrt check instead of this shit
    r, c = find_largest_pair(len(files))
    legends_added = False
    saved_legends = None
    fig, ax = plt.subplots(r, c, figsize=(15, 10))

    for i in range(r):
        for j in range(c):
            if i * c + j >= len(files):
                # remove axis and everything visible on the plot
                ax[i, j].set_axis_off()
                if not legends_added:
                    legends_added = True
                    handles, labels = saved_legends.legendHandles, [text.get_text() for text in saved_legends.texts]
                    # `ax` is the Axes object for the new plot
                    ax[i, j].legend(handles, labels, loc='center',
                            fancybox=True, shadow=True, ncol=3)
            else:
                file = files[i * c + j]
                df = pd.read_csv(directory + file)

                # Remove the columns with MIN and MAX in the name
                df = df.loc[30:, ~df.columns.str.contains('MIN|MAX')]

                # rename step to epoch
                df.rename(columns={'Step': 'Epoch'}, inplace=True)
                # Rename all headers to only contain everything before " - "
                df.columns = [col.split(' - ')[0] for col in df.columns]               

                # refactor dataframe for sns plot
                df_melted = pd.melt(df, id_vars=['Epoch'], value_vars=df.columns[1:], var_name='variable', value_name='value')
                sns.lineplot(data=df_melted, x='Epoch', y="value", hue="variable", ax=ax[i, j], sort=False)
                # add stuff
                ax[i, j].grid()
                name = file.split("_")
                name = name[0] + "/" + name[1] + "_" + name[2].split(".")[0]
                ax[i, j].title.set_text(name)
                saved_legends = ax[i, j].get_legend()
                ax[i, j].get_legend().remove()
                # set xticks to every int
                ax[i, j].set_xticks(df['Epoch'])
                ax[i, j].set_xticklabels(df['Epoch'], rotation=45)

    plt.tight_layout()
    plt.savefig('reports/figures/train_yolo_chute.pdf')
    plt.show()

def _run_model_metrics(data_dir):
    _load_csv_dir(data_dir)

def _plot_stats(stats):
    import pandas as pd
    keys = list(stats.keys())
    data = {'name': [], 'model': [], 'processor': [], 'fps': [],  'load_time': [], 'cutout_time':[], 'inference_time': [], 'postprocess_time': [], 'total_time': []}
    names = []
    for key in keys:
        # if apriltag in key, add apriltag to the name
        model = key.split('_')
        print(model)

        if 'apriltag' in model:
            model = model[0] + "_" + model[1]
        else:
            model = model[0]
        if key == 'conv_2060S':
            names += ['ConvNeXt (RTX 2060 Super)']
        elif key == 'yolo_2060S':
            names += ['YOLO (RTX 2060 Super)']
        elif key == 'conv_apriltag_2060S':
            names += ['ConvNeXt w. AprilTags (RTX 2060 Super)']
            
        elif key == 'conv_3050':
            names += ['ConvNeXt (RTX 3050)']
        elif key == 'yolo_3050':
            names += ['YOLO (RTX 3050)']
        elif key == 'conv_apriltag_3050':
            names += ['ConvNeXt w. AprilTags (RTX 3050)']
        
        elif key == 'conv_4070S':
            names += ['ConvNeXt (RTX 4070 Super)']
        elif key == 'yolo_4070S':
            names += ['YOLO (RTX 4070 Super)']
        elif key == 'conv_apriltag_4070S':
            names += ['ConvNeXt w. AprilTags (RTX 4070 Super)']
            
        elif key == 'conv_i5_12400F':
            names += ['ConvNeXt V2 (Intel i5 12400F)']
        elif key == 'yolo_i5_12400F':
            names += ['YOLO (Intel i5 12400F)']
        elif key == 'conv_apriltag_i5_12400F':
            names += ['ConvNeXt w. AprilTags (Intel i5 12400F)']
        
        elif key == 'conv_i5_5300U':
            names += ['ConvNeXt V2 (Intel i5 5300U)']
        elif key == 'yolo_i5_5300U':
            names += ['YOLO (Intel i5 5300U)']
        elif key == 'conv_apriltag_i5_5300U':
            names += ['ConvNeXt w. AprilTags (Intel i5 5300U)']
        
        elif key == 'conv_ryzen5_3600':
            names += ['ConvNeXt V2 (Ryzen 5 3600)']
        elif key == 'yolo_ryzen5_3600':
            names += ['YOLO (Ryzen 5 3600)']
        elif key == 'conv_apriltag_ryzen5':
            names += ['ConvNeXt w. AprilTags (Ryzen 5 3600)']
        
        data['name'].append(names[-1])
        data['model'].append(model)
        data['processor'].append(names[-1].split("(")[-1].split(")")[0])
        data['fps'].append(stats[key]['fps'])
        data['load_time'].append(stats[key]['load_time'])
        data['inference_time'].append(stats[key]['inference_time'])
        data['postprocess_time'].append(stats[key]['postprocess_time'])
        data['total_time'].append(stats[key]['total_time'])
        if 'cutout_time' in stats[key].keys():
            data['cutout_time'].append(stats[key]['cutout_time'])
        else:
            dummy_array = [np.nan]
            data['cutout_time'].append(dummy_array)

    title_size = 20
    label_size = 18
    text_size = 16

    # Assume data is already defined (e.g., from a file or other source)

    # === 1. Extract values for each group ===

    # For CONV (existing “conv” dataset)
    conv_idx = np.where(np.array(data['model']) == 'conv')
    pos_fps             = np.array(data['fps'])[conv_idx]
    pos_load_time       = np.array(data['load_time'])[conv_idx]
    pos_inference_time  = np.array(data['inference_time'])[conv_idx]
    pos_postprocess_time= np.array(data['postprocess_time'])[conv_idx]
    # Note: Here we use a list comprehension similar to your original code;
    # we assume the ordering of `data['cutout_time']` matches the ordering of models.
    pos_prep_time = np.array([
        x for i, sublist in enumerate(data['cutout_time']) 
        if data['model'][i] == 'conv'
        for x in (sublist if isinstance(sublist, list) else [sublist])
        if not np.isnan(x)
    ])

    # For YOLO (existing “yolo” dataset)
    yolo_idx = np.where(np.array(data['model']) == 'yolo')
    neg_fps             = np.array(data['fps'])[yolo_idx]
    neg_load_time       = np.array(data['load_time'])[yolo_idx]
    neg_inference_time  = np.array(data['inference_time'])[yolo_idx]
    neg_postprocess_time= np.array(data['postprocess_time'])[yolo_idx]
    # Using processors from YOLO as x-tick labels (assumed common across groups)
    genes = np.array(data['processor'])[yolo_idx]

    # For CONV AprilTag (new “conv_apriltag” dataset)
    apriltag_idx = np.where(np.array(data['model']) == 'conv_apriltag')
    apriltag_fps             = np.array(data['fps'])[apriltag_idx]
    apriltag_load_time       = np.array(data['load_time'])[apriltag_idx]
    apriltag_inference_time  = np.array(data['inference_time'])[apriltag_idx]
    apriltag_postprocess_time= np.array(data['postprocess_time'])[apriltag_idx]
    apriltag_prep_time = np.array([
        x for i, sublist in enumerate(data['cutout_time']) 
        if data['model'][i] == 'conv_apriltag'
        for x in (sublist if isinstance(sublist, list) else [sublist])
        if not np.isnan(x)
    ])
    apriltag_prep_time += np.array(data['total_time'])[apriltag_idx] - (apriltag_load_time + apriltag_inference_time + apriltag_postprocess_time + apriltag_prep_time)

    # === 2. Set up bar positions for three groups ===
    # (We assume that the length/order of entries is the same across groups.)
    n_groups = len(pos_load_time)  # number of x-tick categories (processors)
    indices = np.arange(n_groups)

    # Adjust the bar_width if needed (here we use a slightly smaller width so they don't overlap)
    bar_width = 0.25  
    epsilon = 0.015     # small gap to show borders
    line_width = 1
    opacity = 0.7
    text_size = 12      # adjust as desired

    # Define positions for each group:
    conv_positions     = indices - bar_width   # left group
    apriltag_positions = indices               # middle group
    yolo_positions     = indices + bar_width   # right group

    # === 3. Create the plot with three sets of bars ===
    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context("talk")
        
        # --- Plot CONV bars (existing) ---
        conv_load_bar = plt.bar(conv_positions, pos_load_time, bar_width,
                                color='indianred',
                                label='CONV load time')
        conv_prep_bar = plt.bar(conv_positions, pos_prep_time, bar_width-epsilon,
                                bottom=pos_load_time,
                                alpha=opacity,
                                color='white',
                                edgecolor='indianred',
                                linewidth=line_width,
                                hatch='X',
                                label='CONV preprocess time')
        conv_inference_bar = plt.bar(conv_positions, pos_inference_time, bar_width-epsilon,
                                bottom=pos_load_time + pos_prep_time,
                                alpha=opacity,
                                color='white',
                                edgecolor='indianred',
                                linewidth=line_width,
                                hatch='//',
                                label='CONV inference time')
        conv_post_bar = plt.bar(conv_positions, pos_postprocess_time, bar_width-epsilon,
                                bottom=pos_load_time + pos_prep_time + pos_inference_time,
                                alpha=opacity,
                                color='white',
                                edgecolor='indianred',
                                linewidth=line_width,
                                hatch='0',
                                label='CONV postprocess time')
        
        # --- Plot CONV AprilTag bars (new dataset) ---
        # Using a different color (e.g., 'darkorange') to differentiate
        apriltag_load_bar = plt.bar(apriltag_positions, apriltag_load_time, bar_width,
                                    color='darkorange',
                                    label='CONV AprilTag load time')
        apriltag_prep_bar = plt.bar(apriltag_positions, apriltag_prep_time, bar_width-epsilon,
                                    bottom=apriltag_load_time,
                                    alpha=opacity,
                                    color='white',
                                    edgecolor='darkorange',
                                    linewidth=line_width,
                                    hatch='X',
                                    label='CONV AprilTag preprocess time')
        apriltag_inference_bar = plt.bar(apriltag_positions, apriltag_inference_time, bar_width-epsilon,
                                        bottom=apriltag_load_time + apriltag_prep_time,
                                        alpha=opacity,
                                        color='white',
                                        edgecolor='darkorange',
                                        linewidth=line_width,
                                        hatch='//',
                                        label='CONV AprilTag inference time')
        apriltag_post_bar = plt.bar(apriltag_positions, apriltag_postprocess_time, bar_width-epsilon,
                                    bottom=apriltag_load_time + apriltag_prep_time + apriltag_inference_time,
                                    alpha=opacity,
                                    color='white',
                                    edgecolor='darkorange',
                                    linewidth=line_width,
                                    hatch='0',
                                    label='CONV AprilTag postprocess time')
        
        # --- Plot YOLO bars (existing) ---
        yolo_load_bar = plt.bar(yolo_positions, neg_load_time, bar_width,
                                color='royalblue',
                                label='YOLO load time')
        yolo_inference_bar = plt.bar(yolo_positions, neg_inference_time, bar_width-epsilon,
                                    bottom=neg_load_time,
                                    color="white",
                                    hatch='//',
                                    edgecolor='royalblue',
                                    linewidth=line_width,
                                    label='YOLO inference time')
        yolo_post_bar = plt.bar(yolo_positions, neg_postprocess_time, bar_width-epsilon,
                                bottom=neg_load_time + neg_inference_time,
                                color="white",
                                hatch='0',
                                edgecolor="royalblue",
                                linewidth=line_width,
                                label='YOLO postprocess time')
        
        # Assume conv_positions, apriltag_positions, and yolo_positions are arrays 
        # containing the x-coordinate (center) for each bar in each group.
        for i in range(n_groups):
            # Compute total times for each group
            conv_total = pos_load_time[i] + pos_prep_time[i] + pos_inference_time[i] + pos_postprocess_time[i]
            apriltag_total = apriltag_load_time[i] + apriltag_prep_time[i] + apriltag_inference_time[i] + apriltag_postprocess_time[i]
            yolo_total = neg_load_time[i] + neg_inference_time[i] + neg_postprocess_time[i]
            
            # Compute offsets based on the spacing between groups
            dx_left = (apriltag_positions[i] - conv_positions[i]) / 8
            dx_right = (yolo_positions[i] - apriltag_positions[i]) / 8
            
            # Place the text labels
            plt.text(conv_positions[i] - dx_left, conv_total, f"{conv_total:.2f}",
                    ha='center', va='bottom', fontsize=text_size)
            plt.text(apriltag_positions[i], apriltag_total, f"{apriltag_total:.2f}",
                    ha='center', va='bottom', fontsize=text_size)
            plt.text(yolo_positions[i] - dx_right, yolo_total, f"{yolo_total:.2f}",
                    ha='center', va='bottom', fontsize=text_size)

        
        # --- Set x-ticks (using the middle positions) and labels ---
        plt.xticks(apriltag_positions, genes, rotation=45)
        
        # Adjust legend to show only one entry per type (if needed)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                fancybox=True, shadow=True, ncol=3, fontsize=text_size)
        
        plt.ylabel('Time (s)', fontsize=label_size)
        plt.xlabel('Processor', fontsize=label_size)
        sns.despine()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()



    # # make up some fake data
    # pos_fps = np.array(data['fps'])[np.where(np.array(data['model']) == 'conv')]
    # pos_prep_time = np.array([x for sublist in data['cutout_time'] for x in (sublist if isinstance(sublist, list) else [sublist]) if not np.isnan(x)])
    # pos_load_time = np.array(data['load_time'])[np.where(np.array(data['model']) == 'conv')]
    # pos_inference_time = np.array(data['inference_time'])[np.where(np.array(data['model']) == 'conv')]
    # pos_postprocess_time = np.array(data['postprocess_time'])[np.where(np.array(data['model']) == 'conv')]

    # neg_fps = np.array(data['fps'])[np.where(np.array(data['model']) == 'yolo')]
    # neg_load_time = np.array(data['load_time'])[np.where(np.array(data['model']) == 'yolo')]
    # neg_inference_time = np.array(data['inference_time'])[np.where(np.array(data['model']) == 'yolo')]
    # neg_postprocess_time = np.array(data['postprocess_time'])[np.where(np.array(data['model']) == 'yolo')]
    # genes = np.array(data['processor'])[np.where(np.array(data['model']) == 'yolo')]
    
    # with sns.axes_style("white"):
    #     sns.set_style("ticks")
    #     sns.set_context("talk")
        
    #     # plot details
    #     bar_width = 0.35
    #     epsilon = .015
    #     line_width = 1
    #     opacity = 0.7
    #     pos_bar_positions = np.arange(len(pos_load_time))
    #     neg_bar_positions = pos_bar_positions + bar_width

    #     # make bar plots
    #     hpv_pos_mut_bar = plt.bar(pos_bar_positions, pos_load_time, bar_width,
    #                             color='indianred',
    #                             label='CONV load time')
        
    #     hpv_pos_prep = plt.bar(pos_bar_positions, pos_prep_time, bar_width-epsilon,
    #                             bottom=pos_load_time,
    #                             alpha=opacity,
    #                             color='white',
    #                             edgecolor='indianred',
    #                             linewidth=line_width,
    #                             hatch='X',
    #                             label='CONV preprocess time')

    #     hpv_pos_cna_bar = plt.bar(pos_bar_positions, pos_inference_time, bar_width-epsilon,
    #                             bottom=pos_load_time + pos_prep_time,
    #                             alpha=opacity,
    #                             color='white',
    #                             edgecolor='indianred',
    #                             linewidth=line_width,
    #                             hatch='//',
    #                             label='CONV inference time')

    #     hpv_pos_both_bar = plt.bar(pos_bar_positions, pos_postprocess_time, bar_width-epsilon,
    #                             bottom=pos_inference_time + pos_prep_time + pos_load_time,
    #                             alpha=opacity,
    #                             color='white',
    #                             edgecolor='indianred',
    #                             linewidth=line_width,
    #                             hatch='0',
    #                             label='CONV postprocess time')

    #     hpv_neg_mut_bar = plt.bar(neg_bar_positions, neg_load_time, bar_width,
    #                             color='royalblue',
    #                             label='YOLO load time')
    #     hpv_neg_cna_bar = plt.bar(neg_bar_positions, neg_inference_time, bar_width-epsilon,
    #                             bottom=neg_load_time,
    #                             color="white",
    #                             hatch='//',
    #                             edgecolor='royalblue',
    #                             ecolor="royalblue",
    #                             linewidth=line_width,
    #                             label='YOLO inference time')
    #     hpv_neg_both_bar = plt.bar(neg_bar_positions, neg_postprocess_time, bar_width-epsilon,
    #                             bottom=neg_inference_time+neg_load_time,
    #                             color="white",
    #                             hatch='0',
    #                             edgecolor='royalblue',
    #                             ecolor="royalblue",
    #                             linewidth=line_width,
    #                             label='YOLO postprocess time')
        
    #     # now we plot the total time above the bars for each processor and model type
    #     for i in range(len(pos_load_time)):
    #         # conv and yolo time values
    #         conv_time = pos_load_time[i] + pos_prep_time[i] +pos_inference_time[i] + pos_postprocess_time[i]
    #         yolo_time = neg_load_time[i] + neg_inference_time[i] + neg_postprocess_time[i]
    #         # slightly above the bar and to the left for conv and to the right for yolo
    #         dx = (neg_bar_positions[i] - pos_bar_positions[i])/4
    #         # dx = 0
    #         plt.text(pos_bar_positions[i]-dx, conv_time, f"{conv_time:.2f}", fontsize=text_size)
    #         plt.text(neg_bar_positions[i]-dx, yolo_time, f"{yolo_time:.2f}", fontsize=text_size)


    #     plt.xticks((neg_bar_positions+pos_bar_positions)/2, genes, rotation=45)
    #     plt.legend(loc='upper center', bbox_to_anchor=(0.3, 1.07),
    #             fancybox=True, shadow=True, ncol=2, fontsize=text_size)        # plt.tight_layout()
    #     plt.ylabel('Time (s)', fontsize=text_size)
    #     plt.xlabel('Processor', fontsize=text_size)
    #     # plt.yscale('log')

    #     sns.despine()
    #     plt.grid(axis='y')
    #     plt.show()  


def _run_fps_metrics(data_dir: str = 'data/fps/'):
    import pickle
    files = os.listdir(data_dir)
    stats = {}
    for file in files:
        # load pickle file
        means = {}

        with open(data_dir + file, 'rb') as f:
            data = pickle.load(f)
        # take the mean of all values in the dict
        print(f'\n{"_".join(file.split("_")[:-3])}')
        # print(data['inference_time'])
        for key, val in data.items():
            if key not in means.keys():
                means[key] = np.nanmean(val[1:])
                print(f"{key}: {means[key]:.5f}")
        stats["_".join(file.split("_")[:-3])] = means

    _plot_stats(stats)


if __name__ == '__main__':
    save_dict = {}
    # data_dir = 'data/train_data/yolo_chute/'
    data_dir = 'data/predictions/new_run/'
    # data_dir = 'data/noisy_datasets/'
    output = 'data/robustness_results_new.json'
    # _run_extraction(data_dir, output)
    # _run_plotting(output)
    # _run_model_metrics(data_dir)
    _run_fps_metrics()