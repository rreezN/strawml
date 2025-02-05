
#%% Load libraries
from __init__ import *
from pathlib import Path
import yaml
import sklearn
import pandas as pd
from collections import Counter

#%% Load data and labels
dataset_path = Path("D:/HCAI/msc/strawml/data/processed/yolo_format_bbox_straw_whole")  # replace with 'path/to/dataset' for your custom data
labels = sorted(dataset_path.rglob("*.txt"))  # all data in 'labels'

sorted_label_nums = sorted([int(i.__str__().split("\\")[-1].split(".")[0]) for i in labels])
# run thorugh label_nums to check if there are missing labels
for i in range(1, len(sorted_label_nums)):
    if sorted_label_nums[i] - sorted_label_nums[i - 1] != 1:
        # create empty label file for missing label
        for i in range(sorted_label_nums[i - 1] + 1, sorted_label_nums[i]):
            print(dataset_path / f"{i}.txt")
            with open(dataset_path / f"{i}.txt", "w") as lf:
                lf.write("")

labels = sorted(dataset_path.rglob("*.txt"))  # all data in 'labels'

#%% Load classes and create a DataFrame with label counts
yaml_file = "D:/HCAI/msc/strawml/data/processed/0_straw_data_whole.yaml"  # your data YAML with data directories and names dictionary
with open(yaml_file, "r", encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = sorted(classes.keys())

index = [label.stem for label in labels]  # uses base filename as ID (no extension)
labels_df = pd.DataFrame([], columns=cls_idx, index=index)

for label in labels:
    lbl_counter = Counter()

    with open(label, "r") as lf:
        lines = lf.readlines()

    for line in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(float(line.split(" ")[0]))] += 1

    labels_df.loc[label.stem] = lbl_counter

labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

#%% Split data into k-folds
from sklearn.model_selection import KFold

ksplit = 4
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

kfolds = list(kf.split(labels_df))

folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=index, columns=folds)

for i, (train, val) in enumerate(kfolds, start=1):
    folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
    folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

#%% create the directories and dataset YAML files for each split
import datetime

# supported_extensions = [".jpg", ".jpeg", ".png"]

# Initialize an empty list to store image file paths
images = sorted(dataset_path.rglob(f"*.jpg"))

#%% Create the necessary directories and dataset YAML files (unchanged)
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

#%% Copy images and labels to new directories
for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "train",
                "val": "val",
                "names": classes,
            },
            ds_y,
        )

import shutil
for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / k_split / "images"
        lbl_to_path = save_path / split / k_split / "labels"

        # Copy image and label files to new directory (SamefileError if file already exists)
        # first we check if label file exists otherwise create empty txt file for the image
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)

folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

# %%
