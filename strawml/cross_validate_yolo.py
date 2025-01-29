#!/usr/bin/env python
from __init__ import *
import wandb
import os
import collections
import random
from ultralytics import YOLO
import pickle
from pathlib import Path

os.environ['WANDB_TIMEOUT'] = '60'

SweepData = collections.namedtuple(
    "SweepData", ("num", "sweep_id", "sweep_run_name", "config")
)

# # Define the sweep configuration
# sweep_configuration = {
#     "method": "bayes",
#     "name": "straw_sweep",
#     "metric": {"goal": "maximize", "name": "metrics/mAP50-95(B)"},
#     "parameters": {
#         "batch_size": {"values": [8, 12, 16, 24]},
#         "lr": {"max": 0.01, "min": 0.000001},
#         "imgsz": {"values": [256, 512, 640]},
#         "optimizer": {"values": ["Adam", "AdamW", "SGD"]},
#     },
# }


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def train(sweep_data, dataset_yaml):
    model = YOLO("models/yolo11s-obb.pt")

    run_name = "{}-{}".format(sweep_data.sweep_run_name, sweep_data.num)
    config = sweep_data.config
    run = wandb.init(
        group=sweep_data.sweep_id,
        job_type=sweep_data.sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )
    model.train(data=dataset_yaml, imgsz=config["imgsz"], epochs=5, batch=config["batch_size"], lr0=config["lr"], optimizer=config["optimizer"])
    run.finish()
    return model.metrics


def main(data_path, n_folds):
    """
    Main function for running the sweep and performing k-fold cross-validation.
    """
    num_folds = min(n_folds, 5)

    # Initialize wandb sweep
    sweep_id = "meliora/straw_project/872t9e6k"

    # Define the agent function
    def agent_function():
        ds_yamls = sorted(data_path.rglob("*.yaml"))[:num_folds]
        print(ds_yamls)
        sweep_run = wandb.init()
        sweep_id = sweep_run.sweep_id or "unknown"
        sweep_url = sweep_run.get_sweep_url()
        project_url = sweep_run.get_project_url()
        sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
        sweep_run.notes = sweep_group_url
        sweep_run.save()
        sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
        sweep_run_id = sweep_run.id
        sweep_run.finish()
        wandb.sdk.wandb_setup._setup(_reset=True)
        metrics = {}

        for num in range(num_folds):
            reset_wandb_env()
            metrics[num] = train(
                SweepData(
                    sweep_id=sweep_id,
                    num=num,
                    sweep_run_name=sweep_run_name,
                    config=dict(sweep_run.config),
                ),
                ds_yamls[num],
            )

        # Save the metrics for this sweep run
        with open(f"metrics_{sweep_run.id}.pkl", "wb") as f:
            pickle.dump(metrics, f)
        sweep_run.finish()

    # Launch the sweep agent
    wandb.agent(sweep_id, function=agent_function)


if __name__ == "__main__":
    data_path = Path(
        "D:/HCAI/msc/strawml/data/processed/yolo_format_bbox_straw"
    )
    n_folds = 2
    main(data_path, n_folds=n_folds)
