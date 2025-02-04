#!/usr/bin/env python
from __init__ import *
import wandb
import os
import collections
import random
from ultralytics import YOLO
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace

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

def test(model, test_dataset_yaml, run_id):
    """
    Test the trained model on a different dataset.
    """
    # Run the test
    metrics = model.val(data=test_dataset_yaml, project="straw_project", name=run_id)
    with wandb.init(id=run_id, resume="allow") as run:
        name = run.name
        project = run.project
        wandb.log({f"test/mAP50-95(B)": metrics.box.map})
        wandb.log({f"test/mAP50(B)": metrics.box.map50})
    
    save_path = f"{project}/{run_id}/"
    print(f"Test results saved to {save_path}")
    # delete the files locally after testing
    os.system(f"rm -r {save_path}")

def train(sweep_data, dataset_yaml):
    model = YOLO("models/yolo11s-obb.pt")

    run_name = "{}-{}".format(sweep_data.sweep_run_name, sweep_data.num)
    config = sweep_data.config

    run = wandb.init(
        group=sweep_data.sweep_id,
        job_type="train",
        name=run_name,
        config=config,
        reinit=True
    )
    run_id = run.id
    model.train(
        data=dataset_yaml,
        imgsz=config["imgsz"],
        epochs=30,
        batch=config["batch_size"],
        lr0=config["lr"],
        optimizer=config["optimizer"], 
        project="straw_project",
        name = run_name
    )
    save_path = f"straw_project/{run_name}/"
    run.finish()
    return model, run_id, save_path  # Return trained model


def main(data_path, test_data_yaml, n_folds, id):
    """
    Main function for running the sweep and performing k-fold cross-validation.
    """
    num_folds = min(n_folds, 4)

    # Initialize wandb sweep
    sweep_id = f"meliora/straw_project/{id}"

    # Define the agent function
    def agent_function():
        ds_yamls = sorted(data_path.rglob("*.yaml"))[:num_folds]
        sweep_run = wandb.init()
        sweep_id = sweep_run.sweep_id or "unknown"
        sweep_url = sweep_run.get_sweep_url()
        project_url = sweep_run.get_project_url()
        sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
        sweep_run.notes = sweep_group_url
        sweep_run.save()
        sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
        sweep_run_id = sweep_run.id
        sweep_run_project = sweep_run.project or "unknown"
        sweep_run.finish()
        os.system(f"rm -r {sweep_run_project}/{sweep_run_name}/")
        wandb.sdk.wandb_setup._setup(_reset=True)
        
        trained_models = []
        for num in range(num_folds):
            reset_wandb_env()
            sweep_data = SweepData(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
            trained_model, run_id, save_path= train(sweep_data, ds_yamls[num])
            test(trained_model, test_data_yaml, run_id)
            # remove the files locally after testing
            os.system(f"rm -r {save_path}")
            # remove the files from wandb storage
        sweep_run.finish()
        os.system(f"wandb sync --clean-force --clean-old-hours 0")
    # Launch the sweep agent
    wandb.agent(sweep_id, function=agent_function)

def get_args() -> Namespace:
    # Create the parser
    parser = ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('--path', type=str, default="/work3/s194247/yolo_format_bbox_straw_whole_4fold")
    parser.add_argument('--test_data_yaml', type=str, default="/work3/s194247/test_data_set.yaml")
    parser.add_argument('--n_folds', type=int, default=4)
    parser.add_argument('--id', type=str, default="6eg448ia")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_path = Path(args.path)
    main(data_path = data_path, test_data_yaml=args.test_data_yaml, n_folds=args.n_folds, id=args.id)
