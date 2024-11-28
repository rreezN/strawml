import os
import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

# Define sweep configuration
sweep_configuration = {
    "method": "bayes",
    "name": "chute_sweep",
    "metric": {"goal": "minimize", "name": "metrics/mAP50-95(B)"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "lr": {"max": 0.01, "min": 0.00001},
        "imgsz": {"values": [256, 512, 640]},
        "optimizer": {"values": ["AdamW"]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="chute-sweep")
# sweep_id = 'meliora/chute-sweep/9hutzgjm'

# Define the sweep function
def train_sweep():
    # Initialize a new wandb run
    with wandb.init() as run:
        # Load a pre-trained YOLO model
        model = YOLO("models/yolo11s-obb.pt")
        model.to("cuda")

        # Get the sweep configuration parameters
        config = wandb.config

        # Set up training parameters based on the sweep configuration
        model.train(
            data="data/processed/chute_data.yaml",
            imgsz=512,
            epochs=50,
            batch=config.batch_size,
            lr0=config.lr,
            optimizer=config.optimizer,
            device="cuda",
            augment=True,
        )

# Run the sweep agent to execute the sweep
if __name__ == '__main__':
    wandb.agent(sweep_id, function=train_sweep, count=30)  # Set count or leave empty for infinite runs