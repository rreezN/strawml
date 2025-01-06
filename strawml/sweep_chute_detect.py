import os
import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

# Define sweep configuration
sweep_configuration = {
    "method": "bayes",
    "name": "straw_sweep",
    "metric": {"goal": "maximize", "name": "metrics/mAP50-95(B)"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64]},
        "lr": {"max": 0.01, "min": 0.000001},
        "imgsz": {"values": [256, 512, 640]},
        "optimizer": {"values": ["Adam", "AdamW", "SGD"]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="yolo-sweep-v2")
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
            data="data/processed/0_straw_data.yaml",
            imgsz=config.imgsz,
            epochs=50,
            batch=config.batch_size,
            lr0=config.lr,
            optimizer=config.optimizer,
            device="cuda",
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=45.0,
            translate=0.1,
            scale=0.5,
            shear=10.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            bgr=0.0,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.4,
            crop_fraction=1
        )

# Run the sweep agent to execute the sweep
if __name__ == '__main__':
    wandb.agent(sweep_id, function=train_sweep, count=60)  # Set count or leave empty for infinite runs