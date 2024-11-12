import os
import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

# Define sweep configuration
sweep_configuration = {
    "method": "bayes",
    "name": "chute_sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "lr": {"max": 0.1, "min": 0.0001},
        "imgsz": {"values": [256, 512, 640]},
        "optimizer": {"values": ["SGD", "Adam", "AdamW", "ADOPT"]},
    },
}

# Check if sweep_id is set in environment variables
sweep_id = os.getenv("WANDB_SWEEP_ID")

if not sweep_id:
    # Initialize the sweep in wandb and store the sweep_id in an environment variable
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="chute-sweep")
    os.environ["WANDB_SWEEP_ID"] = sweep_id

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
        results = model.train(
            data="data/processed/chute_data.yaml",
            imgsz=config.imgsz,
            epochs=50,
            batch=config.batch_size,
            lr0=config.lr,
            optimizer=config.optimizer,
            device="cuda",
            augment=True,
        )

        # Log results
        wandb.log({"val_acc": results.metrics["accuracy"]})  # Log accuracy or any other metric

# Run the sweep agent to execute the sweep
if __name__ == '__main__':
    wandb.agent(sweep_id, function=train_sweep, count=20)  # Set count or leave empty for infinite runs