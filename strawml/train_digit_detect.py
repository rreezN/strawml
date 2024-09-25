"""
This script is used to train the digit detection model. Importantly we use the Ultralytics YOLOv8 model
as the base model, and HWD+ dataset as the data we train and test on. The model is saved as a .pt file.

Inspiration while writing the code for this projec t is taken from: 
    
    https://github.com/thawro/yolov8-digits-detection?tab=readme-ov-file

"""
from __init__ import *
from ultralytics import YOLO

def export_model(ckpt_path: str, imgsz: tuple[int, int] = (256, 256), format: str = "onnx"):
    model = YOLO(ckpt_path)
    model.export(format=format, imgsz=imgsz)


def train_model(
    ckpt_path: str = "yolov8n-obb.pt",
    data_path: str = "yolo_HWD+.yaml",
    epochs: int = 100,
    imgsz: tuple[int, int] = (256, 256),
    batch: int = 128,
    train: bool = True,
    export: bool = True,
):
    if train:
        model = YOLO(ckpt_path)
        model.train(data=data_path, epochs=epochs, imgsz=imgsz, device=0, batch=batch)
    if export:
        export_model(ckpt_path, imgsz)


if __name__ == "__main__":
    CKPT_PATH = "models/yolov8n-obb.pt"
    EPOCHS = 100
    IMGSZ = (256, 256)
    BATCH = 128
    train_model(
        ckpt_path=CKPT_PATH, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, train=False, export=True
    )