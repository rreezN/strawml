from __init__ import *
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


# Load a pre-trained YOLOv10s model
model = YOLO("models/yolo11s-obb.pt")
model.to("cuda")
# wandb.init(project="yolov-final-runs", entity='meliora')
# add_wandb_callback(model)
# Load a pre-trained YOLOv10n model
# model.to("cuda")
if __name__ == '__main__':
    # train the model on our own dataset
    results = model.train(
            data="data/processed/0_chute_data.yaml",
            imgsz=579,
            epochs=300,
            batch=32,
            lr0=0.0021696499235375,
            optimizer="Adam",
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
            crop_fraction=1,
            project="yolov-final-runs")
    # results = model.val()  # evaluate model performance on the validation set
    # results = model("data/chute_digit.jpg")  # predict on an image
    # results = model("data/processed/yolo_format/train/frame_0.jpg")  # predict on a video
    # success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format
    # results[0].show()  # display results