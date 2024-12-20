from __init__ import *
import data.dataloader as dl
from strawml.models.chute_finder.yolo import ObjectDetect
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("models/obb_best.pt")
    
    validation_results = model.val(data="data/processed/sensors.yaml", imgsz=512, batch=8, conf=0.25, iou=0.6, device="0")

    print(validation_results)

    