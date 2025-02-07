from __init__ import *
import data.dataloader as dl
from strawml.models.chute_finder.yolo import ObjectDetect
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO("models/obb_cutout_best.pt")
    # run model on one image
    # img = cv2.imread()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # model.predict("data/processed/yolo_format_bbox_sensors/val/0.jpg", save=True, imgsz=608)
    # results.show()
    validation_results = model.val(data="data/processed/test_data_set_chute.yaml", imgsz=608, batch=1, max_det=1000)

    print(validation_results)

    