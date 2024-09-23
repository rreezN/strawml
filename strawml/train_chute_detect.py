from __init__ import *
from ultralytics import YOLO

# Initialize model
model = YOLO("models\yolov8n-obb.pt")


if __name__ == '__main__':
    # Use the model
    results = model.train(data="data/processed/chute_data.yaml", epochs=100)
    model.save("models\yolov8n-obb-chute.pt")
    # results = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format