from __init__ import *
from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("models/yolov8n.pt")
model.to("cuda")
if __name__ == '__main__':
    # train the model on our own dataset
    results = model.train(data="data/processed/digits_on_chute_data.yaml", imgsz=256, epochs=2, optimizer="adam")

    # save model
    model.save("models/yolov8n-digits-adam.pt")

    # results = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format