from __init__ import *
from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("models/yolov8n-obb.pt")
model.to("cuda")
if __name__ == '__main__':
    # train the model on our own dataset
    # results = model.train(data="data/processed/chute_data.yaml", imgsz=256, epochs=10)

    # save model
    # model.save("models/yolov8n-obb-s.pt")

    # results = model.val()  # evaluate model performance on the validation set
    # results = model("data/chute_digit.jpg")  # predict on an image
    # results = model("data/processed/yolo_format/train/frame_148.jpg")  # predict on a video
    # success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format
    # results[0].show()  # display results