from __init__ import *
from ultralytics import YOLO
import cv2
import h5py
from strawml.data.make_dataset import decode_binary_image

# Load a pre-trained YOLOv10n model
model = YOLO("models/yolov8n-obb_1_cuda.pt")

if __name__ == '__main__':
    # train the model on our own dataset
    # results = model.train(data="data/interim/chute_data.yaml", epochs=2, imgsz=640)

    # save model
    # model.save("models/yolov8n-obb_1_cuda.pt")

    # Perform object detection on an image
    # load the image from disk and convert it to RGB (OpenCV ordering)
    hf = h5py.File('data/interim/chute_detection.hdf5', 'r')
    # # grab a random frame from the file
    frame = hf["frame_0"]['image'][...]
    frame = decode_binary_image(frame)
    # convert frame to rgb
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # frame = cv2.imread('data/boay.jpg')

    results = model.predict(source=frame, save=True, save_txt=True)  # save predictions as labels
# # Display the results
# # View results
    results[0].show()