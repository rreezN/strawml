"""
What have we tried?

DATA WISE
1. Data without adding hyphens
2. Data with added hyphens
3. Data with one class only (digits) and added hyphens
4. Data with one class only (digits), added hyphens and each image has multiple digits
5. All of the above but now with the white backgrounds changed to off-white colors (not always the same color)
    - The complete white color might be too specific of a color and thus when it comes to finding the digits in the chute image, it might be difficult
6. resize the overlay images between 14x14 and 28x28 and random combinations of these sizes
    - also gamma correction of the background images

SAHI WISE
1. Using the YOLO model with a larger number of slices (not too large) is better to detect
the small squares. This could imply that the model is not able to detect multiple digits per image, why
we should try to train with multiple digits per image.
"""

from __init__ import *
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


# Load a pre-trained YOLOv10n model
model = YOLO("models/yolov8s.pt")
model.to("cuda")
wandb.init(entity='dtu-dnd')
add_wandb_callback(model, enable_model_checkpointing=True)


if __name__ == '__main__':
    # train the model on our own dataset
    results = model.train(project='test', data="data/processed/digits_on_chute_data.yaml", epochs=100, optimizer="AdamW")

    # save model
    model.save("models/yolov8s-digits-one-cls-multi-d-100e.pt")

    # results = model.val()  # evaluate model performance on the validation set
    # results = model("data/chute_digit.jpg")  # predict on an image
    # results = model("data/processed/yolo_format/train/frame_148.jpg")  # predict on a video
    # success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format
    # results[0].show()  # display results