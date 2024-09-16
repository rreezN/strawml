from __init__ import *
import random
import cv2
import numpy as np
from strawml.models.chute_finder.OD import ObjectDetect

class VideoStreamCustom:
    def __init__(self, model_name=None, object_detect=True, yolo_threshold=0.3) -> None:
        self.object_detect = object_detect
        self.yolo_threshold = yolo_threshold
        self.model_name = model_name
        if object_detect:
            self.OD = ObjectDetect(model_name, yolo_threshold=yolo_threshold)
            self.color = [''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(len(self.OD.classes))]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord, labels_conf = results
        n = len(labels)

        for i in range(n):  
            # plot polygon around the object based on the coordinates cord
            if 'obb' in self.model_name:
                x1, y1, x2, y2, x3, y3, x4, y4 = cord[i].flatten()
                x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
                # draw lines between the corners
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 0), 2)
                cv2.line(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.line(frame, (x4, y4), (x1, y1), (0, 255, 0), 2)
                # plot label on the object
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                x1, y1, x2, y2 = cord[i].flatten()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # plot label on the object
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.OD.classes[int(x)]
    
    def __call__(self, src: int = 0):
        video = cv2.VideoCapture(src)
        while True:
            ret, image = video.read()
            # flip the image
            image = cv2.flip(image, 1)
            if ret and self.object_detect:
                results = self.OD.score_frame(image) # This takes a lot of time if ran on CPU
                # Plot boxes from YOLO on DeepFace output
                image = self.plot_boxes(results, image)
                cv2.imshow('Video', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
    