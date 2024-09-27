from __init__ import *
import random
import cv2
import numpy as np
from strawml.models.chute_finder.yolo import ObjectDetect
from strawml.models.digit_finder.sahiyolo import SahiYolo
import time
import threading

class VideoStreamCustom:
    def __init__(self, model_name=None, object_detect=True, yolo_threshold=0.3, device='cuda', verbose=False, sahi=False) -> None:
        self.object_detect = object_detect
        self.yolo_threshold = yolo_threshold
        self.model_name = model_name
        if object_detect:
            if sahi:
                self.OD = SahiYolo(model_name, device=device, verbose=False)
            else:
                self.OD = ObjectDetect(model_name, yolo_threshold=yolo_threshold, device=device, verbose=False)


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        if 'obb' in self.model_name:
            labels, cord, labels_conf, angle_rad = results
        else:
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
                # plot a line with angle_rad as the angle of the object
                angle_sin = np.sin(angle_rad[i].detach().cpu().numpy())
                angle_cos = np.cos(angle_rad[i].detach().cpu().numpy())
                # plot the line from the center of the object
                x_center = (x1 + x2 + x3 + x4) // 4
                y_center = (y1 + y2 + y3 + y4) // 4
                cv2.line(frame, (x_center, y_center), (x_center + int(500 * angle_cos), y_center + int(500 * angle_sin)), (0, 0, 255), 2)
                cv2.line(frame, (x_center, y_center), (x_center - int(500 * angle_cos), y_center - int(500 * angle_sin)), (0, 0, 255), 2)

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

    def __call__(self, video):
        with open('data/hkvision_credentials.txt', 'r') as f:
            credentials = f.read().splitlines()
            username = credentials[0]
            password = credentials[1]
            ip = credentials[2]
            rtsp_port = credentials[3]

        while True:
            try:
                start_time = time.time()  # Record the start time

                success, image = video.read()
                if not success:
                    print("Warning: Failed to read frame from stream, skipping...")
                    time.sleep(0.1)  # Short delay before retrying
                    video.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
                    continue

                image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
                results = self.OD.score_frame(image)  # This takes a lot of time if ran on CPU
                # Plot boxes from YOLO on DeepFace output
                image = self.plot_boxes(results, image)
                cv2.imshow('Video', image)

                end_time = time.time()  # Record the end time
                frame_time = end_time - start_time
                # print(f"Time between frames: {frame_time:.4f} seconds")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        video.release()
        cv2.destroyAllWindows()