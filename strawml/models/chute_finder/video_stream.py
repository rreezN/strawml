from __init__ import *
import random
import cv2
import numpy as np
from strawml.models.chute_finder.yolo import ObjectDetect
from strawml.models.digit_finder.sahiyolo import SahiYolo
from strawml.models.straw_classifier import chute_cropper as cc
import time
import threading
import torch
import timm
import h5py
from torchvision import transforms

class VideoStreamCustom:
    def __init__(self, model_name=None, object_detect=True, yolo_threshold=0.3, device='cpu', verbose=False, sahi=False) -> None:
        self.object_detect = object_detect
        self.yolo_threshold = yolo_threshold
        self.model_name = model_name
        if object_detect:
            if sahi:
                self.OD = SahiYolo(model_name, device=device, verbose=False)
            else:
                self.OD = ObjectDetect(model_name, yolo_threshold=yolo_threshold, device=device, verbose=False)
        self.camera_params = self.load_camera_params()
        self.tag_connections = [(11, 12), (12, 13), (13, 14), (14, 19), (19, 20), (20, 21), 
                                (21, 22), (22, 26), (26, 25), (25, 24), (24, 23), (23, 18), 
                                (18, 17), (17, 16), (16, 15), (15, 11)]
        self.mean, self.std = self.load_normalisation_constants()
        self.transform = transforms.Normalize(mean=self.mean, std=self.std)
        self.resize = transforms.Resize((384, 384))

    def load_normalisation_constants(self):
        # Loads the normalisation constants from 
        with h5py.File("data/processed/augmented/chute_detection.hdf5", 'r') as f:
            mean = f.attrs['mean']
            std = f.attrs['std']
        return mean, std
    
    def load_camera_params(self):
        # open npz file
        with np.load("fiducial_marker/calibration.npz") as data:
            cameraMatrix = data['cameraMatrix']
            distCoeffs = data['distCoeffs']
            rvecs = data['rvecs']
            tvecs = data['tvecs']
        return {"cameraMatrix": cameraMatrix, "distCoeffs": distCoeffs, "rvecs": rvecs, "tvecs": tvecs}

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

           
    def fix_frame(self, frame: np.ndarray, blur: bool = False, balance: int = 1) -> np.ndarray:
        """
        Fix the frame by undistorting it using the camera parameters.

        Params:
        -------
        frame: np.ndarray
            The frame to be undistorted
        blur: bool
            A boolean flag to indicate if the frame is to be blurred
        balance: int
            The balance factor to be used for undistortion
        
        Returns:
        --------
        np.ndarray
            The undistorted frame
        """
        K, D = self.camera_params["cameraMatrix"], self.camera_params["distCoeffs"]
        h,  w = frame.shape[:2]
        new_K = K.copy()
        new_K[0,0] *= balance  # Scale fx
        new_K[1,1] *= balance  # Scale fy
            
        undistorted_image = cv2.fisheye.undistortImage(frame, K, D, Knew=new_K,new_size=(w,h))
        if blur:
            image = cv2.GaussianBlur(frame, (7, 7), 0.7) 
            return image.astype(np.uint8)
        return undistorted_image 

    def prepare_for_inference(self, frame, results):
        # rotate and crop the frame to the chute bbox
        bbox_chute = results[1].flatten().cpu().detach().numpy() # x1, y1, x2, y2, x3, y3, x4, y4
        # bbox = [bbox_chute[6], bbox_chute[7], bbox_chute[0], bbox_chute[1], bbox_chute[2], bbox_chute[3], bbox_chute[4], bbox_chute[5]]
        frame_data, bbox_chute = cc.rotate_and_crop_to_bbox(frame, bbox_chute)
        # get edge features
        edges = cv2.Canny(frame_data, 100, 200)
        edges = edges.reshape(1, edges.shape[0], edges.shape[1])
        edges = torch.from_numpy(edges)/255
        # normalise with stats saved
        frame_data = self.transform(torch.from_numpy(frame_data).permute(2, 0, 1).float())
        # stack the images together
        cutout_image = np.concatenate((frame_data, edges), axis=0)
        # reshape to 4, 384, 384
        cutout_image = self.resize(torch.from_numpy(cutout_image))
        cutout_image = cutout_image.unsqueeze(0)
        return cutout_image

    def __call__(self, video):
        with open('data/hkvision_credentials.txt', 'r') as f:
            credentials = f.read().splitlines()
            username = credentials[0]
            password = credentials[1]
            ip = credentials[2]
            rtsp_port = credentials[3]

        model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', pretrained=False, in_chans=4, num_classes=11)            
        model.load_state_dict(torch.load("models/vit_classifier_best.pth"))
        frame_count = 0
        start_time = time.time()
        while True:
            try:
                success, image = video.read()
                if not success:
                    print("Warning: Failed to read frame from stream, skipping...")
                    time.sleep(0.1)  # Short delay before retrying
                    video.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
                    continue
                image = self.fix_frame(image)
                # image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
                results = self.OD.score_frame(image)  # This takes a lot of time if ran on CPU
                cutout_image = self.prepare_for_inference(image, results)
                output = model(cutout_image)
                _, predicted = torch.max(output, 1)
                # write the predicted value in the image
                cv2.putText(image, f'Straw Level: {predicted[0]*10:.2f} %', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                image = self.plot_boxes(results, image)
                image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                # write the fps in the image
                cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Video', image)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        video.release()
        cv2.destroyAllWindows()