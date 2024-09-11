# TODO: Save data continuously to a video file to account for memory limitations
import threading
import time
import cv2
import os
from tqdm import tqdm

# Define video capture class
class VideoCaptureAsync:
    def __init__(self, src=0, imgsz=None, driver=None):
        self.src = src
        if driver is None:
            self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src, driver)
            
        if type(imgsz) is tuple:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgsz[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgsz[1])
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None
        
    def get(self, var1):
        return self.cap.get(var1)

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()
    
    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
