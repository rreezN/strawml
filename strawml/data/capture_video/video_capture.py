# TODO: Save data continuously to a video file to account for memory limitations
import threading
import time
import cv2
import os
from tqdm import tqdm

# Define video capture class
class VideoCaptureAsync:
    def __init__(self, duration, fps, src=0, driver=None, data_path='data/raw/videos', byte_limit=1e9):
        self.src = src
        if driver is None:
            self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src, driver)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.duration = duration
        self.fps = fps
        self.data_path = data_path
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None
        self.out = None
        self.byte_limit = byte_limit # 1e9 bytes = 1 GB # 1e6 bytes = 1 MB

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
        
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))

        size = (frame_width, frame_height) 
        
        file_index = 0
        while os.path.exists(f'{self.data_path}/video_recording_{file_index}.avi'):
            file_index += 1
        
        self.out = cv2.VideoWriter(f'{self.data_path}/video_recording_{file_index}.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                self.fps, size)
        
        n_frames = self.duration * self.fps
        curr_frame = 0
        pbar = tqdm(total=n_frames, ncols=0)
        
        while curr_frame < n_frames:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            if grabbed:
                self.out.write(frame)
                curr_frame += 1
                pbar.update(1)

            # Check file size
            if os.path.exists(f'{self.data_path}/video_recording_{file_index}.avi'):
                if os.path.getsize(f'{self.data_path}/video_recording_{file_index}.avi') > self.byte_limit:  # 1e9 bytes = 1 GB # 1e6 bytes = 1 MB
                    # Close current file
                    self.out.release()
                    # Open new file with incremented index
                    file_index += 1
                    self.out = cv2.VideoWriter(f'{self.data_path}/video_recording_{file_index}.avi', 
                                        cv2.VideoWriter_fourcc(*'MJPG'), 
                                        self.fps, size)
        self.started = False

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame, self.started

    def stop(self):
        self.started = False
        self.out.release()
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()