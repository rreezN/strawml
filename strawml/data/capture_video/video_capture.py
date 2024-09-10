# TODO: Save data continuously to a video file to account for memory limitations
# Written by Luis Mesas
import threading
import time
import cv2

# Define video capture class
class VideoCaptureAsync:
    def __init__(self, duration, fps, src=0, driver=None):
        self.src = src
        if driver is None:
            self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src, driver)
        self.duration = duration
        self.fps = fps
        self.frame_period = 1.0 / fps  # time between frames in seconds
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None
        self.out = None

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
        time_end = time.time() + self.duration
        
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))

        size = (frame_width, frame_height) 
        
        file_index = 0
        self.out = cv2.VideoWriter(f'filename_{file_index}.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                self.fps, size) 
        frames = 0
        while time.time() <= time_end:
            start_time = time.time()  # start time of the loop
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            if grabbed:
                self.out.write(frame)
                frames += 1
                # sleep for the remaining time to maintain the desired frame rate
                time.sleep(max(0, self.frame_period - (time.time() - start_time)))

            # Check file size
            if os.path.getsize(f'filename_{file_index}.avi') > 1e9:  # 1e9 bytes = 1 GB
                # Close current file
                self.out.release()
                # Open new file with incremented index
                file_index += 1
                self.out = cv2.VideoWriter(f'filename_{file_index}.avi', 
                                    cv2.VideoWriter_fourcc(*'MJPG'), 
                                    self.fps, size)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.out.release()
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()