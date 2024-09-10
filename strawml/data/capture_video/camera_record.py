import cv2
from video_capture import VideoCaptureAsync
import time
import threading
import queue

def record_video(duration, fps=30):
    print("STARTED: record_video")
    capture = VideoCaptureAsync(src=1, duration=duration, fps=fps)

    frame_period = 1.0 / fps  # time between frames in seconds
    capture.start()    
    start_time = time.time()
    
    while True:
        ret, frame = capture.read()
        if ret:
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if capture.started == False:
            break
    capture.stop()
    cv2.destroyAllWindows()
    print("STOPPED: record_video")

if __name__ == '__main__':
    record_video(20)