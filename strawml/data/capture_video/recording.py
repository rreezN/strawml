from __init__ import *
import cv2
from video_capture import VideoCaptureAsync
import time
from tqdm import tqdm
import os

def record_video(duration, fps=30, src=0, data_path='data/raw/videos', byte_limit=1e9):
    print("STARTED: record_video\n")
    if not os.path.exists(data_path):
        print(os.getcwd())
    cap = cv2.VideoCapture(src)
    start_time = time.time()
    time_end = time.time() + duration
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height) 
    
    file_index = 0
    while os.path.exists(f'{data_path}/filename_{file_index}.avi'):
        file_index += 1
    out = cv2.VideoWriter(f'{data_path}/filename_{file_index}.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            fps, size) 
    frames = 0

    pbar = tqdm(bar_format='{l_bar}{bar} | Elapsed Time: {elapsed}', ncols=0)

    while time.time() <= time_end:
        elapsed_time = time.time() - start_time
        grabbed, frame = cap.read()
        if grabbed:
            out.write(frame)
            frames += 1

        # Check file size
        if os.path.exists(f'{data_path}/filename_{file_index}.avi'):
            if os.path.getsize(f'{data_path}/filename_{file_index}.avi') > byte_limit:  # 1e9 bytes = 1 GB # 1e6 bytes = 1 MB
                # Close current file
                out.release()
                # Open new file with incremented index
                file_index += 1
                out = cv2.VideoWriter(f'{data_path}/filename_{file_index}.avi', 
                                    cv2.VideoWriter_fourcc(*'MJPG'), 
                                    fps, size)
        pbar.set_postfix_str(f'{elapsed_time:.2f}s')

    out.release()
    pbar.close()
    print("\nSTOPPED: record_video")

if __name__ == '__main__':
    duration=10
    fps=30
    data_path='data/raw/videos'
    byte_limit=1e9
    src=0
    record_video(duration, fps, src, data_path, byte_limit)