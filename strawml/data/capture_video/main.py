from __init__ import *
import cv2
from video_capture import VideoCaptureAsync
import time
from tqdm import tqdm


def record_video(duration, fps=30, data_path='data/raw/videos', byte_limit=1e9):
    print("STARTED: record_video\n")
    capture = VideoCaptureAsync(src=0, duration=duration, fps=fps, data_path=data_path, byte_limit=byte_limit)
    start_time = time.time()
    # Initialize tqdm with a custom format to show only the elapsed time
    pbar = tqdm(bar_format='{l_bar}{bar} | Elapsed Time: {elapsed}', ncols=0)
    capture.start()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= duration:
            break
        pbar.set_postfix_str(f'{elapsed_time:.2f}s')
    pbar.close()
    capture.stop()
    cv2.destroyAllWindows()

    print("\nSTOPPED: record_video")

if __name__ == '__main__':
    duration=10
    fps=30
    data_path='data/raw/videos'
    byte_limit=1e9
    record_video(duration, fps, data_path, byte_limit)