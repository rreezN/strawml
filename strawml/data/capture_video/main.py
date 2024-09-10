from __init__ import *
import cv2
from video_capture import VideoCaptureAsync
import time
from tqdm import tqdm
import argparse
import os

def record_video(duration, fps=30, data_path='data/raw/videos', byte_limit=1e9):
    print("STARTED: record_video\n")
    capture = VideoCaptureAsync(src=0, duration=duration, fps=fps, data_path=data_path, byte_limit=byte_limit)
    # Initialize tqdm with a custom format to show only the elapsed time
    capture.start()
    while True:
        if not capture.started:
            break
    capture.stop()
    cv2.destroyAllWindows()
    print("\nSTOPPED: record_video")

def main(args: argparse.Namespace):
    if args.mode == 'record':
        record_video(args.duration, args.fps, args.data_path, args.byte_limit)
    elif args.mode == 'validate':
        # Validate the video file
        if not os.path.exists(f'{args.data_path}/{args.vid_name}'):
            raise FileNotFoundError(f'Video file {args.vid_name} does not exist...')
        
        cap = cv2.VideoCapture(f'{args.data_path}/{args.vid_name}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print(f'Video duration: {duration:.2f}s')
        print(f'Number of frames: {frame_count}')
        cap.release()

def argparser():
    parser = argparse.ArgumentParser(description='Record video from webcam')
    parser.add_argument('mode', type=str, choices=['record', 'validate'], help='Mode to run the script in')
    parser.add_argument('--duration', type=int, default=10, help='Duration of the video in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--data_path', type=str, default='data/raw/videos', help='Path to save the video')
    parser.add_argument('--byte_limit', type=float, default=1e9, help='Maximum size of the video file in bytes')
    parser.add_argument('--vid_name', type=str, default='video_recording_0.avi', help='Name of the video file')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    main(args)
