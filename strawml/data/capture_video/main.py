from __init__ import *
import cv2
from video_capture import VideoCaptureAsync
import time
from tqdm import tqdm
import argparse
import os

def record_video(duration, fps=30, data_path='data/raw/videos', byte_limit=1e9):
    print("STARTED: record_video\n")
    imgsz = (1024, 768)
    # imgsz = None
    capture = VideoCaptureAsync(src=0, imgsz=imgsz)
    capture.start()
    
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    size = (frame_width, frame_height) 
    
    file_index = 0
    while os.path.exists(f'{data_path}/video_recording_{file_index}.avi'):
        file_index += 1
    
    out = cv2.VideoWriter(f'{data_path}/video_recording_{file_index}.avi', 
                            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                            fps, size)
    
    n_frames = duration * fps
    curr_frame = 0
    pbar = tqdm(total=n_frames, ncols=0)
    
    while curr_frame < n_frames:
        start_time = time.time()
        grabbed, frame = capture.read()
        if grabbed:
            out.write(frame)
            curr_frame += 1
            pbar.update(1)
            if os.path.exists(f'{data_path}/video_recording_{file_index}.avi'):
                if os.path.getsize(f'{data_path}/video_recording_{file_index}.avi') > byte_limit:
                    out.release()
                    file_index += 1
                    out = cv2.VideoWriter(f'{data_path}/video_recording_{file_index}.avi', 
                                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                        fps, size)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elapsed_time = time.time() - start_time
        frame_time = 1.0/fps  # Ideal frame time in seconds
        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time) 
    pbar.close()
    capture.stop()
    cv2.destroyAllWindows()
    print("FINISHED: record_video\n")

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
        # play the video
        if args.play_video:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()

def argparser():
    parser = argparse.ArgumentParser(description='Record video from webcam')
    parser.add_argument('mode', type=str, choices=['record', 'validate'], help='Mode to run the script in')
    parser.add_argument('--duration', type=int, default=1800, help='Duration of the video in seconds')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--data_path', type=str, default='/home/dnd/Documents/capture_video/videos/', help='Path to save the video')
    parser.add_argument('--byte_limit', type=float, default=1e9, help='Maximum size of the video file in bytes')
    parser.add_argument('--vid_name', type=str, default='video_recording_0.avi', help='Name of the video file')
    parser.add_argument('--play_video', action='store_true', help='Play the video after validation')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    main(args)
