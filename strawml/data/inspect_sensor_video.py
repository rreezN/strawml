from __init__ import *
import os
import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm 
import datetime
import time
from argparse import ArgumentParser, Namespace
import psutil
import shutil
import json


def inspect_sensor_video(video_path: str, sensor_file_path: str):
    # the timestamp is in the filename with format NAME NAME NAME 2_HKVision_HKVision_20241102105959_20241102112224_1532587042
    start_timestamp = datetime.datetime.strptime(video_path.split('_')[-3], '%Y%m%d%H%M%S')
    # The actual video is happening 12 hours later than the timestamp
    start_timestamp = start_timestamp + datetime.timedelta(hours=12)
    
    # Load the sensor data from the sensor file
    sensor_df = pd.read_excel(sensor_file_path)
    
    # Remove the first three rows and first column
    sensor_df = sensor_df.iloc[3:, 1:]
    
    timestamp_col = sensor_df.columns[0]
    sensor_col = sensor_df.columns[1]
    sensor_data = sensor_df[sensor_col].values
    time_stamp = sensor_df[timestamp_col].values
    time_stamp = pd.to_datetime(time_stamp)
    time_stamp = time_stamp.sort_values()
    
    # Display the video
    cap = cv2.VideoCapture(video_path)
    # Calculate frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps  # Duration of each frame in seconds
    paused = False  # Start with play mode
    
    while(cap.isOpened()):
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get the current frame timestamp
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_timestamp = start_timestamp + datetime.timedelta(seconds=frame_number * frame_duration)
            
            # Find the sensor data closest to the current frame
            time_diffs = np.abs(time_stamp - current_timestamp)
            closest_idx = np.argmin(time_diffs)
            closest_timestamp = time_stamp[closest_idx]
            closest_data = sensor_data[closest_idx]
            
            
            # Debugging print to check if data changes
            # print(f'Frame {frame_number} | Current Timestamp: {current_timestamp} | Closest Timestamp: {closest_timestamp} | Closest Data: {closest_data}')
            
            # Prepare text to display
            display_text = f'Sensor reading: {closest_data}'
            timestamp_text = closest_timestamp.strftime('%Y-%m-%d %H:%M:%S')

            # Resize the frame to fit the screen
            frame = cv2.resize(frame, (1280, 720))
            
            # Overlay timestamp and data on the frame
            cv2.putText(frame, f'Frame: {frame_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Current Timestamp: {current_timestamp.strftime("%Y-%m-%d %H:%M:%S")}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Closest Timestamp: {timestamp_text}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, display_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Video with Timestamped Data', frame)

        # Wait for a key press and toggle pause/play on space bar press
        key = cv2.waitKey(int(frame_duration * 1000) if not paused else 0) & 0xFF
        if key == ord('q'):  # Quit on 'q' key
            break
        elif key == ord(' '):  # Toggle pause/play on space bar
            paused = not paused
        
    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str, default='data/raw/sensor/videos/Pin drum Chute 2_HKVision_HKVision_20241102105959_20241102112224_1532587042.mp4')
    parser.add_argument('--sensor_file_path', type=str, default='data/raw/sensor/studentexport.xlsx')
    args = parser.parse_args()
    inspect_sensor_video(args.video_path, args.sensor_file_path)
