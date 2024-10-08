import cv2
from cv2 import aruco
import numpy as np
from typing import Tuple, Optional
import os
import pupil_apriltags
import time


class ArucoDetector:
    def __init__(self, marker_dict, detector_params, window=False):
        self.marker_dict = marker_dict
        self.detector_params = detector_params
        self.window = window
        self.detector = self.create_detector()

    def create_detector(self):
        self.detector_params.minMarkerPerimeterRate = 0.01
        return aruco.ArucoDetector(self.marker_dict, self.detector_params)

    def detectMarkers(self, frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.detector.detectMarkers(frame)

    def drawMarkers(self, frame, marker_corners, marker_IDs):
        if marker_corners:
            for ids, corners in zip(marker_IDs, marker_corners):
                cv2.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                )
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                # top_left = corners[1].ravel()
                # bottom_right = corners[2].ravel()
                # bottom_left = corners[3].ravel()
                cv2.putText(
                    frame,
                    f"id: {ids[0]}",
                    top_right,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.3,
                    (200, 100, 0),
                    2,
                    cv2.LINE_AA,
                )
        return frame

    def __call__(self, frame: Optional[np.ndarray] = None, cap = None):
        if self.window:
            if not cap:
                cap = cv2.VideoCapture(0)
            while True:
                success, frame = cap.read()
                if not success:
                    continue
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                marker_corners, marker_IDs, reject = self.detectMarkers(gray_frame)
                print(marker_IDs)
                frame = self.drawMarkers(frame, marker_corners, marker_IDs)
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            marker_corners, marker_IDs, reject = self.detectMarkers(frame)
            frame = self.drawMarkers(frame, marker_corners, marker_IDs)
            return frame, marker_corners, marker_IDs


class AprilDetector:
    def __init__(self, detector, ids, window=False):
        self.detector = detector
        self.window = window
        self.ids = ids

    def draw(self, frame, tags):
        # flip the self.ids dictionary
        ids_ = {int(v): k for k, v in self.ids.items()}
        number_tags = []
        chute_tags = []
        for t in tags:
            corners = t.corners
            if t.tag_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                number_tags.append(t)
                cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            else:
                chute_tags.append(t)
                cv2.polylines(frame, [corners.astype(np.int32)], True, (255, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"{int(t.tag_id)}", tuple(corners[0].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        try:
            chute_right = []
            chute_left = []
            for tag in chute_tags:               
                corners = tag.corners
                top_right = corners[1]
                bottom_right = corners[2]
                # get center
                center = (top_right + bottom_right) / 2
                max_x = max(corners[:, 0])

                # find the largest x value
                if tag.tag_id in [11, 12, 13, 14, 19, 20, 21, 22]: 
                    chute_right = [(max_x, center[1])]
                else:
                    chute_left = [(max_x, center[1])]

            # The logic is as follows:
            # 1. For each number tag, find the center of the chute tag that is closest to it on the y-axis in let and right
            # 2. Draw a line going from the right site of the number tag going horizontaly to x being the x value of the right chute tag plus the difference between the x value of the number tag and the x value of the left chute tag
            for tag in number_tags:
                corners = tag.corners
                top_right = corners[1]
                bottom_right = corners[2]
                center = (top_right + bottom_right) / 2
                min_distance_right = float('inf')
                min_distance_left = float('inf')
                closest_left_chute = None
                closest_right_chute = None

                for chute in chute_right:
                    distance = abs(chute[1] - center[1])
                    if distance < min_distance_right:
                        min_distance_right = distance
                        closest_right_chute = chute

                for chute in chute_left:
                    distance = abs(chute[1] - center[1])
                    if distance < min_distance_left:
                        min_distance_left = distance
                        closest_left_chute = chute

                if closest_right_chute and closest_left_chute:
                    line_begin = (int(center[0]), int(center[1]))
                    line_end = (int(closest_right_chute[0] + (closest_left_chute[0] - center[0])), int(center[1]))
                    cv2.line(frame, tuple(line_begin), tuple(line_end), (105, 105, 105), 2)
                    cv2.putText(frame, f"{int(tag.tag_id) * 10}%", (int(closest_right_chute[0] + (closest_left_chute[0] - center[0]))+35, int(center[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            center_chute = (chute_right[0][0] + chute_left[0][0]) // 2
            number_tags = sorted(number_tags, key=lambda x: x.tag_id)
        
            number_tag_centers = np.array([tag.center for tag in number_tags])
            
            # Test with a fixed y value as the predicted value from the model
            y_value = 479
            cv2.circle(frame, (int(center_chute),  y_value), 20, (0, 255, 255), -1)

            NEW_PERCENTAGE, PIXEL_VALUE = self.non_equidistant_interpolation(y_value, number_tag_centers)
            # plot the percentage on the frame
            if NEW_PERCENTAGE == 100:
                y_value = int(np.max(number_tag_centers[:, 1]))
            elif NEW_PERCENTAGE == 0:
                y_value = int(np.min(number_tag_centers[:, 1]))

            # The overlay mismatch is due to linear interpolation. We should instead account for non-equdistant distances between the tags
            lower_level_y_value = int(np.max(number_tag_centers[:, 1]))
            new_y_value = frame.shape[0] - int(PIXEL_VALUE) + (frame.shape[0] - lower_level_y_value)
            cv2.circle(frame, (int(center_chute),  new_y_value), 20, (0, 255, 0), -1)
            cv2.putText(frame, f"{NEW_PERCENTAGE:.2f}%", (int(center_chute) - 20, new_y_value - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception as e:
            pass
        return frame

    def non_equidistant_interpolation(self, y_value, data):
        
        # go through data iteratively and take point1 and point2
        # if y_value is between point1 and point2, interpolate
        # if y_value is less than point1, return 100%
        # if y_value is greater than point2, return 0%
        for i in range(len(data) - 1):
            point1 = data[i]
            point2 = data[i + 1]
            if point1[1] >= y_value >= point2[1]:
                # interpolate
                decimal = np.interp(y_value, [point2[1], point1[1]], [100, 0])
                # get the y-value of the found decimal
                y_value = np.interp(decimal, [100, 0], [point2[1], point1[1]])
                print((decimal + 10 - i)*10, y_value)
                return (decimal + 10 - i)*10, y_value
            elif y_value > data[0][1]:
                return 100, data[0][1]
            elif y_value < data[-1][1]:
                return 0, data[-1][1]
        return 0, 0


    def __call__(self, frame: Optional[np.ndarray] = None, cap = None) -> None:
        with open('data/hkvision_credentials.txt', 'r') as f:
            credentials = f.read().splitlines()
            username = credentials[0]
            password = credentials[1]
            ip = credentials[2]
            rtsp_port = credentials[3]
            
        if self.window and frame is None:
            if not cap:
                cap = cv2.VideoCapture(0)
            while True:
                success, frame = cap.read()
                # resize frame
                if not success:
                    print("Warning: Failed to read frame from stream, skipping...")
                    time.sleep(0.1)  # Short delay before retrying
                    cap.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tags = self.detector.detect(gray)            
                frame = self.draw(frame, tags)
                frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        elif frame is not None:
            # resize frame half the size
            frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray)
            frame = self.draw(frame, tags)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            return tags