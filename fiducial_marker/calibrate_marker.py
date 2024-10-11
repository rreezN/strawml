from __init__ import *
import cv2
from cv2 import aruco
import numpy as np
from make_marker import ARUCO_DICT
import threading
from threading import Thread
from queue import Queue


def create_aruco_grid(dict_type: str, 
                      squaresX: int = 5,
                      squaresY: int = 7,
                      squareLength: int = 100,
                      markerLength: int = 60,
                      borderBits: int = 1,
                      showImage: bool = False) -> None:
    """
    Create a ChAruCo board with the given parameters and save it to a file.
    
    Inspired functions from:
        https://github.com/opencv/opencv/blob/4.x/samples/cpp/tutorial_code/objectDetection/create_board_charuco.cpp
    """
    margins = squareLength - markerLength
    boardSize = (squaresX, squaresY)
    marker_dict = aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
    gridboard = aruco.CharucoBoard(size=boardSize, markerLength=markerLength, squareLength=squareLength, dictionary=marker_dict)
    # Save the gridboard to a file
    imageSize = (squaresX * squareLength + 2 * margins, squaresY * squareLength + 2 * margins)
    boardImage = gridboard.generateImage(imageSize, margins, borderBits) # type: ignore
    
    if showImage:
        cv2.imshow('Gridboard', boardImage)
        cv2.waitKey(0)
    
    img_path = f"fiducial_marker/aruco_grids/{dict_type}_gridboard.png"
    cv2.imwrite(img_path, boardImage)

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
class RTSPStream:
    """
    Wrapper class to detect AprilTags in a real-time video stream. The class inherits from the AprilDetector class and
    provides methods to receive frames from the video stream, detect AprilTags in the frames, draw the detected tags on
    the frame, and display the frame with the detected tags. The class also provides a method to detect AprilTags in a
    single frame.
    
    Inspired functions from:
        https://github.com/ddelago/Aruco-Marker-Calibration-and-Pose-Estimation/blob/master/calibration/calibration_ChAruco.py

    NOTE Threading is necessary here because we are dealing with an RTSP stream.
    """
    def __init__(self, credentials_path, window=True, rtsp = False, cap: str | int | None = None):
        if cap is None and rtsp is True:
            self.cap = self.create_capture(credentials_path)
        elif type(cap) is int or cap is None:
            self.cap = cv2.VideoCapture(cap)
        elif type(cap) is str:
            self.cap = cap
        self.q = Queue()
    
    def create_capture(self, params: str) -> cv2.VideoCapture:
        """
        Create a video capture object to receive frames from the RTSP stream.

        Params:
        -------
        params: str
            The path to the file containing the RTSP stream credentials

        Returns:
        --------
        cv2.VideoCapture
            The video capture object to receive frames from the RTSP stream
        """
        with open('data/hkvision_credentials.txt', 'r') as f:
            credentials = f.read().splitlines()
            username = credentials[0]
            password = credentials[1]
            ip = credentials[2]
            rtsp_port = credentials[3]
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
        return cap

    def receive_frame(self, cap: cv2.VideoCapture) -> None:
        """
        Read frames from the video stream and store them in a queue.

        Params:
        -------
        cap: cv2.VideoCapture
            The video capture object to receive frames from the RTSP stream
        
        Returns:
        --------
        None
            Nothing is returned, only the frames are stored in a queue
        """
        ret, frame = cap.read()
        self.q.put(frame)
        while ret and cap.isOpened():
            ret, frame = cap.read()
            self.q.put(frame)        
    
    def display_frame(self, gridboard, cap: cv2.VideoCapture, marker_dict, min_acceptable_images):
        """
        Display the frames with the detected AprilTags.
        
        Params:
        -------
        cap: cv2.VideoCapture
            The video capture object to receive frames from the RTSP stream
        
        Returns:
        --------
        None
            Nothing is returned, only the frames are displayed with the detected AprilTags
        """
        from tqdm import tqdm
        # Corners discovered in all images processed
        corners_all = []

        # Aruco ids corresponding to corners discovered 
        ids_all = [] 

        # Determined at runtime
        image_size = None 

        # The more valid captures, the better the calibration
        validCaptures = 0
        pbar = tqdm(total=self.q.qsize(), desc="Calibrating camera")
        while True:
            if not self.q.empty():
                pbar.update(1)
                frame = self.q.get()
                proportion = max(frame.shape) / 1000.0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Find the aruco markers
                corners, ids, _ = aruco.detectMarkers(gray, marker_dict)
                # if none is found we continue
                if ids is None:
                    frame = cv2.resize(frame, (int(frame.shape[1]/proportion), int(frame.shape[0]/proportion)))
                    cv2.imshow('Charuco board', frame)
                    if cv2.waitKey(1) & 0xFF ==ord('q'):
                        break
                    continue

                # Outline the aruco markers found in our query image
                frame = aruco.drawDetectedMarkers(
                    image=frame, 
                    corners=corners)

                # Get charuco corners and ids from detected aruco markers
                response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=gridboard)
                
                if cv2.waitKey(1) & 0xFF ==ord('q'):
                    break

                # If a Charuco board was found, collect image/corner points
                # Requires at least 20 squares for a valid calibration image
                if response > 20:
                    # Add these corners and ids to our calibration arrays
                    corners_all.append(charuco_corners)
                    ids_all.append(charuco_ids)
                    
                    # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                    frame = aruco.drawDetectedCornersCharuco(
                        image=frame,
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids)
                
                    # If our image size is unknown, set it now
                    if not image_size:
                        image_size = gray.shape[::-1]
                    
                    # Reproportion the image, maxing width or height at 1000
                    frame = cv2.resize(frame, (int(frame.shape[1]/proportion), int(frame.shape[0]/proportion)))

                    # Pause to display each image, waiting for key press
                    cv2.imshow('Charuco board', frame)
                    if cv2.waitKey(1) & 0xFF ==ord('q'):
                        break
                    validCaptures += 1
                    pbar.set_postfix_str(f"Valid: {validCaptures}")
                    # if validCaptures == min_acceptable_images:
                        # break
            else:
                break
        if not type(cap) == str:
            cap.release()
        cv2.destroyAllWindows()
        
        print("{} valid captures".format(validCaptures))
        if validCaptures < min_acceptable_images:
            print("Calibration was unsuccessful. We couldn't detect enough charucoboards in the video.")
            print("Perform a better capture or reduce the minimum number of valid captures required.")
            exit()
            
        # Make sure we were able to calibrate on at least one charucoboard
        if len(corners_all) == 0:
            print("Calibration was unsuccessful. We couldn't detect charucoboards in the video.")
            print("Make sure that the calibration pattern is the same as the one we are looking for (ARUCO_DICT).")
            exit()
        print("Generating calibration...")
        
        # Now that we've seen all of our images, perform the camera calibration
        calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(charucoCorners=corners_all, 
                                                                                           charucoIds=ids_all, 
                                                                                           board=gridboard, 
                                                                                           imageSize=image_size, 
                                                                                           cameraMatrix=None, 
                                                                                           distCoeffs=None) # type: ignore
        # Print matrix and distortion coefficient to the console
        print("Camera intrinsic parameters matrix:\n{}".format(cameraMatrix))
        print("\nCamera distortion coefficients:\n{}".format(distCoeffs))
        return calibration, cameraMatrix, distCoeffs, rvecs, tvecs
    
    def load_queue(self) -> None:
        """
        Load the images from the path self.cap to the queue.

        Returns:
        --------
        None
            Nothing is returned, only the images are loaded to the queue
        """
        import os
        from tqdm import tqdm
        images = os.listdir(self.cap)[400:2000]
        for img in tqdm(images, desc="Loading images to queue"):
            frame = cv2.imread(os.path.join(self.cap, img))
            self.q.put(frame)

        # frame = cv2.imread(os.path.join(self.cap, "frame_520.jpg"))
        # self.q.put(frame)

    def __call__(self, gridboard, marker_dict, min_acceptable_images, cap: None|str|cv2.VideoCapture = None) -> None:
        """
        Upon calling the object, if self.window is True and frame is None, the frames are received from the video stream
        and displayed with the detected AprilTags. If frame is not None, the detected AprilTags are drawn on the frame.
        If a cap object is passed, the threads are redefined with the new cap.

        Params:
        -------
        frame: np.ndarray
            The frame in which the AprilTags are to be detected
        cap: cv2.VideoCapture
            The video capture object to receive frames from the RTSP stream

        Returns:
        --------
        None
            Nothing is returned, only the frames are displayed with the detected AprilTags
        """
        # if cap is not None:
        #     self.cap = cap
        if type(self.cap) == str:
            # Here we load the images in the path self.cap to the queue
            self.load_queue()
            return self.display_frame(gridboard, self.cap, marker_dict, min_acceptable_images)
            # self.thread2 = ThreadWithReturnValue(target=self.display_frame, args=(gridboard, self.cap, marker_dict, min_acceptable_images, ))
            # self.thread2.start()
        else:
            self.thread1 = threading.Thread(target=self.receive_frame, args=(self.cap,))
            self.thread2 = ThreadWithReturnValue(target=self.display_frame, args=(gridboard, self.cap, marker_dict, min_acceptable_images, ))
            self.thread1.start()
            self.thread2.start()
        return self.thread2.join()


def calibrate_camera(dict_type: str,
                     min_acceptable_images: int, 
                     squaresX: int = 5,
                     squaresY: int = 7,
                     squareLength: int = 100,
                     markerLength: int = 60,
                     rtsp: bool = True) -> None:
    """
    Calibrate the camera using the ChAruCo board.
    """
    # cap = cv2.VideoCapture("data/calibration/vlc-record-2024-10-10-14h01m18s-rtsp___10.5.242.32_554_Streaming_Channels_101-.mp4")
    cap = "data/calibration/images/"
    # cap = 0
    boardSize = (squaresX, squaresY)
    marker_dict = aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
    gridboard = aruco.CharucoBoard(size=boardSize, markerLength=markerLength, squareLength=squareLength, dictionary=marker_dict)
    calibrate_stream = RTSPStream(credentials_path='data/hkvision_credentials.txt', rtsp=rtsp, cap=cap)
    calibration, cameraMatrix, distCoeffs, rvecs, tvecs = calibrate_stream(gridboard, marker_dict, min_acceptable_images) # type: ignore
    np.savez(f"fiducial_marker/calibration.npz", calibration=calibration, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvecs=rvecs, tvecs=tvecs)
    
    
if __name__ == "__main__":
    calibrate_camera("DICT_4X4_50", 20, rtsp=False)