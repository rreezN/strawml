import cv2
import time


class VideoStreamStraw:
    def __init__(self):
        pass

    
    def edge_detect(self, image):
        image_size = image.shape
        cropped = image[0:image_size[1], (image_size[0]//2)+200:(image_size[0]//2) + 400]
        edges = cv2.Canny(cropped, 100, 200)
        return edges
    
    def __call__(self, video):
        with open('data/hkvision_credentials.txt', 'r') as f:
            credentials = f.read().splitlines()
            username = credentials[0]
            password = credentials[1]
            ip = credentials[2]
            rtsp_port = credentials[3]

        while True:
            try:
                start_time = time.time()  # Record the start time

                success, image = video.read()
                if not success:
                    print("Warning: Failed to read frame from stream, skipping...")
                    time.sleep(0.1)  # Short delay before retrying
                    video.open(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101")
                    continue

                image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
                image = self.edge_detect(image)
                cv2.imshow('Video', image)

                end_time = time.time()  # Record the end time
                frame_time = end_time - start_time
                # print(f"Time between frames: {frame_time:.4f} seconds")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        video.release()
        cv2.destroyAllWindows()