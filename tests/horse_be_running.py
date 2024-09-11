import cv2


def main():
    # Indlæser video
    cap = cv2.VideoCapture("../video_03/Muybridge_race_horse_animated.gif")

    # Sætter fps (Ikke for høj, da at der ikke er så mange frames i denne video)
    fps = 10

    # Træk højde og bredde 
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Afspil video med rette fps
    success, frame = cap.read()
    while success:
        success, frame = cap.read()
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1000//fps)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()