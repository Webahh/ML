import time
import cv2 as cv

from hand_pose_detector import HandPoseDetector
from visualizer import visualize


def main():
    # Camera Setup
    camera = cv.VideoCapture(0)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 960 * 1.5)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 640 * 1.5)

    # Get HandPoseDetector and Visualizer instances
    hand_pose = HandPoseDetector()
    visualizer = visualize(info=False)  # info=False => Dont display joint positions

    while True:
        # Read image from camera
        ok, img = camera.read()
        if not ok:
            print("Failed to fetch frame from camera. Exiting!")
            break

        # Flip image and detect hand poses in it
        img = cv.flip(img, 1)
        hands = hand_pose.detect(img)

        # Use the visualizer to display the webcam image, aswell as the hand poses in it.
        # If the Visualizer terminates, terminate this loop as well.
        if not visualizer.send_img_pose(img, hands):
            break

        # Lock on 15 FPS
        time.sleep(1.0 / 15.0)

    # Cleanup
    camera.release()
    visualizer.terminate()


if __name__ == "__main__":
    main()
