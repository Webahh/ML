import time
import cv2 as cv

from hand_pose_detector import HandPoseDetector
from visualizer import visualize
from augment import AugmentationPipeline, mirror, translate
from gesture import Gesture


# Use the Augmentation pipeline to build generate modified gestures, based on
# a input gesture. This can be used to generate more training data based on existing data.
def augment_pipeline_example(gesture: Gesture) -> [Gesture]:
    # Build the pipeline
    pipeline = AugmentationPipeline()

    # Create a mirrored clone
    pipeline.add(mirror)

    # Create copies moved a little bit up
    pipeline.add(translate, offset=[0, -10000, 0])

    # Run the pipeline
    gestures = pipeline.augment(gesture)

    return gestures


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

        # NOTE: This is just for fun:
        # Using the augmentation pipeline to generate 3 ghost hands if only one hand is visible.
        # This has no practical application, but showcases the augmentation pipeline
        if len(hands) == 1:
            # Convert the hands to a gesture, since the AugmentationPipeline works on gestures
            gesture = Gesture.from_hands(hands)

            # Run the pipeline
            gestures = augment_pipeline_example(gesture)

            # Convert the gestures back to hands, so they can be visualized
            hands = [g.to_hands()[0] for g in gestures]

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
