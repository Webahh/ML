import cv2
import pickle
import os
from hand_pose_detector import HandPoseDetector
from augment import AugmentationPipeline, mirror
from gesture import Gesture


detector = HandPoseDetector()


def process_video(video_path: str, label: str) -> Gesture:
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hands = detector.detect(frame)
        frames.append(hands)

    cap.release()
    return Gesture(label=label, frames=frames)


def save_gesture(savedir: str, gesture: Gesture, suffix=""):
    path = f"{savedir}/{gesture.label}_{suffix}.pkl"

    with open(path, "wb") as f:
        pickle.dump(gesture, f)

    print(f"Gespeichert: {path}")


def generate_gestures(data: [(str, str)]) -> [Gesture]:
    # Detect gestures
    gestures = [process_video(file, label) for file, label in data]

    # Filter out gestures whith no frames
    gestures = [g for g in gestures if len(g.frames)]

    # Setup Aumentation pipeline to generate mirrored copies of the entire dataset
    pipeline = AugmentationPipeline()
    pipeline.add(mirror)

    # Run augmentation pipeline for each gesture
    augmented_gestures = []
    for g in gestures:
        augmented_gestures.extend(pipeline.augment(g))

    return augmented_gestures


if __name__ == "__main__":
    training_data = [
        ("ressources/videos/A.mp4", "A"),
        ("ressources/videos/test.mp4", "Test"),
    ]

    os.makedirs("ressources/gestures", exist_ok=True)
    gestures = generate_gestures(training_data)

    # Save all gestures
    for i, g in enumerate(gestures):
        save_gesture("ressources/gestures", g, str(i))
