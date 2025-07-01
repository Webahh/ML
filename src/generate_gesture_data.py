import os
import cv2
import pickle
import numpy as np
from main import HandPoseDetector, Hand
from dataclasses import replace
from gesture import Gesture


def mirror_hand(hand: Hand) -> Hand:
    # Spiegelung an der Y-Achse (x-Werte umkehren)
    mirrored_landmarks = {
        k: np.array([-v[0], v[1], v[2]]) for k, v in hand.landmarks.items()
    }
    return replace(hand, landmarks=mirrored_landmarks)


def process_video(video_path: str, label: str, augment=True) -> Gesture:
    cap = cv2.VideoCapture(video_path)
    detector = HandPoseDetector()
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hands = detector.detect(rgb)

        if hands:
            all_hands = []
            for hand in hands:
                all_hands.append(hand)

                if augment:
                    all_hands.append(mirror_hand(hand))

            frames.append(all_hands)

    cap.release()
    return Gesture(label=label, frames=frames)


def save_gesture(gesture: Gesture, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(gesture, f)
    print(f"Gespeichert: {filename}")


if __name__ == "__main__":
    gesture = process_video("videos/A.mp4", "A", augment=False)  # true == mirrored
    os.makedirs("gestures", exist_ok=True)
    save_gesture(gesture, "gestures/A.pkl")
