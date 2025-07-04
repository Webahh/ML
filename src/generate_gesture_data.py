import multiprocessing
import uuid
import sys

import cv2
import pickle
import os
from hand_pose_detector import HandPoseDetector, Hand
from augment import AugmentationPipeline, mirror, translate
from gesture import Gesture


detector = HandPoseDetector()


def process_video(video_path: str, label: str) -> Gesture:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    print(f"Processing video {label}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hands = detector.detect(frame)

        # Always have a left and right hand in each frame, if there are no hands detected, use empty hands...
        left = Hand.empty(left=True)
        right = Hand.empty(left=False)

        for hand in hands:
            if hand.left_hand:
                left = hand
            else:
                right = hand

        frames.append([left, right])

    cap.release()

    print(f"Finished processing video {label}")
    return Gesture(label=label, frames=frames, fps=fps).upscale_fps()


def save_gesture(savedir: str, gesture: Gesture, augtype: str = "orig"):
    uid = uuid.uuid4().hex[:4]
    filename = f"{gesture.label}_{augtype}_{uid}.pkl"
    path = os.path.join(savedir, filename)

    with open(path, "wb") as f:
        pickle.dump(gesture, f)

    print(f"Gespeichert: {path}")


def delete_old_gestures(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(output_dir):
        if file.endswith(".pkl"):
            os.remove(os.path.join(output_dir, file))

    print(f"Deleted old gestures in {output_dir}.")


def generate_gestures(video_dir="ressources/videos", output_dir="ressources/gestures"):
    delete_old_gestures(output_dir)

    if not os.path.isdir(video_dir):
        print(
            f"No video-gestures found in '{video_dir}'. No new gestures will be generated."
        )
        return

    training_data = []
    for file in os.listdir(video_dir):
        if file.startswith("alph_fw_") and file.endswith(".mp4"):
            label = file.replace("alph_fw_", "").replace(".mp4", "").upper()
            training_data.append((os.path.join(video_dir, file), label))

    if not training_data:
        print("Keine passenden Videos gefunden.")
        return

    base_gestures = []

    if "-s" in sys.argv or "--single-thread" in sys.argv:
        for video, label in training_data:
            base_gestures.append(process_video(video, label))
    else:
        # Multiprocessing for videos
        cpu_count = multiprocessing.cpu_count()
        num_processes = max(1, int(cpu_count * 0.8))
        with multiprocessing.get_context("spawn").Pool(processes=num_processes) as pool:
            base_gestures = pool.starmap(process_video, training_data)

    print("Video processing finished, running augmentation pipeline")

    base_gestures = [g for g in base_gestures if len(g.frames)]

    pipeline = AugmentationPipeline()
    pipeline.add("trans", translate, offset=[1500, 0])
    pipeline.add("mirr", mirror)

    for gesture in base_gestures:
        print(f"Augmenting {gesture.label}...")
        for aug_gesture, augtype in pipeline.augment(gesture):
            save_gesture(output_dir, aug_gesture, augtype=augtype)

    print(
        f"{len(base_gestures)} Videos verarbeitet und mit Augmentierungen gespeichert."
    )


if __name__ == "__main__":
    generate_gestures()
