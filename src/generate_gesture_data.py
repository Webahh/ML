import multiprocessing
import uuid
import sys

import cv2
import pickle
import os
from hand_pose_detector import HandPoseDetector, Hand, POS_MAX
from augment import AugmentationPipeline, mirror, random_zoom, random_translate
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

    # # only get a 2 second part of each video
    # gestures = (
    #     Gesture(label=label, frames=frames, fps=fps)
    #     .upscale_fps()
    #     .to_parts(part_len_secs=2)
    # )
    # if len(gestures) >= 2:
    #     return [gestures[1]]
    # else:
    #     return gestures

    return [Gesture(label=label, frames=frames, fps=fps)]


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


def generate_gestures(
    video_dir="ressources/videos_prototype", output_dir="ressources/gestures"
):
    delete_old_gestures(output_dir)

    if not os.path.isdir(video_dir):
        print(
            f"No video-gestures found in '{video_dir}'. No new gestures will be generated."
        )
        return

    import re

    training_data = []
    for file in os.listdir(video_dir):
        if file.startswith("alph_fw_") and file.endswith(".mp4"):
            match = re.match(r"alph_fw_([A-Z])", file, re.IGNORECASE)
            if match:
                label = match.group(1).upper()
                training_data.append((os.path.join(video_dir, file), label))
                print(f"{file} → Label: {label}")
            else:
                print(f"Warnung: Konnte kein Label aus Datei '{file}' extrahieren.")

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

    base_gestures = [g for parts in base_gestures for g in parts if len(g.frames)]

    pipeline = AugmentationPipeline()
    pipeline.add("mirr", mirror)
    pipeline.add("rzoom", random_zoom, count=5, min_factor=0.2, max_factor=2.2)
    pipeline.add("rtrans", random_translate, count=5, max_offset=(POS_MAX // 3))

    def augment_gesture(gesture):
        print(f"Augmenting {gesture.label}...")
        for aug_gesture, augtype in pipeline.augment(gesture):
            save_gesture(output_dir, aug_gesture, augtype=augtype)

    for gesture in base_gestures:
        augment_gesture(gesture)

    # if "-s" in sys.argv or "--single-thread" in sys.argv:
    #     for gesture in base_gestures:
    #         augment_gesture(gesture)

    # else:
    #     cpu_count = multiprocessing.cpu_count()
    #     num_processes = max(1, int(cpu_count * 0.8))
    #     with multiprocessing.get_context("spawn").Pool(processes=num_processes) as pool:
    #         pool.apply(augment_gesture, base_gestures)

    print(
        f"{len(base_gestures)} Videos verarbeitet und mit Augmentierungen gespeichert."
    )


if __name__ == "__main__":
    generate_gestures("ressources/videos")
