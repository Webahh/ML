from hand_pose_detector import Hand, POS_MAX
from dataclasses import replace
import numpy as np
from gesture import Gesture


def apply_on_gesture(f, gesture: Gesture) -> Gesture:
    """Applies a function f on every hand in every frame of the gesture. f should return a hand"""

    new_frames = []
    for frame in gesture.frames:
        hands = []
        for hand in frame:
            if hand.is_empty():
                hands.append(Hand.empty(hand.left_hand))
            else:
                hands.append(f(hand))

        new_frames.append(hands)

    return replace(gesture, frames=new_frames)


"""
PIPELINE FUNCTIONS:
The following functions can be used as pipeline functions

A pipeline function takes a gesture as input, and returns any number of  modified copies.
Addiditonally pipeline function can take keyword arguments

"""


def mirror(gesture) -> [Gesture]:
    """
    Takes a gesture and mirros it,
    returning the mirrored version.
    This is usefull for automatically generating mirrored versions of gestures
    """

    def mirror_hand(hand: Hand) -> Hand:
        # Spiegelung an der Y-Achse (x-Werte umkehren)
        mirrored_landmarks = {
            k: np.array([-v[0], v[1], v[2]]) for k, v in hand.landmarks.items()
        }

        off = int(POS_MAX / 2)
        mirrored_wrist = (hand.wrist_pos - off) * [-1, 1] + off

        return replace(
            hand,
            landmarks=mirrored_landmarks,
            left_hand=(not hand.left_hand),
            wrist_pos=mirrored_wrist,
        )

    return [apply_on_gesture(mirror_hand, gesture)]


def translate(gesture: Gesture, offset: [int, int]) -> [Gesture]:
    """
    Takes a gesture and produces a copy of it with the given offset applied

    """

    def offset_hand(hand: Hand) -> Hand:
        return replace(hand, wrist_pos=np.array(hand.wrist_pos + offset))

    return [apply_on_gesture(offset_hand, gesture)]


def random_translate(gesture: Gesture, max_offset: int = 20) -> [Gesture]:
    offset = np.random.randint(-max_offset, max_offset + 1, size=2)
    return translate(gesture, offset)


def scale(gesture: Gesture, factor: float = 1.1) -> [Gesture]:
    def scale_hand(hand: Hand) -> Hand:
        scaled_landmarks = {
            k: np.array(v * factor, dtype=np.int16) for k, v in hand.landmarks.items()
        }
        scaled_wrist = np.array(hand.wrist_pos * factor, dtype=np.int16)
        return replace(hand, landmarks=scaled_landmarks, wrist_pos=scaled_wrist)

    return [apply_on_gesture(scale_hand, gesture)]


def jitter(gesture: Gesture, noise_level: float = 5.0) -> [Gesture]:
    """
    simulates noises with random shifts (e.g. camera or handshaking)
    """

    def jitter_hand(hand: Hand) -> Hand:
        new_landmarks = {
            name: pos + np.random.normal(0, noise_level, size=3)
            for name, pos in hand.landmarks.items()
        }
        new_wrist = hand.wrist_pos + np.random.normal(0, noise_level, size=2)
        return replace(hand, landmarks=new_landmarks, wrist_pos=new_wrist)

    return [apply_on_gesture(jitter_hand, gesture)]


def zoom(gesture: Gesture, scale_factor: float = 1.2) -> [Gesture]:
    """
    Moves the hand to the camera
    """

    def zoom_hand(hand: Hand) -> Hand:
        anchor = hand.wrist_pos

        new_landmarks = {
            name: anchor + (pos - anchor) * scale_factor
            for name, pos in hand.landmarks.items()
        }

        return replace(hand, landmarks=new_landmarks)

    return [apply_on_gesture(zoom_hand, gesture)]


def random_zoom(gesture: Gesture, min_factor=0.8, max_factor=1.2) -> [Gesture]:
    """
    Random zoom, for a random position on the z axis
    """
    factor = np.random.uniform(min_factor, max_factor)
    return zoom(gesture, scale_factor=factor)


def drop_frames(gesture: Gesture, drop_rate: float = 0.1) -> [Gesture]:
    """
    Removes analog to the drop_rate some frames
    """
    total = len(gesture.frames)
    keep_mask = np.random.rand(total) > drop_rate
    new_frames = [frame for i, frame in enumerate(gesture.frames) if keep_mask[i]]

    if not new_frames:
        new_frames = [gesture.frames[len(gesture.frames) // 2]]

    return [replace(gesture, frames=new_frames)]


"""

END OF PIPELINE FUNCTIONS

"""


class AugmentationPipeline:
    """
    Generates a pipeline that takes a Gesture and calls a number of augmentation functions on it.
    This pipeline will return the result of the Augmentation
    """

    def __init__(self):
        self.__pipeline = []

    def add(self, name: str, func, **kwargs):
        """Adds a step to the pipeline, that is applied for every gesture"""

        def wrapper(gesture: Gesture):
            return [(g, name) for g in func(gesture, **kwargs)]

        self.__pipeline.append(wrapper)

    def augment(self, gesture: Gesture) -> list[tuple[Gesture, str]]:
        """Returns list of (augmented_gesture, augmentation_name)"""
        results = [(gesture, "orig")]

        for func in self.__pipeline:
            new_results = []

            for g, label in results:
                augmented = func(g)
                for aug_g, aug_label in augmented:
                    combined_label = (
                        f"{label}+{aug_label}" if label != "orig" else aug_label
                    )
                    new_results.append((aug_g, combined_label))

            results.extend(new_results)

        return results
