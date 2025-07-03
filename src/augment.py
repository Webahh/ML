from hand_pose_detector import Hand, POS_MAX
from dataclasses import replace
import numpy as np
from gesture import Gesture


def apply_on_gesture(f, gesture: Gesture) -> Gesture:
    """Applies a function f on every hand in every frame of the gesture. f should return a hand"""

    new_frames = []
    for frame in gesture.frames:
        hands = [f(hand) for hand in frame]
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

    # TODO random translate, scale, länge von manchen knochen ggf ändern?
    # TODO von 60 FPS aauf 5, 10, 20, 30 reduzieren


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
                    combined_label = f"{label}+{aug_label}" if label != "orig" else aug_label
                    new_results.append((aug_g, combined_label))

            results.extend(new_results)

        return results
