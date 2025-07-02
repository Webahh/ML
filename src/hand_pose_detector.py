import copy
from dataclasses import dataclass
import cv2 as cv
import mediapipe as mp
import numpy as np


LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_FINGER_MCP",
    "PINKY_FINGER_PIP",
    "PINKY_FINGER_DIP",
    "PINKY_FINGER_TIP",
]


POS_MAX = 32767


def find_hand_area(lms, w, h):
    """Finds the area in which a hand is located"""

    np_landmarks = np.empty((0, 2), int)

    for _, lm in enumerate(lms.landmark):
        # Get the x and y location of each landmark while clamping it to be inside of the image dimensions
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)

        np_landmarks = np.append(np_landmarks, [np.array((x, y))], axis=0)

    return cv.boundingRect(np_landmarks)


def find_wrist_pos(lms, w, h):
    return np.array(
        [int(lms.landmark[0].x * POS_MAX), int(lms.landmark[0].y * POS_MAX)]
    )


def normalize_landmarks(lms):
    """Converts the landmark data to i16 with each point being relative to the wrist"""

    # Convert all landmark data to i16
    int_lms = np.empty((21, 3), np.int16)

    r = 32767

    def clamp(x):
        return max(-r, min(r, int(x * r)))

    for index, lm in enumerate(lms.landmark):
        x = clamp(lm.x)
        y = clamp(lm.y)
        z = clamp(lm.z)

        int_lms[index] = np.array([x, y, z])

    wrist = copy.deepcopy(int_lms[0])
    normalized = int_lms - wrist

    landmarks = {}
    for i, p in enumerate(normalized):
        landmarks[LANDMARK_NAMES[i]] = p

    return landmarks


def find_landmark_pos(lms, w, h):
    ret = np.empty((21, 2), int)

    for index, lm in enumerate(lms.landmark):
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)

        ret[index] = np.array([x, y])

    return ret


@dataclass(frozen=True)
class Hand:
    # Is this the left or the right hand?
    left_hand: bool

    # Wrist position relative to the upper left corner of the image
    wrist_pos: [int, int]

    # Hand landmarks relative to the wrist
    # Reference: https://ai.google.dev/static/edge/mediapipe/images/solutions/hand-landmarks.png
    landmarks: {}

    # Area in px where the hand is located relative within the img (x, y, w, h)
    hand_area: (int, int, int, int)

    # landmark positions in px within the image
    landmark_pos: []

    def is_empty(self):
        return self.wrist_pos[0] == -100 and self.wrist_pos[1] == -100

    @staticmethod
    def empty(left: bool):
        return Hand(
            left_hand=bool,
            wrist_pos=np.array([-100, -100]),
            landmarks={name: np.array([0, 0, 0]) for name in LANDMARK_NAMES},
            hand_area=(0, 0, 0, 0),
            landmark_pos=[],
        )


class HandPoseDetector:
    def __init__(self):
        hands = mp.solutions.hands

        self.__hands = hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def detect(self, img, *args, **kwargs) -> [Hand]:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img.flags.writeable = False

        results = self.__hands.process(img, *args, **kwargs)
        w, h = img.shape[1], img.shape[0]

        ret = []

        if results.multi_hand_landmarks is not None:
            for lms, lr in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand = Hand(
                    left_hand="Left" in str(lr),
                    wrist_pos=find_wrist_pos(lms, w, h),
                    landmarks=normalize_landmarks(lms),
                    hand_area=find_hand_area(lms, w, h),
                    landmark_pos=find_landmark_pos(lms, w, h),
                )

                ret.append(hand)

        img.flags.writeable = True

        return ret
