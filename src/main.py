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
    return int(lms.landmark[0].x * w / 32767), int(lms.landmark[0].y * h / 32767)


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
    wrist_pos: (int, int)

    # Hand landmarks relative to the wrist
    # Reference: https://ai.google.dev/static/edge/mediapipe/images/solutions/hand-landmarks.png
    landmarks: {}

    # Area in px where the hand is located relative within the img (x, y, w, h)
    hand_area: (int, int, int, int)

    # landmark positions in px within the image
    landmark_pos: []


class HandPoseDetector:
    def __init__(self):
        hands = mp.solutions.hands

        self.__hands = hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def detect(self, img, *args, **kwargs):
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

        return ret


def draw_skeleton(img, pose: Hand):
    # Make left hand joints blue and right hand joints red
    col_joint = (0, 0, 255) if pose.left_hand else (255, 0, 0)

    # Make the bones white
    col_bone = (255, 255, 255)

    def joint(p, pos):
        cv.circle(img, tuple(p), 4, col_joint, -1)
        cv.addText(
            img,
            f"({pos[0]}, {pos[1]}, {pos[2]})",
            (p[0] + 5, p[1]),
            color=col_joint,
            pointSize=8,
            nameFont="NotoSans",
        )

    def bone(p1, p2):
        cv.line(img, tuple(p1), tuple(p2), col_bone, 2)

    def bone_mesh(*args):
        for index in range(len(args) - 1):
            bone(pose.landmark_pos[args[index]], pose.landmark_pos[args[index + 1]])

    bone_mesh(0, 1, 2, 3, 4)
    bone_mesh(5, 6, 7, 8)
    bone_mesh(9, 10, 11, 12)
    bone_mesh(13, 14, 15, 16)
    bone_mesh(0, 17, 18, 19, 20)
    bone_mesh(1, 5, 9, 13, 17)

    for i, p in enumerate(pose.landmark_pos):
        joint(p, pose.landmarks[LANDMARK_NAMES[i]])


def main():
    # Camera Setup
    camera = cv.VideoCapture(0)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 960 * 1.5)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 640 * 1.5)

    hand_pose = HandPoseDetector()

    while True:
        # Exit on Q
        if key := cv.waitKey(35):
            if key == ord("q"):
                print("Q was pressed. Exiting!")
                break

        # Read image from camera
        ok, img = camera.read()
        if not ok:
            print("Failed to fetch frame from camera. Exiting!")
            break

        # Flip image and copy it. The copy (hi_img) is presented to the user, while the original (img)
        # is used to perform inference on the hand pose.
        img = cv.flip(img, 1)
        hi_img = copy.deepcopy(img)

        # Prepare the image for classification
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img.flags.writeable = False

        # Detect hand poses within the image
        pose = hand_pose.detect(img)

        for hand in pose:
            draw_skeleton(hi_img, hand)

        # Present the image
        cv.imshow("Handpose", hi_img)

    # Destroy Camera
    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
