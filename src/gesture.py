from dataclasses import dataclass, replace
from hand_pose_detector import Hand, LANDMARK_NAMES
import numpy as np


@dataclass(frozen=True)
class Gesture:
    label: str
    frames: list[list[Hand]]
    fps: float = 1.0

    @staticmethod
    def from_hands(hands: [Hand], label="Unknown"):
        """Converts a list of hands into a gesture with one frame"""
        return Gesture(label=label, frames=[hands])

    def to_hands(self) -> [Hand]:
        """Converts a gesture with one frame into a list of hands"""
        return self.frames[0]

    def upscale_fps(self, target=60):
        t = target // self.fps
        o = target % self.fps

        if t <= 1:
            return self

        overflow = o

        new_frames = []
        for i, frame in enumerate(self.frames[:-1]):
            start = frame
            stop = self.frames[i + 1]

            assert len(start) == 2
            assert len(stop) == 2

            steps = max(0, t - 1)
            overflow += o

            if overflow > target:
                steps = t + 1
                overflow = overflow % self.fps

            left = interpolate_hand(start[0], stop[0], steps)
            right = interpolate_hand(start[1], stop[1], steps)

            new_frames.extend([[l, r] for l, r in zip(left, right)])

        new_frames.append(self.frames[-1])

        og_length = len(self.frames) * (1 / self.fps)
        new_length = len(new_frames) * (1 / self.fps)

        stretch = new_length / og_length

        return replace(self, frames=new_frames, fps=self.fps * stretch)


def hand_mat(hand: Hand):
    return np.array([hand.landmarks[name] for name in LANDMARK_NAMES])


def mat_landmark(mat):
    return {LANDMARK_NAMES[i]: v for i, v in enumerate(mat)}


def interpolate_hand(start: Hand, stop: Hand, stops: int) -> [Hand]:
    if stop.is_empty() or start.is_empty():
        ret = [start]
        ret.extend([Hand.empty(start.left_hand) for _ in range(int(stops))])
        return ret

    p1 = hand_mat(start)
    p2 = hand_mat(stop)
    delta = p2 - p1

    wrist_p1 = start.wrist_pos
    wrist_p2 = stop.wrist_pos
    wrist_delta = wrist_p2 - wrist_p1

    new_frames = [start]

    for stop in range(int(stops)):
        landmarks = np.array(p1 + delta * (1.0 / stops), dtype=np.int16)
        wrist_pos = np.array(wrist_p1 + wrist_delta * (1.0 / stops), dtype=np.int16)

        new_frames.append(
            replace(start, landmarks=mat_landmark(landmarks), wrist_pos=wrist_pos)
        )

    return new_frames
