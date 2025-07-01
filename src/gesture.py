from dataclasses import dataclass
from hand_pose_detector import Hand


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
