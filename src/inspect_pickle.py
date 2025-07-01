import pickle
import time
from gesture import Gesture
from visualizer import visualize

pkl_path = "ressources/gestures/Test_2.pkl"

with open(pkl_path, "rb") as f:
    gesture: Gesture = pickle.load(f)

print(f"\nLabel: {gesture.label}")
print(f"Anzahl der Frames insgesamt: {len(gesture.frames)}\n")

max_frames = min(20, len(gesture.frames))
print("FPS: " + str(gesture.fps))


for frame_idx in range(max_frames):
    frame = gesture.frames[frame_idx]
    print(f"Frame {frame_idx + 1} enthält {len(frame)} Hand/Hände:")

    for hand_idx, hand in enumerate(frame):
        print(f"  Hand {hand_idx + 1}:")
        print(f"    Linke Hand: {hand.left_hand}")
        print(f"    Wrist Position: {hand.wrist_pos}")
        for key in ["WRIST", "THUMB_TIP", "INDEX_FINGER_TIP"]:
            if key in hand.landmarks:
                print(f"    {key}: {hand.landmarks[key]}")
    print("-" * 40)


# Visualize Skeleton
visualizer = visualize(info=False)
for hands in gesture.frames:
    if not visualizer.send_pose(hands):
        break

    time.sleep(1.0 / gesture.fps)
