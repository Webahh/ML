import pickle
from generate_gesture_data import Gesture

pkl_path = "gestures/A.pkl"

with open(pkl_path, "rb") as f:
    gesture: Gesture = pickle.load(f)

print(f"\nLabel: {gesture.label}")
print(f"Anzahl der Frames insgesamt: {len(gesture.frames)}\n")

max_frames = min(20, len(gesture.frames))

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
