import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import ctypes

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       model_complexity=1,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75,
                       max_num_hands=2)
draw = mp.solutions.drawing_utils

# Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for intuitive interaction
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Get handedness (left or right)
            label = hand_label.classification[0].label  # 'Left' or 'Right'

            # Draw landmarks
            draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of thumb and index finger
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            thumb_tip = (int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * w),
                         int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * h))
            index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                         int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))
            # Get the center of the wrist landmark to display the label
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            # Calculate distance between thumb and index finger
            distance = hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

            if label == 'Left':
                # Control brightness using the left hand
                brightness_level = np.interp(distance, [15, 220], [0, 100])
                sbc.set_brightness(int(brightness_level))
                cv2.line(frame, (index_tip[0], index_tip[1]), (thumb_tip[0], thumb_tip[1]), (0, 255, 0), 3) 
				# Display the hand label on the video frame
                cv2.putText(frame, label, (wrist_x, wrist_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif label == 'Right':
                # Control volume using the right hand
                volume_level = np.interp(distance, [15, 220], [-65.25, 1])
                volume.SetMasterVolumeLevel(volume_level, None)
                cv2.line(frame, (index_tip[0], index_tip[1]), (thumb_tip[0], thumb_tip[1]), (0, 255, 0), 3)
                # Display the hand label on the video frame
                cv2.putText(frame, label, (wrist_x, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display video feed
    cv2.imshow("Hand Control", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
