# Import Libraries
import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       model_complexity=1,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75,
                       max_num_hands=2)
draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read video frame by frame
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (Mediapipe expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(frame_rgb)

    # Check if hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Get handedness (left or right hand)
            label = hand_label.classification[0].label  # 'Left' or 'Right'

            # Draw landmarks on the hand
            draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the center of the wrist landmark to display the label
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

            # Display the hand label on the video frame
            cv2.putText(frame, label, (wrist_x, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Hand Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
