import cv2
import mediapipe as mp
import pyautogui
import time

# Mediapipe Hands Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Swipe detection thresholds
SWIPE_DISTANCE = 80     # Minimum pixels moved
SWIPE_TIME = 0.2       # Max time for a swipe action

# Store previous finger position
prev_x, prev_y = 0, 0
prev_time = time.time()

# Gesture action cooldown
last_action_time = 0
ACTION_COOLDOWN = 0.5

def detect_swipe(x, y):
    global prev_x, prev_y, prev_time, last_action_time

    # Time and position difference
    dx = x - prev_x
    dy = y - prev_y
    current_time = time.time()
    dt = current_time - prev_time

    prev_x, prev_y = x, y
    prev_time = current_time

    # Check cooldown
    if current_time - last_action_time < ACTION_COOLDOWN:
        return None

    # Swipe detection (fast movement in short time)
    if dt < SWIPE_TIME:
        # Horizontal Swipe
        if dx > SWIPE_DISTANCE:
            last_action_time = current_time
            return "RIGHT"
        elif dx < -SWIPE_DISTANCE:
            last_action_time = current_time
            return "LEFT"
        # Vertical Swipe
        elif dy < -SWIPE_DISTANCE:
            last_action_time = current_time
            return "UP"
        elif dy > SWIPE_DISTANCE:
            last_action_time = current_time
            return "DOWN"
    return None

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Index finger tip (landmark 8)
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)

                gesture = detect_swipe(x, y)

                # Trigger actions
                if gesture == "LEFT":
                    pyautogui.press("left")
                    print("Swipe LEFT → LEFT arrow")
                elif gesture == "RIGHT":
                    pyautogui.press("right")
                    print("Swipe RIGHT → RIGHT arrow")
                elif gesture == "UP":
                    pyautogui.press("up")
                    print("Swipe UP → JUMP")
                elif gesture == "DOWN":
                    pyautogui.press("down")
                    print("Swipe DOWN → ROLL")

        cv2.imshow("Poki Subway Surfer - Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
