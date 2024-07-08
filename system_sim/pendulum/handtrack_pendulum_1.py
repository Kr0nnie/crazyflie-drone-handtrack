import numpy as np
import cv2
import mediapipe as mp

# Constants
POINT1 = np.array([0, 1.95])
POINT2 = np.array([0, 1.05])  # Initial middle point (will be updated dynamically)
POINT3 = np.array([0, 0.15])  # Initial bottom point (mass)
GRAVITY = 9.81
LENGTH = np.linalg.norm(POINT1 - POINT3)
TIME_STEP = 0.02

# Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# State variables
grabbing = False

# Simulation variables
theta = np.arctan2(POINT1[1] - POINT3[1], POINT1[0] - POINT3[0])
omega = 0

def update_simulation():
    global theta, omega, POINT2, POINT3
    alpha = -(GRAVITY / LENGTH) * np.sin(theta)
    omega += alpha * TIME_STEP
    theta += omega * TIME_STEP
    x = LENGTH * np.sin(theta)
    y = -LENGTH * np.cos(theta) + POINT1[1]
    POINT3 = np.array([x, y])
    POINT2 = (POINT1 + POINT3) / 2

def main():
    global grabbing, POINT3

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_pos = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y])

                if grabbing:
                    # Update Point 3 position to hand position
                    POINT3 = hand_pos * np.array([640, 480])
                    POINT3[1] = 480 - POINT3[1]
                    POINT3 /= 100  # Scale for visualization
                    POINT2 = (POINT1 + POINT3) / 2
                else:
                    # Check if hand is near Point 3 to grab
                    hand_distance = np.linalg.norm(hand_pos * np.array([640, 480]) - POINT3 * 100)
                    if hand_distance < 50:
                        grabbing = True

        else:
            grabbing = False

        if not grabbing:
            update_simulation()

        # draw pendulum
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.line(image, tuple((POINT1 * 100).astype(int)), tuple((POINT2 * 100).astype(int)), (255, 0, 0), 3)
        cv2.line(image, tuple((POINT2 * 100).astype(int)), tuple((POINT3 * 100).astype(int)), (255, 0, 0), 3)
        cv2.circle(image, tuple((POINT3 * 100).astype(int)), 10, (0, 0, 255), -1)
        cv2.imshow('Pendulum Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
