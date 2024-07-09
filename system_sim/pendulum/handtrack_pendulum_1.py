import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

# Constants
POINT1 = np.array([0, 1.95])
POINT2 = np.array([0, 1.05])  # Initial middle point (will be updated dynamically)
POINT3 = np.array([0, 0.15])  # Initial bottom point (mass)
GRAVITY = 9.81
LENGTH = np.linalg.norm(POINT1 - POINT3)
TIME_STEP = 0.05

# Hand tracking init
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
hand_grabbed = False

theta = np.arctan2(POINT1[1] - POINT3[1], POINT1[0] - POINT3[0])
omega = 0

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
center_x = frame_width // 2
POINT1 = np.array([center_x / 100, 0])

initial_theta = theta
initial_length = LENGTH

def pendulum_simulation():
    global theta, omega, POINT2, POINT3, LENGTH
    alpha = -(GRAVITY / LENGTH) * np.sin(theta)
    omega += alpha * TIME_STEP
    theta += omega * TIME_STEP
    x3 = LENGTH * np.sin(theta)
    y3 = LENGTH * np.cos(theta)
    POINT3 = np.array([x3, y3]) + POINT1
    POINT2 = (POINT1 + POINT3) / 2

def main():
    global hand_grabbed, POINT3, POINT2, LENGTH, theta, omega, initial_theta, initial_length

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            lmList = hand['lmList']
            fingers = detector.fingersUp(hand)
            palm = lmList[9]

            if fingers == [0, 0, 0, 0, 0]:
                hand_grabbed = True
                POINT3 = np.array([(palm[0]) / 100, palm[1] / 100])
                LENGTH = np.linalg.norm(POINT1 - POINT3)
                initial_length = LENGTH
                theta = np.arctan2(POINT1[1] - POINT3[1], POINT1[0] - POINT3[0])
                initial_theta = theta
                omega = 0
                POINT2 = (POINT1 + POINT3) / 2
            else:
                if hand_grabbed:
                    hand_grabbed = False
                    theta = initial_theta
                    LENGTH = initial_length

        if not hand_grabbed:
            pendulum_simulation()

        # Draw pendulum
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.line(img, tuple((POINT1 * 100).astype(int)), tuple((POINT2 * 100).astype(int)), (255, 0, 0), 3)
        cv2.line(img, tuple((POINT2 * 100).astype(int)), tuple((POINT3 * 100).astype(int)), (255, 0, 0), 3)
        cv2.circle(img, tuple((POINT3 * 100).astype(int)), 10, (0, 0, 255), -1)
        cv2.imshow('Pendulum Hand Tracking', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
