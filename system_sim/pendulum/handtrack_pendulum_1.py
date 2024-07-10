import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
GRAVITY = 9.81
DAMPING = 0.05
TIME_STEP = 0.05

# Initial points for the pendulum
POINT1 = np.array([0, 1.95])
POINT2 = np.array([0, 1.05])  # Initial middle point (will be updated dynamically)
POINT3 = np.array([0, 0.15])  # Initial bottom point (mass)
LENGTH = np.linalg.norm(POINT1 - POINT3)

# Hand tracking init
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
hand_grabbed = False

# Initial conditions
theta = np.arctan2(POINT1[1] - POINT3[1], POINT1[0] - POINT3[0])
omega = 0
initial_theta = theta
initial_length = LENGTH

# Convert screen width to initial point for pendulum
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
center_x = frame_width // 2
POINT1 = np.array([center_x / 100, 0])

def pendulum_state_space(t, y):
    theta, omega = y
    dydt = [omega, -DAMPING * omega - (GRAVITY / LENGTH) * np.sin(theta)]
    return dydt

def pendulum_simulation(theta, omega, t_eval):
    y0 = [theta, omega]
    sol = solve_ivp(pendulum_state_space, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval)
    return sol.y[0], sol.y[1]

def main():
    global hand_grabbed, POINT3, POINT2, LENGTH, theta, omega, initial_theta, initial_length

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    point, = ax.plot([], [], 'ro')
    ax.set_xlim(-LENGTH - 0.1, LENGTH + 0.1)
    ax.set_ylim(-LENGTH - 0.1, LENGTH + 0.1)
    ax.set_aspect('equal', 'box')
    ax.set_title('Pendulum Simulation')
    fig.show()

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
                POINT3 = np.array([(palm[1]) / 100, (palm[0] - POINT1[0]) / 100])
                LENGTH = np.linalg.norm(POINT1 - POINT3)
                initial_length = LENGTH
                theta = np.arctan2(POINT1[1] - POINT3[1], POINT1[0] - POINT3[0])
                initial_theta = theta
                omega = 0
                POINT2 = (POINT1 + POINT3) / 2
                print("point 3:", POINT3 , "...palm",palm[1], ",", palm[0])

            else:
                if hand_grabbed:
                    hand_grabbed = False
                    theta = initial_theta
                    LENGTH = initial_length

        t_eval = np.linspace(0, TIME_STEP, 10)
        if not hand_grabbed:
            theta_vals, omega_vals = pendulum_simulation(theta, omega, t_eval)
            theta = theta_vals[-1]
            omega = omega_vals[-1]
            print("point 3:", POINT3)

        x3 = -LENGTH * np.sin(theta)
        y3 = -LENGTH * np.cos(theta)
        POINT3 = -np.array([x3, y3]) + POINT1
        POINT2 = (POINT1 + POINT3) / 2

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.line(img, tuple((POINT1 * 100).astype(int)), tuple((POINT3 * 100).astype(int)), (255, 0, 0), 3)
        cv2.circle(img, tuple((POINT1 * 100).astype(int)), 10, (0, 0, 255), 2)
        cv2.circle(img, tuple((POINT2 * 100).astype(int)), 10, (0, 0, 255), 2)
        cv2.circle(img, tuple((POINT3 * 100).astype(int)), 10, (0, 0, 255), -1)
        cv2.imshow('Pendulum Hand Tracking', img)

        line.set_data([0, x3], [0, y3])
        point.set_data(x3, y3)
        fig.canvas.draw()
        fig.canvas.flush_events()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
