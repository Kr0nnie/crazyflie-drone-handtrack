import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
GRAVITY = 2
DAMPING = 0.05
TIME_STEP = 0.05

# IC for pendulum
POINT1 = np.array([0, 2.5])
POINT2 = np.array([0, 1.05])  
POINT3 = np.array([0, 0.15]) 
LENGTH = np.linalg.norm(POINT1 - POINT3)

# Initialize hand tracking
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.6, maxHands=1)
hand_grabbed = False

# Initial conditions
theta = 0
omega = 0

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
    global hand_grabbed, POINT3, POINT2, LENGTH, theta, omega

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    point, = ax.plot([], [], 'ro')
    ax.set_xlim(-3, 3)
    ax.set_ylim(5, -0.1)
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
            #print("point 3:", POINT3, "palm:", palm[1], ",", palm[0])

            if fingers == [0, 0, 0, 0, 0]:
                hand_grabbed = True
                POINT3 = np.array([(palm[0]) / 100, (palm[1]) / 100]) 
                LENGTH = np.linalg.norm(POINT1 - POINT3)
                theta = - np.arctan2(POINT1[1] - POINT3[1], POINT1[0] - POINT3[0]) - (2/4 * np.pi)
                omega = 0
                POINT2 = (POINT1 + POINT3) / 2
                #print("PP 3:", POINT3, "PPalm:", palm[0], ",", palm[1])

            else:
                if hand_grabbed:
                    hand_grabbed = False

        t_eval = np.linspace(0, TIME_STEP, 10)
        if not hand_grabbed:
            theta_vals, omega_vals = pendulum_simulation(theta, omega, t_eval)
            theta = theta_vals[-1]
            omega = omega_vals[-1]
            #print("point 3:", POINT3)

        x3 = LENGTH * np.sin(theta)
        y3 = LENGTH * np.cos(theta)
        if not hand_grabbed:
            POINT3 = np.array([x3, y3]) + POINT1
        POINT2 = (POINT1 + POINT3) / 2
        

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.line(img, (int(POINT1[0] * 100), int(POINT1[1] * 100)), (int(POINT3[0] * 100), int(POINT3[1] * 100)), (255, 0, 0), 3)
        cv2.circle(img, (int(POINT1[0] * 100), int(POINT1[1] * 100)), 10, (0, 0, 255), 2)
        cv2.circle(img, (int(POINT2[0] * 100), int(POINT2[1] * 100)), 10, (0, 0, 255), 2)
        cv2.circle(img, (int(POINT3[0] * 100), int(POINT3[1] * 100)), 10, (0, 0, 255), -1)
        cv2.imshow('Pendulum Hand Tracking', img)

        line.set_data([0, x3], [0, y3])
        point.set_data(x3, y3)
        fig.canvas.draw()
        fig.canvas.flush_events()

        P2 = (POINT2 / 1.5) - (center_x / 150)
        P3 = (POINT3 / 1.5) - (center_x / 150)
        P2[1] = -P2[1]
        P3[1] = -P3[1]
        print("P2:", P2, "  P3:", P3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
