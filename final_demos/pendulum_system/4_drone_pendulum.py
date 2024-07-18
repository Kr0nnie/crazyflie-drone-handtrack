import logging
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cvzone.HandTrackingModule import HandDetector
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper
from cflib.positioning.motion_commander import MotionCommander

# Constants
GRAVITY = 2
DAMPING = 0.05
TIME_STEP = 0.05
MIN_LENGTH = 1.5

# Pendulum initial conditions
init_POINT1 = np.array([0, 1.400])
init_POINT4 = np.array([0, -0.500])
LENGTH = np.linalg.norm(init_POINT1 - init_POINT4)
init_POINT2 = (2 * init_POINT1 + init_POINT4) / 3
init_POINT3 = (init_POINT1 + 2 * init_POINT4) / 3
theta = 0
omega = 0

# Initialize hand tracking
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.7, maxHands=1)
hand_grabbed = False

# Convert screen width to initial point for pendulum
center_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
center_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
POINT1 = np.array([center_x / 100, 0])

# Initialize Crazyflie URIs
URI1 = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E6')
URI2 = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E7E9')
URI3 = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E9')
URI4 = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E7E9')
cflib.crtp.init_drivers()
logging.basicConfig(level=logging.ERROR)
print("Initializing drones...")

# Function for pendulum dynamics
def pendulum_state_space(t, y):
    theta, omega = y
    dydt = [omega, -DAMPING * omega - (GRAVITY / LENGTH) * np.sin(theta)]
    return dydt

# Function for pendulum simulation
def pendulum_simulation(theta, omega, t_eval):
    y0 = [theta, omega]
    sol = solve_ivp(pendulum_state_space, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval)
    return sol.y[0], sol.y[1]

# Function to reset estimator
def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

# Function to start position logging
def start_position_logging(scf):
    log_conf = LogConfig(name='Position', period_in_ms=200)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.start()

# Main function
def main():
    global hand_grabbed, POINT4, POINT2, POINT3, LENGTH, theta, omega

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    point, = ax.plot([], [], 'ro')
    ax.set_xlim(-3, 3)
    ax.set_ylim(5, -0.1)
    ax.set_aspect('equal', 'box')
    ax.set_title('Pendulum Simulation')
    fig.show()

    with SyncCrazyflie(URI1, cf=Crazyflie(rw_cache='./cache')) as scf1, \
         SyncCrazyflie(URI2, cf=Crazyflie(rw_cache='./cache')) as scf2, \
         SyncCrazyflie(URI3, cf=Crazyflie(rw_cache='./cache')) as scf3, \
         SyncCrazyflie(URI4, cf=Crazyflie(rw_cache='./cache')) as scf4:
        
        reset_estimator(scf1)
        start_position_logging(scf1)
        reset_estimator(scf2)
        start_position_logging(scf2)
        reset_estimator(scf3)
        start_position_logging(scf3)
        reset_estimator(scf4)
        start_position_logging(scf4)

        hlc1 = scf1.cf.high_level_commander
        hlc2 = scf2.cf.high_level_commander
        hlc3 = scf3.cf.high_level_commander
        hlc4 = scf4.cf.high_level_commander
        
        # Take off with delays
        
        hlc4.takeoff(0.9, 2.0)
        hlc4.go_to(init_POINT4[0], init_POINT4[1], 0.6, 0, 3.0, relative=False)
        print("POINT4:", init_POINT4)        
        time.sleep(3)

        hlc3.takeoff(0.6, 2.0)
        hlc3.go_to(init_POINT3[0], init_POINT3[1], 0.9, 0, 3.0, relative=False)
        print("POINT3:", init_POINT3)        
        time.sleep(3)

        hlc2.takeoff(1.2, 2.0)
        hlc2.go_to(init_POINT2[0], init_POINT2[1], 1.2, 0, 3.0, relative=False)
        print("POINT2:", init_POINT2)        
        time.sleep(3)

        hlc1.takeoff(1.5, 2.0)
        hlc1.go_to(init_POINT1[0], init_POINT1[1], 1.5, 0, 3.0, relative=False)
        print("POINT1:", init_POINT1)
        time.sleep(2)

        try:
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break

                img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
                hands, img = detector.findHands(img)

                POINT4 = init_POINT4

                if hands:
                    hand = hands[0]
                    lmList = hand['lmList']
                    fingers = detector.fingersUp(hand)
                    palm = lmList[9]

                    if fingers == [0, 0, 0, 0, 0]:
                        hand_grabbed = True
                        POINT4 = np.array([(palm[0]) / 100, (palm[1]) / 100])
                        LENGTH = np.linalg.norm(POINT1 - POINT4)
                        if LENGTH < MIN_LENGTH:
                            LENGTH = MIN_LENGTH
                            direction = (POINT4 - POINT1) / np.linalg.norm(POINT4 - POINT1)
                            POINT4 = POINT1 + direction * LENGTH
                        theta = - np.arctan2(POINT1[1] - POINT4[1], POINT1[0] - POINT4[0]) - (1/2 * np.pi)
                        omega = 0
                        POINT2 = (2 * POINT1 + POINT4) / 3
                        POINT3 = (POINT1 + 2 * POINT4) / 3

                    else:
                        if hand_grabbed:
                            hand_grabbed = False

                t_eval = np.linspace(0, TIME_STEP, 10)
                if not hand_grabbed:
                    theta_vals, omega_vals = pendulum_simulation(theta, omega, t_eval)
                    theta = theta_vals[-1]
                    omega = omega_vals[-1]

                x4 = LENGTH * np.sin(theta)
                y4 = LENGTH * np.cos(theta)
                if not hand_grabbed:
                    POINT4 = np.array([x4, y4]) + POINT1
                    if np.linalg.norm(POINT1 - POINT4) < MIN_LENGTH:
                        POINT4 = POINT1 + (POINT4 - POINT1) / np.linalg.norm(POINT4 - POINT1) * MIN_LENGTH
                        LENGTH = MIN_LENGTH
                POINT2 = (2 * POINT1 + POINT4) / 3
                POINT3 = (POINT1 + 2 * POINT4) / 3

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.line(img, (int(POINT1[0] * 100), int(POINT1[1] * 100)), (int(POINT4[0] * 100), int(POINT4[1] * 100)), (255, 0, 0), 3)
                cv2.circle(img, (int(POINT1[0] * 100), int(POINT1[1] * 100)), 10, (0, 0, 255), 2)
                cv2.circle(img, (int(POINT2[0] * 100), int(POINT2[1] * 100)), 10, (0, 0, 255), 2)
                cv2.circle(img, (int(POINT3[0] * 100), int(POINT3[1] * 100)), 10, (0, 255, 0), 2)
                cv2.circle(img, (int(POINT4[0] * 100), int(POINT4[1] * 100)), 10, (0, 0, 255), -1)

                cv2.imshow('Pendulum Hand Tracking', img)

                line.set_data([0, x4], [0, y4])
                point.set_data(x4, y4)
                fig.canvas.draw()
                fig.canvas.flush_events()

                P1 = (POINT1 / 1.6) - (center_x / 160)
                P2 = (POINT2 / 1.6) - (center_x / 160)
                P3 = (POINT3 / 1.6) - (center_x / 160)
                P4 = (POINT4 / 1.6) - (center_x / 160)
                P1[1] = -( P1[1] + (center_y / 350) )
                P2[1] = -( P2[1] + (center_y / 350) ) 
                P3[1] = -( P3[1] + (center_y / 350) )                
                P4[1] = -( P4[1] + (center_y / 350) )
                print("P1:", P1, "  P2:", P2, "  P3:", P3, "  P4:", P4)

                hlc1.go_to(P1[0], P1[1], 1.5, 0, 0.5, relative=False)
                hlc2.go_to(P2[0], P2[1], 1.2, 0, 0.5, relative=False)
                hlc3.go_to(P3[0], P3[1], 0.9, 0, 0.5, relative=False)
                hlc4.go_to(P4[0], P4[1], 0.6, 0, 0.5, relative=False)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            hlc1.land(0, 2.0)
            hlc2.land(0, 2.0)
            hlc3.land(0, 2.0)
            hlc4.land(0, 2.0)
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    main()
