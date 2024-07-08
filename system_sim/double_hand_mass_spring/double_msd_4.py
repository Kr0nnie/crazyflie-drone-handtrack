#not good

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

# Parameters for mass-spring-damper system
m1 = 1.0  # mass 1
k1 = 6.0  # spring constant 1
b1 = 0.3  # damping coefficient 1
m2 = 1.0  # mass 2
k2 = 6.0  # spring constant 2
b2 = 0.3  # damping coefficient 2

# Track mass positions
mass_position_1 = -0.5
mass_velocity_1 = 0.0
mass_position_2 = 0.5
mass_velocity_2 = 0.0
hand_grabbed = False

# Track time
start_time = time.time()
times = []
positions_1 = []
positions_2 = []

def simulate_system(t, y):
    dydt = [
        y[2],
        y[3],
        -k1/m1*(y[0]+0.5) - b1/m1*y[2] + k2/m1*(y[1]-0.5-(y[0]+0.5)) + b2/m1*(y[3]-y[2]),
        -k2/m2*(y[1]-0.5-(y[0]+0.5)) - b2/m2*(y[3]-y[2])
    ]
    return dydt

def grab_position_1(mass_position_2):
    def state_space_model(t, x, u):
        dxdt = [
            x[1],
            (-k1 * (x[0] + 0.5) - b1 * x[1] + k2 * (u - 0.5 - (x[0] + 0.5)) + b2 * (0 - x[1])) / m1
        ]
        return dxdt
    
    def simulate_state_space(t_span, x0, u):
        sol = solve_ivp(state_space_model, t_span, x0, args=(u,), t_eval=np.linspace(t_span[0], t_span[1], 1000))
        return sol.t, sol.y[0], sol.y[1]
    
    x0 = [mass_position_1, mass_velocity_1]
    t_span = [0, 0.1]
    t_sim, pos_grab_1, vel_grab_1 = simulate_state_space(t_span, x0, mass_position_2)

    return pos_grab_1[-1], vel_grab_1[-1]

def apply_impulse_response(mass_position_1, mass_velocity_1, mass_position_2, mass_velocity_2, impulse_force=0.0, impulse_duration=0.01):
    def impulse_system(t, y):
        dydt = [
            y[2],
            y[3],
            -k1/m1*(y[0]+0.5) - b1/m1*y[2] + k2/m1*(y[1]-0.5-(y[0]+0.5)) + b2/m1*(y[3]-y[2]) + impulse_force/m1 if t < impulse_duration else 0,
            -k2/m2*(y[1]-0.5-(y[0]+0.5)) - b2/m2*(y[3]-y[2]) + impulse_force/m2 if t < impulse_duration else 0
        ]
        return dydt

    t_span = [0, 0.1]
    y0 = [mass_position_1, mass_position_2, mass_velocity_1, mass_velocity_2]
    sol = solve_ivp(impulse_system, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 1000))

    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]

def main():
    cflib.crtp.init_drivers()

    uri1 = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E7E7')
    uri2 = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

    hand_detector = HandDetector(detectionCon=0.8, maxHands=1)
    cap = cv2.VideoCapture(0)

    with SyncCrazyflie(uri1, cf=Crazyflie(rw_cache='./cache')) as scf1, SyncCrazyflie(uri2, cf=Crazyflie(rw_cache='./cache')) as scf2:
        with MotionCommander(scf1) as hlc1, MotionCommander(scf2) as hlc2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
            line1, = ax1.plot([], [], 'b-')
            line2, = ax2.plot([], [], 'r-')
            plt.ion()

            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    break
                
                hands = hand_detector.findHands(img, draw=False)
                if hands:
                    lmList = hands[0]['lmList']
                    if len(lmList) >= 9:
                        x_thumb, y_thumb = lmList[4][:2]
                        x_index, y_index = lmList[8][:2]
                        distance = np.sqrt((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2)

                        if distance < 50:
                            hand_grabbed = True
                            mass_position_2 = (x_thumb - 320) / 320 * 1.0
                            mass_position_1, mass_velocity_1 = grab_position_1(mass_position_2)
                        else:
                            hand_grabbed = False
                    else:
                        hand_grabbed = False

                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time > 0.1:
                    start_time = current_time
                    t_span = [0, elapsed_time]
                    x0 = [mass_position_1, mass_position_2, mass_velocity_1, mass_velocity_2]
                    sol = solve_ivp(simulate_system, t_span, x0, t_eval=[elapsed_time])
                    mass_position_1 = sol.y[0, -1] 
                    mass_velocity_1 = sol.y[2, -1]
                    mass_position_2 = sol.y[1, -1] 
                    mass_velocity_2 = sol.y[3, -1]

                    # drones gotta be 0.3 meters apart
                    if abs(mass_position_2 - mass_position_1) < 0.3:
                        if mass_position_2 > mass_position_1:
                            mass_position_2 = mass_position_1 + 0.3
                        else:
                            mass_position_1 = mass_position_2 + 0.3

                    times.append(current_time)
                    positions_1.append(mass_position_1)
                    positions_2.append(mass_position_2)

                    if times and positions_1 and positions_2:
                        line1.set_xdata(times)
                        line1.set_ydata(positions_1)
                        line2.set_xdata(times)
                        line2.set_ydata(positions_2)
                        ax1.set_xlim(max(times) - 15, max(times) + 1)
                        ax1.set_ylim(min(positions_1) - 1, max(positions_1) + 1)
                        ax2.set_xlim(max(times) - 15, max(times) + 1)
                        ax2.set_ylim(min(positions_2) - 1, max(positions_2) + 1)
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                cv2.putText(img, f'Mass Position 1: {mass_position_1:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(img, f'Mass Position 2: {mass_position_2:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(img, f'Hand Grabbed: {hand_grabbed}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Image", img)

                hlc1.go_to(mass_position_1, 0, 0.75, 0, 0.5, relative=False)
                hlc2.go_to(mass_position_2, 0, 0.75, 0, 0.5, relative=False)

            hlc1.land(0, 2.0)
            hlc2.land(0, 2.0)
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    main()
