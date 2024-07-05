import logging
import time
import cv2
import math
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

# Parameters of msd system
m = 1  # mass
k = 6  # spring 
b = 0.2  # damping

# Initialize shits
mass_position = 0.0
mass_velocity = 0.0
hand_grabbed = False

# Track time
start_time = time.time()
times = []
positions = []

def simulate_system(t, y):
    dydt = [y[1], -k/m * y[0] - b/m * y[1]]
    return dydt

def apply_impulse_response(mass_position, mass_velocity, impulse_force=1.0, impulse_duration=0.1):
    def impulse_system(t, y):
        impulse = impulse_force if t <= impulse_duration else 0
        dydt = [y[1], -k/m * y[0] - b/m * y[1] + impulse/m]
        return dydt

    t_span = [0, 0.1]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(impulse_system, t_span, [mass_position, mass_velocity], t_eval=t_eval)
    return sol.t, sol.y[0], sol.y[1]

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

cflib.crtp.init_drivers()

logging.basicConfig(level=logging.ERROR)

def position_callback(timestamp, data, logconf):
    #global drone_x
    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    #drone_x = x
    print(f'Position: x={x:.2f}, y={y:.2f}, z={z:.2f}')

def start_position_logging(scf):
    log_conf = LogConfig(name='Position', period_in_ms=50)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(position_callback)
    log_conf.start()

def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def main():
    global mass_position, mass_velocity, hand_grabbed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    detector = HandDetector(detectionCon=0.8, maxHands=1)

    # Initialize msd shit
    plt.ion()  # Interactive mode ON
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-')
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    center_x = frame_width // 2
    last_sim_time = 0

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        reset_estimator(scf)
        start_position_logging(scf)

        hlc = scf.cf.high_level_commander
        hlc.takeoff(0.5, 2.0)
        time.sleep(1)

        try:
            while True:
                current_time = time.time() - start_time
                success, img = cap.read()
                if not success:
                    print("Error: Could not read frame from video capture.")
                    break
                img = cv2.flip(img, 1)
                hands, img = detector.findHands(img)
                cv2.imshow("Image", img)  # Show image
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if hands:
                    hand = hands[0]
                    lmList = hand['lmList']  # List of 21 Landmark points
                    fingers = detector.fingersUp(hand)

                    if fingers == [0, 0, 0, 0, 0]:  # All fingers down
                        hand_grabbed = True
                        wrist = lmList[0]
                        mass_position = (wrist[0] - center_x) / 300  # Centering and scaling position
                        mass_velocity = 0 
                        times.append(current_time)
                        positions.append(mass_position)
                    else:
                        if hand_grabbed:
                            hand_grabbed = False
                            x0 = [mass_position, mass_velocity]
                            t_sim, pos_sim, vel_sim = apply_impulse_response(mass_position, mass_velocity)
                            mass_position = pos_sim[-1]
                            mass_velocity = vel_sim[-1]
                            real_time_start = current_time
                            real_times = [real_time_start + t for t in t_sim]
                            times.extend(real_times)
                            positions.extend(pos_sim)
                            last_sim_time = real_time_start + t_sim[-1]

                if not hand_grabbed:
                    elapsed_time = current_time - last_sim_time
                    t_span = [0, elapsed_time]
                    x0 = [mass_position, mass_velocity]
                    sol = solve_ivp(simulate_system, t_span, x0, t_eval=[elapsed_time])
                    mass_position = sol.y[0, -1]
                    mass_velocity = sol.y[1, -1]
                    times.append(current_time)
                    positions.append(mass_position)
                    last_sim_time = current_time

                # Update plot
                if times and positions:
                    line.set_xdata(times)
                    line.set_ydata(positions)
                    ax.set_xlim(0, max(times) + 1)
                    ax.set_ylim(min(positions) - 1, max(positions) + 1)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                cv2.putText(img, f'Mass Position: {mass_position:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(img, f'Hand Grabbed: {hand_grabbed}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                #print(mass_position)

                # Control the drone based on mass position
                #scaled_x_position = drone_x - mass_position  # Adjust drone x-coordinate based on mass position
                hlc.go_to(mass_position, 0, 0.75, 0, 0.5, relative=False)

        finally:
            hlc.land(0, 2.0)
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()  # Interactive mode OFF
            plt.show()

if __name__ == "__main__":
    main()
