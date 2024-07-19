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

# Track mass positions
mass_position_1 = 0
mass_velocity_1 = 0.0
hand_grabbed = False

# Track time
start_time = time.time()
times = []
positions_1 = []

def simulate_system(t, y):
    dydt = [
        y[1],
        -k1/m1*(y[0]) - b1/m1*y[1]
    ]
    return dydt

def grab_position_1():
    def state_space_model(t, x):
        dxdt = [
            x[1],
            (-k1 * (x[0]) - b1 * x[1]) / m1
        ]
        return dxdt
    
    def simulate_state_space(t_span, x0):
        sol = solve_ivp(state_space_model, t_span, x0, t_eval=np.linspace(t_span[0], t_span[1], 1000))
        return sol.t, sol.y[0], sol.y[1]
    
    x0 = [mass_position_1, mass_velocity_1]
    t_span = [0, 0.1]
    t_sim, pos_grab_1, vel_grab_1 = simulate_state_space(t_span, x0)

    return pos_grab_1[-1], vel_grab_1[-1]

def apply_impulse_response(mass_position_1, mass_velocity_1, impulse_force=0.0, impulse_duration=0.01):
    def impulse_system(t, y):
        u = impulse_force if t <= impulse_duration else 0
        dydt = [
            y[1],
            -k1/m1*y[0] - b1/m1*y[1] + u/m1
        ]
        return dydt

    t_span = [0, 0.1]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(impulse_system, t_span, [mass_position_1, mass_velocity_1], t_eval=t_eval)
    return sol.t, sol.y[0], sol.y[1]

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.6, maxHands=1)

# Plot setup
plt.ion()  # Interactive mode ON
fig, ax1 = plt.subplots()
line1, = ax1.plot([], [], 'r-')
ax1.set_xlim(0, 10)
ax1.set_ylim(-2, 2)
ax1.set_ylabel('Position 1')
ax1.set_xlabel('Time (s)')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_width // 2

last_sim_time = 0

# Initialize Crazyflie URI
URI1 = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E7E9')
cflib.crtp.init_drivers()
logging.basicConfig(level=logging.ERROR)

def position_callback(timestamp, data, logconf):
    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    print(f'Position: x={x:.2f}, y={y:.2f}, z={z:.2f}')

def start_position_logging(scf):
    log_conf = LogConfig(name='Position', period_in_ms=200)
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
    global mass_position_1, mass_velocity_1, hand_grabbed, last_sim_time
    with SyncCrazyflie(URI1, cf=Crazyflie(rw_cache='./cache')) as scf1:
        reset_estimator(scf1)
        start_position_logging(scf1)

        hlc1 = scf1.cf.high_level_commander
        hlc1.takeoff(0.5, 2.0)
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
                    lmList = hand['lmList']
                    fingers = detector.fingersUp(hand)

                    if fingers == [0, 0, 0, 0, 0]:
                        hand_grabbed = True
                        palm = lmList[0]
                        mass_position_1 = (palm[0] - center_x) / 150
                        mass_velocity_1 = 0
                        mass_position_1, mass_velocity_1 = grab_position_1()
                        times.append(current_time)
                        positions_1.append(mass_position_1)
                    else:
                        if hand_grabbed:
                            hand_grabbed = False
                            x0 = [mass_position_1, mass_velocity_1]
                            t_sim, pos_sim_1, vel_sim_1 = apply_impulse_response(mass_position_1, mass_velocity_1)
                            mass_position_1 = pos_sim_1[-1]
                            mass_velocity_1 = vel_sim_1[-1]
                            
                            real_time_start = current_time
                            real_times = [real_time_start + t for t in t_sim]
                            times.extend(real_times)
                            positions_1.extend(pos_sim_1)
                            last_sim_time = real_time_start + t_sim[-1]  

                if not hand_grabbed:
                    elapsed_time = current_time - last_sim_time
                    t_span = [0, elapsed_time]
                    x0 = [mass_position_1, mass_velocity_1]
                    sol = solve_ivp(simulate_system, t_span, x0, t_eval=[elapsed_time])
                    mass_position_1 = sol.y[0, -1] 
                    mass_velocity_1 = sol.y[1, -1]

                    times.append(current_time)
                    positions_1.append(mass_position_1)
                    last_sim_time = current_time

                if times and positions_1:
                    line1.set_xdata(times)
                    line1.set_ydata(positions_1)
                    ax1.set_xlim(max(times) - 15, max(times) + 1)
                    ax1.set_ylim(min(positions_1) - 1, max(positions_1) + 1)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                cv2.putText(img, f'Mass Position 1: {mass_position_1:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(img, f'Hand Grabbed: {hand_grabbed}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Image", img)

                hlc1.go_to(mass_position_1, 0, 0.75, 0, 0.5, relative=False)

        finally:
            hlc1.land(0, 2.0)
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()  # Interactive mode OFF
            plt.show()

if __name__ == "__main__":
    main()
