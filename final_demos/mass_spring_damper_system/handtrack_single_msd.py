import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cvzone.HandTrackingModule import HandDetector
import cv2
import time

# Parameters for mass spring damper system
m = 1.0  # mass
k = 5.5  # spring constant
b = 0.3  # damping coefficient

# Hand tracking
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initial position and velocity
mass_position = 0.0
mass_velocity = 0.0
hand_grabbed = False

start_time = time.time()
times = []
positions = []

def simulate_system(t, y):
    dydt = [y[1], -k/m * y[0] - b/m * y[1]]
    return dydt

plt.ion()  # Interactive mode ON
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_width // 2

last_sim_time = 0

def apply_impulse_response(mass_position, mass_velocity, impulse_force=1.0, impulse_duration=0.1):
    def impulse_system(t, y):
        impulse = impulse_force if t <= impulse_duration else 0
        dydt = [y[1], -k/m * y[0] - b/m * y[1] + impulse/m]
        return dydt

    t_span = [0, 0.1]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(impulse_system, t_span, [mass_position, mass_velocity], t_eval=t_eval)
    return sol.t, sol.y[0], sol.y[1]

while True:
    current_time = time.time() - start_time
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        fingers = detector.fingersUp(hand)

        if fingers == [0, 0, 0, 0, 0]:
            hand_grabbed = True
            wrist = lmList[9]
            mass_position = (wrist[0] - center_x) / 150
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

    if times and positions:
        line.set_xdata(times)
        line.set_ydata(positions)
        ax.set_xlim(max(times) - 15, max(times) + 1)
        ax.set_ylim(min(positions) - 1, max(positions) + 1)
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Draw mass
    square_size = 50
    pos = int(center_x + mass_position * 150)
    bottom_y = frame_height - square_size

    cv2.rectangle(img, (pos - square_size//2, bottom_y - square_size), (pos + square_size//2, bottom_y), (30, 30, 255), -1)
    cv2.putText(img, "m", (pos - 17, bottom_y - 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    # Draw spring and damper
    spring_color = (0, 255, 255)
    damper_color = (255, 0, 255)
    
    # Spring (left)
    cv2.line(img, (0, bottom_y - square_size//2 - 10), (pos - square_size//2, bottom_y - square_size//2 - 10), spring_color, 2)
    
    # Damper (left)
    cv2.line(img, (0, bottom_y - square_size//2 + 10), (pos - square_size//2, bottom_y - square_size//2 + 10), damper_color, 2)
    
    # Display information
    cv2.putText(img, f'Mass Position: {mass_position:.2f}', (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(img, f'Hand Grabbed: {hand_grabbed}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Interactive mode OFF
plt.show()
