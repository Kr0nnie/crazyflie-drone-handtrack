import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cvzone.HandTrackingModule import HandDetector
import cv2
import time

# Parameters for the mass-spring-damper system
m = 1.0  # mass
k = 1.0  # spring constant
b = 0.1  # damping coefficient

# Hand tracking setup
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Variables to track the mass state
mass_position = 0.0
mass_velocity = 0.0
hand_grabbed = False

# Time tracking
start_time = time.time()
times = []
positions = []

def simulate_system(t, y):
    dydt = [y[1], -k/m * y[0] - b/m * y[1]]
    return dydt

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position')

# Get the frame width to center the hand position
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
center_x = frame_width // 2

# Initialize a variable to store the last simulation end time
last_sim_time = 0

while True:
    current_time = time.time() - start_time
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        lmList = hand['lmList']  # List of 21 Landmark points
        fingers = detector.fingersUp(hand)

        if fingers == [0, 0, 0, 0, 0]:  # All fingers down
            hand_grabbed = True
            wrist = lmList[0]
            mass_position = (wrist[0] - center_x) / 100  # Centering and scaling position
            mass_velocity = 0  # Resetting velocity when grabbed
        else:
            if hand_grabbed:
                hand_grabbed = False
                x0 = [mass_position, mass_velocity]
                t_span = [0, 0.1]  # Simulate for a longer period to capture oscillations
                t_eval = np.linspace(t_span[0], t_span[1], 1000)
                sol = solve_ivp(simulate_system, t_span, x0, t_eval=t_eval)
                mass_position = sol.y[0, -1]
                mass_velocity = sol.y[1, -1]
                # Append the simulated results to times and positions
                real_time_start = current_time
                real_times = [real_time_start + (t - t_eval[0]) for t in t_eval]
                times.extend(real_times)
                positions.extend(sol.y[0])
                last_sim_time = real_time_start + t_eval[-1]  # Update the last simulation time

    if not hand_grabbed:
        # Continue the simulation using the last state
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
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()
