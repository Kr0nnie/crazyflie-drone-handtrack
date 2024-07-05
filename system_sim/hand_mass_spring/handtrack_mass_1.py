# + hand position and grabbing works well
# - mass spring damping reaction doesnt work


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cvzone.HandTrackingModule import HandDetector
import cv2
import time

# parameters for the mass-spring-damper system
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
            mass_position = wrist[0]  # Using x position of wrist to simulate mass position
            mass_velocity = 0  # Resetting velocity when grabbed
        else:
            if hand_grabbed:
                hand_grabbed = False
                x0 = [mass_position, mass_velocity]
                sol = solve_ivp(simulate_system, [0, 0.1], x0, t_eval=[0, 0.1])
                mass_position = sol.y[0, -1]
                mass_velocity = sol.y[1, -1]
    
    times.append(current_time)
    positions.append(mass_position)

    # Update plot
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


