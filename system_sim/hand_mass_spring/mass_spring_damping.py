import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from control.matlab import ss, impulse, step

# Parameters for the mass-spring-damper system
m = 1.0  # mass
k = 1.0  # spring constant
b = 0.1  # damping coefficient

# System matrices
A = np.array([
    [0, 1],
    [-k/m, -b/m]
])
B = np.array([
    [0],
    [1/m]
])
C = np.eye(2)
D = np.array([
    [0],
    [0]
])

# State-space system
sys = ss(A, B, C, D)

# Initial conditions
x0 = np.array([1.0, 0.0])  # initial displacement and velocity

# Time vector
t = np.linspace(0, 20, 1000)

# Impulse response
y_impulse, t_impulse = impulse(sys, t, X0=x0)

# Step response
y_step, t_step = step(sys, t, X0=x0)

# Initial response using solve_ivp
def mass_spring_damper(t, y):
    return [y[1], (-b * y[1] - k * y[0]) / m]

sol = solve_ivp(mass_spring_damper, [0, 20], x0, t_eval=t)
displacement_initial = sol.y[0]
velocity_initial = sol.y[1]

# Extracting displacement and velocity
displacement_impulse = y_impulse[:, 0]
velocity_impulse = y_impulse[:, 1]

displacement_step = y_step[:, 0]
velocity_step = y_step[:, 1]

# Plotting the impulse response
plt.figure()
plt.plot(t_impulse, displacement_impulse, label='Displacement (impulse)')
plt.plot(t_impulse, velocity_impulse, label='Velocity (impulse)')
plt.xlabel('Time [s]')
plt.ylabel('Response')
plt.legend()
plt.title('Impulse Response of Mass-Spring-Damper System')
plt.grid(True)

# Plotting the step response
plt.figure()
plt.plot(t_step, displacement_step, label='Displacement (step)')
plt.plot(t_step, velocity_step, label='Velocity (step)')
plt.xlabel('Time [s]')
plt.ylabel('Response')
plt.legend()
plt.title('Step Response of Mass-Spring-Damper System')
plt.grid(True)

# Plotting the initial condition response
plt.figure()
plt.plot(t, displacement_initial, label='Displacement (initial)')
plt.plot(t, velocity_initial, label='Velocity (initial)')
plt.xlabel('Time [s]')
plt.ylabel('Response')
plt.legend()
plt.title('Initial Condition Response of Mass-Spring-Damper System')
plt.grid(True)

plt.show()
