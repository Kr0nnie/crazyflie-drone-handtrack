import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
from scipy.integrate import solve_ivp

m1 = 1.0  # mass 1
k1 = 5.0  # spring constant 1
b1 = 0.5  # damping coefficient 1
m2 = 1.0  # mass 2
k2 = 3.0  # spring constant 2
b2 = 0.3  # damping coefficient 2

A = np.array([
    [0, 1, 0],
    [(-k1 -k2)/m1, (-b1 -b2)/m1, b2/b1],
    [k2/m2, b2/m2, -b2]
])

B = np.array([
    [0],
    [k1],
    [-k2]
])

C = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

D = np.array([
    [0],
    [0],
    [0]
])

sys = ss(A, B, C, D)

x0 = np.array([1, 0, 0])

t = np.linspace(0, 10, 1000)
y, t = impulse(sys, t, x0)

x1 = y[:,0]

plt.figure()
plt.plot(t, x1, label='x1')
plt.show()
