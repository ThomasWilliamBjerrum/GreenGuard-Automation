# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:18:51 2025

@author: bentl
"""

import numpy as np
import matplotlib.pyplot as plt

# Vehicle Parameters
L = 0.3  # Wheelbase (m)
v_max = 1.0  # Maximum linear velocity (m/s)
delta_max = np.radians(30)  # Maximum steering angle (30 degrees)

# Control Gains
k_p = 0.3
k_alpha = 5.0
k_beta = -0.5

# Eigenvalue Analysis for Stability
A = np.array([[-k_p, 0, 0],
              [0, -(k_alpha - k_p), -k_beta],
              [0, -k_p, 0]])
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues of the error dynamics:", eigenvalues)
print("Real parts:", eigenvalues.real)
print("All real parts negative (stable)?", np.all(eigenvalues.real < 0))
print("Convergence time constant (rho):", 1 / abs(min(eigenvalues.real)))
if np.any(eigenvalues.imag != 0):
    print("Oscillation period (alpha, beta):", 2 * np.pi / abs(eigenvalues.imag[0]), "seconds")

# Circle Parameters
R = 1.0  # Radius of the unit circle (m)
omega_c = 0.2  # Angular speed of the goal point on the circle (rad/s)
v_nominal = omega_c * R  # Desired speed along the circle (m/s)

# Simulation Parameters
dt = 0.01  # Time step (s)
t_max = 30.0  # Simulation time (s)
t = np.arange(0, t_max, dt)
num_steps = len(t)

# Initial Conditions (start outside the circle)
x = 2.0  # Initial x position (m)
y = 2.0  # Initial y position (m)
theta = np.radians(0)  # Initial heading (radians)
state = np.array([x, y, theta])

# Arrays to Store Trajectory
trajectory = np.zeros((num_steps, 3))
trajectory[0, :] = state

# Goal: Moving point on the unit circle
def get_goal_point(t):
    # Goal moves along the circle at constant angular speed
    theta_g = omega_c * t
    x_g = R * np.cos(theta_g)
    y_g = R * np.sin(theta_g)
    # Tangent direction for orientation
    theta_g_orientation = np.arctan2(np.cos(theta_g), -np.sin(theta_g))
    return x_g, y_g, theta_g_orientation

# Control Law
def control_law(state, x_g, y_g, theta_g):
    x, y, theta = state
    # Distance to goal
    rho = np.sqrt((x - x_g)**2 + (y - y_g)**2)
    # Angle to goal
    alpha = np.arctan2(y_g - y, x_g - x) - theta
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # Normalize to [-pi, pi]
    # Angle between heading and goal orientation
    beta = theta_g - theta
    beta = np.arctan2(np.sin(beta), np.cos(beta))
    
    # Control inputs
    v = k_p * rho + v_nominal
    v = np.clip(v, 0, v_max)  # Limit velocity
    omega = k_alpha * alpha + k_beta * beta
    
    # Compute steering angle
    if v < 1e-3:  # Avoid division by zero
        delta = 0
    else:
        delta = np.arctan((omega * L) / v)
        delta = np.clip(delta, -delta_max, delta_max)
    
    return v, delta

# Simulation Loop
for i in range(1, num_steps):
    # Current state
    x, y, theta = state
    
    # Get moving goal point
    x_g, y_g, theta_g = get_goal_point(t[i])
    
    # Compute control inputs
    v, delta = control_law(state, x_g, y_g, theta_g)
    
    # Update state (bicycle model kinematics)
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = (v / L) * np.tan(delta)
    
    # Integrate state
    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt
    theta = np.arctan2(np.sin(theta), np.cos(theta))  # Normalize angle
    
    state = np.array([x, y, theta])
    trajectory[i, :] = state

# Plot Trajectory
plt.figure(figsize=(8, 8))
# Plot unit circle
circle = plt.Circle((0, 0), R, fill=False, color='r', label='Unit Circle')
plt.gca().add_artist(circle)
# Plot trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Vehicle Trajectory')
plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', label='Start')
plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', label='End')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Trajectory Following Unit Circle')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Plot Errors Over Time
rho_history = np.zeros(num_steps)
alpha_history = np.zeros(num_steps)
beta_history = np.zeros(num_steps)

for i in range(num_steps):
    x, y, theta = trajectory[i, :]
    x_g, y_g, theta_g = get_goal_point(t[i])
    rho_history[i] = np.sqrt((x - x_g)**2 + (y - y_g)**2)
    alpha_history[i] = np.arctan2(y_g - y, x_g - x) - theta
    beta_history[i] = theta_g - theta

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, rho_history, 'b-')
plt.ylabel('Distance (m)')
plt.title('Tracking Errors')
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(t, np.degrees(alpha_history), 'g-')
plt.ylabel('Alpha (degrees)')
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(t, np.degrees(beta_history), 'r-')
plt.ylabel('Beta (degrees)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.tight_layout()
plt.show()