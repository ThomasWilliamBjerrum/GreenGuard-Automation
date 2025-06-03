import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Rover parameters
r = 0.25  # Wheel radius (m)
L = 1.718  # Wheelbase (m)
W = 1.0  # Track width (m)
l = 0.5  # Distance from CG to front axle (m)
v = 3.0  # Forward velocity (m/s)
dt = 0.001  # Time step (s), matched to inverse kinematics
t_final = 10.0
t = np.arange(0, t_final + dt, dt)
N = len(t)

# Steering input (unchanged)
delta = 0.2 * np.sin(2 * np.pi * t / 10) + 1e-6

# Smooth steering angle
max_delta_rate = 0.5  # rad/s
delta_smoothed = np.zeros(N)
delta_smoothed[0] = delta[0]
for i in range(1, N):
    delta_change = delta[i] - delta_smoothed[i-1]
    delta_change = max(min(delta_change, max_delta_rate * dt), -max_delta_rate * dt)
    delta_smoothed[i] = delta_smoothed[i-1] + delta_change
delta = delta_smoothed

# Compute Ackermann steering angles
delta_left = np.zeros(N)
delta_right = np.zeros(N)
theta_dot = np.zeros(N)
omega_r = np.zeros(N)
omega_l = np.zeros(N)
phi_dot_right = np.zeros(N)
phi_dot_left = np.zeros(N)
for i in range(N):
    tan_delta = np.tan(delta[i])
    delta_left[i] = np.arctan(tan_delta / (tan_delta * (W/L) + 1))
    delta_right[i] = np.arctan(tan_delta / (tan_delta * (-W/L) + 1))
    R = L / tan_delta if abs(tan_delta) > 1e-6 else 1e6
    theta_dot[i] = v / R
    omega_r[i] = (v / r) - (W / (2 * r)) * theta_dot[i]
    omega_l[i] = (v / r) + (W / (2 * r)) * theta_dot[i]

# State and residuals
state = np.zeros((N, 3))  # [x, y, theta]
residuals = np.zeros((N, 8))

def derivatives(state, v, theta_dot):
    x_dot = v * np.cos(state[2])
    y_dot = v * np.sin(state[2])
    return np.array([x_dot, y_dot, theta_dot])

# Simulation loop
for i in range(N-1):
    k1 = derivatives(state[i, :], v, theta_dot[i])
    k2 = derivatives(state[i, :] + 0.5 * dt * k1, v, theta_dot[i])
    k3 = derivatives(state[i, :] + 0.5 * dt * k2, v, theta_dot[i])
    k4 = derivatives(state[i, :] + dt * k3, v, theta_dot[i])
    state[i+1, :] = state[i, :] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    x_dot = v * np.cos(state[i+1, 2])
    y_dot = v * np.sin(state[i+1, 2])
    theta = state[i+1, 2]

    phi_dot_right[i] = (1/r) * (x_dot * np.cos(delta_right[i] + theta) - y_dot * np.sin(delta_right[i] + theta) - l * np.cos(delta_right[i]) * theta_dot[i])
    phi_dot_left[i] = (1/r) * (x_dot * np.cos(delta_left[i] + theta) - y_dot * np.sin(delta_left[i] + theta) - l * np.cos(delta_left[i]) * theta_dot[i])

    residuals[i, 0] = -x_dot * np.sin(theta) + y_dot * np.cos(theta) + (W/2) * theta_dot[i] - r * omega_r[i]
    residuals[i, 1] = x_dot * np.cos(theta) - y_dot * np.sin(theta)
    residuals[i, 2] = x_dot * np.sin(theta) + y_dot * np.cos(theta) + (W/2) * theta_dot[i] - r * omega_l[i]
    residuals[i, 3] = x_dot * np.cos(theta) - y_dot * np.sin(theta)
    residuals[i, 4] = x_dot * np.cos(delta_right[i] + theta) - y_dot * np.sin(delta_right[i] + theta) - l * np.cos(delta_right[i]) * theta_dot[i] - r * phi_dot_right[i]
    residuals[i, 5] = -x_dot * np.sin(delta_right[i] - theta) + y_dot * np.cos(delta_right[i] - theta) + l * np.sin(delta_right[i]) * theta_dot[i]
    residuals[i, 6] = x_dot * np.cos(delta_left[i] + theta) - y_dot * np.sin(delta_left[i] + theta) - l * np.cos(delta_left[i]) * theta_dot[i] - r * phi_dot_left[i]
    residuals[i, 7] = -x_dot * np.sin(delta_left[i] - theta) + y_dot * np.cos(delta_left[i] - theta) + l * np.sin(delta_left[i]) * theta_dot[i]

# Last step
i = N-1
x_dot = v * np.cos(state[i, 2])
y_dot = v * np.sin(state[i, 2])
theta = state[i, 2]
phi_dot_right[i] = (1/r) * (x_dot * np.cos(delta_right[i] + theta) - y_dot * np.sin(delta_right[i] + theta) - l * np.cos(delta_right[i]) * theta_dot[i])
phi_dot_left[i] = (1/r) * (x_dot * np.cos(delta_left[i] + theta) - y_dot * np.sin(delta_left[i] + theta) - l * np.cos(delta_left[i]) * theta_dot[i])
residuals[i, :] = [
    -x_dot * np.sin(theta) + y_dot * np.cos(theta) + (W/2) * theta_dot[i] - r * omega_r[i],
    x_dot * np.cos(theta) - y_dot * np.sin(theta),
    x_dot * np.sin(theta) + y_dot * np.cos(theta) + (W/2) * theta_dot[i] - r * omega_l[i],
    x_dot * np.cos(theta) - y_dot * np.sin(theta),
    x_dot * np.cos(delta_right[i] + theta) - y_dot * np.sin(delta_right[i] + theta) - l * np.cos(delta_right[i]) * theta_dot[i] - r * phi_dot_right[i],
    -x_dot * np.sin(delta_right[i] - theta) + y_dot * np.cos(delta_right[i] - theta) + l * np.sin(delta_right[i]) * theta_dot[i],
    x_dot * np.cos(delta_left[i] + theta) - y_dot * np.sin(delta_left[i] + theta) - l * np.cos(delta_left[i]) * theta_dot[i] - r * phi_dot_left[i],
    -x_dot * np.sin(delta_left[i] - theta) + y_dot * np.cos(delta_left[i] - theta) + l * np.sin(delta_left[i]) * theta_dot[i]
]

# Plotting setup
fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot(2, 3, (1, 2))
traj_line, = ax1.plot([], [], 'b-', label='Trajectory')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Animated Ackermann Vehicle Kinematic Simulation')
ax1.grid(True)
ax1.axis('equal')
ax1.legend()

vehicle = plt.Rectangle((0, 0), L, W, fill=True, color='lightblue', alpha=0.5)
ax1.add_patch(vehicle)
wheel_length, wheel_width = 0.2, 0.1
rear_left_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, color='gray')
rear_right_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, color='gray')
front_left_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, color='gray')
front_right_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, color='gray')
for wheel in [rear_left_wheel, rear_right_wheel, front_left_wheel, front_right_wheel]:
    ax1.add_patch(wheel)
center_point, = ax1.plot([], [], 'ro')

ax2 = plt.subplot(2, 3, 3)
ax2.plot(t, state[:, 2], 'r-', label='theta')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Theta (rad)')
ax2.set_title('Orientation')
ax2.grid(True)
ax2.legend()

ax3 = plt.subplot(2, 3, 4)
ax3.plot(t, v * np.cos(state[:, 2]), 'b-', label='x_dot')
ax3.plot(t, v * np.sin(state[:, 2]), 'r-', label='y_dot')
ax3.plot(t, theta_dot, 'g-', label='theta_dot')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocities (m/s or rad/s)')
ax3.set_title('Velocities')
ax3.grid(True)
ax3.legend()

ax4 = plt.subplot(2, 3, 5)
ax4.plot(t, delta, 'g-', label='delta')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Delta (rad)')
ax4.set_title('Steering Angle')
ax4.grid(True)
ax4.legend()

ax6 = plt.subplot(2, 3, 6)
ax6.plot(t, omega_r, 'b-', label='omega_r')
ax6.plot(t, omega_l, 'r-', label='omega_l')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Wheel Angular Velocities (rad/s)')
ax6.set_title('Wheel Angular Velocities')
ax6.grid(True)
ax6.legend()

def update(frame):
    frame_idx = frame * 100
    if frame_idx >= N:
        frame_idx = N - 1
    x, y, theta = state[frame_idx, :]
    steer = delta[frame_idx]
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    vehicle.set_xy((x - L/2 * cos_th + W/2 * sin_th, y - L/2 * sin_th - W/2 * cos_th))
    vehicle.set_angle(np.degrees(theta))
    
    rear_left_x = x - (L/2) * cos_th + (W/2 + wheel_width/2) * sin_th
    rear_left_y = y - (L/2) * sin_th - (W/2 + wheel_width/2) * cos_th
    rear_left_wheel.set_xy((rear_left_x - wheel_length/2 * cos_th + wheel_width/2 * sin_th,
                            rear_left_y - wheel_length/2 * sin_th - wheel_width/2 * cos_th))
    rear_left_wheel.set_angle(np.degrees(theta))
    
    rear_right_x = x - (L/2) * cos_th - (W/2 + wheel_width/2) * sin_th
    rear_right_y = y - (L/2) * sin_th + (W/2 + wheel_width/2) * cos_th
    rear_right_wheel.set_xy((rear_right_x - wheel_length/2 * cos_th - wheel_width/2 * sin_th,
                             rear_right_y - wheel_length/2 * sin_th + wheel_width/2 * cos_th))
    rear_right_wheel.set_angle(np.degrees(theta))
    
    cos_th_steer = np.cos(theta + steer)
    sin_th_steer = np.sin(theta + steer)
    front_left_x = x + (L/2) * cos_th + (W/2 + wheel_width/2) * sin_th
    front_left_y = y + (L/2) * sin_th - (W/2 + wheel_width/2) * cos_th
    front_left_wheel.set_xy((front_left_x - wheel_length/2 * cos_th_steer + wheel_width/2 * sin_th_steer,
                             front_left_y - wheel_length/2 * sin_th_steer - wheel_width/2 * cos_th_steer))
    front_left_wheel.set_angle(np.degrees(theta + steer))
    
    front_right_x = x + (L/2) * cos_th - (W/2 + wheel_width/2) * sin_th
    front_right_y = y + (L/2) * sin_th + (W/2 + wheel_width/2) * cos_th
    front_right_wheel.set_xy((front_right_x - wheel_length/2 * cos_th_steer - wheel_width/2 * sin_th_steer,
                              front_right_y - wheel_length/2 * sin_th_steer + wheel_width/2 * cos_th_steer))
    front_right_wheel.set_angle(np.degrees(theta + steer))
    
    center_point.set_data([x], [y])
    traj_line.set_data(state[:frame_idx+1, 0], state[:frame_idx+1, 1])
    
    x_data, y_data = state[:frame_idx+1, 0], state[:frame_idx+1, 1]
    if len(x_data) > 0:
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        padding = max((x_max - x_min), (y_max - y_min)) * 0.1 + 1.0
        ax1.set_xlim(x_min - padding, x_max + padding)
        ax1.set_ylim(y_min - padding, y_max + padding)
    
    return (vehicle, rear_left_wheel, rear_right_wheel, front_left_wheel, front_right_wheel, center_point, traj_line)

n_frames = N // 100
ani = FuncAnimation(fig, update, frames=range(n_frames), interval=50, blit=True)
plt.show()