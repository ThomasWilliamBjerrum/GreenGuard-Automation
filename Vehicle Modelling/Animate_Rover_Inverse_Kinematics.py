import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Rover parameters
r = 0.25  # Wheel radius (m)
L = 1.718  # Wheelbase (m)
W = 1.0  # Track width (m)
dt = 0.001  # Time step (s)
t_final = 10.0
t = np.arange(0, t_final + dt, dt)
N = len(t)

# Simulate forward dynamics state (for demonstration, we recreate it as in the original code)
# In a real scenario, this would be saved from the forward dynamics run
delta_forward = 0.2 * np.sin(2 * np.pi * t / 10) + 1e-6
max_delta_rate = 0.5  # rad/s
delta_smoothed = np.zeros(N)
delta_smoothed[0] = delta_forward[0]
for i in range(1, N):
    delta_change = delta_forward[i] - delta_smoothed[i-1]
    delta_change = max(min(delta_change, max_delta_rate * dt), -max_delta_rate * dt)
    delta_smoothed[i] = delta_smoothed[i-1] + delta_change
delta_forward = delta_smoothed

v_known = 3.0
tan_delta = np.tan(delta_forward)
R = np.where(np.abs(tan_delta) > 1e-6, L / tan_delta, 1e6)
theta_dot_known = v_known / R

# State integration (forward dynamics)
state = np.zeros((N, 3))
for i in range(N - 1):
    k1 = np.array([v_known * np.cos(state[i, 2]), v_known * np.sin(state[i, 2]), theta_dot_known[i]])
    k2 = np.array([v_known * np.cos(state[i, 2] + 0.5 * dt * k1[2]), 
                   v_known * np.sin(state[i, 2] + 0.5 * dt * k1[2]), theta_dot_known[i]])
    k3 = np.array([v_known * np.cos(state[i, 2] + 0.5 * dt * k2[2]), 
                   v_known * np.sin(state[i, 2] + 0.5 * dt * k2[2]), theta_dot_known[i]])
    k4 = np.array([v_known * np.cos(state[i, 2] + dt * k3[2]), 
                   v_known * np.sin(state[i, 2] + dt * k3[2]), theta_dot_known[i]])
    state[i+1, :] = state[i, :] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# Inverse Kinematics: Compute derivatives from state
x_dot = np.zeros(N)
y_dot = np.zeros(N)
theta_dot = np.zeros(N)
for i in range(1, N-1):
    x_dot[i] = (state[i+1, 0] - state[i-1, 0]) / (2 * dt)
    y_dot[i] = (state[i+1, 1] - state[i-1, 1]) / (2 * dt)
    theta_dot[i] = (state[i+1, 2] - state[i-1, 2]) / (2 * dt)
# Forward fill edges
x_dot[0], x_dot[-1] = x_dot[1], x_dot[-2]
y_dot[0], y_dot[-1] = y_dot[1], y_dot[-2]
theta_dot[0], theta_dot[-1] = theta_dot[1], theta_dot[-2]

# Reconstruct v and theta_dot
v_reconstructed = np.sqrt(x_dot**2 + y_dot**2)
theta_dot_reconstructed = theta_dot

# Compute delta (steering angle)
delta = np.arctan2(theta_dot_reconstructed * L, v_reconstructed)
delta = np.where(np.abs(v_reconstructed) < 1e-6, 0, delta)  # Avoid division by zero

# Compute omega_r and omega_l
omega_r = (v_reconstructed / r) + (W / (2 * r)) * theta_dot_reconstructed
omega_l = (v_reconstructed / r) - (W / (2 * r)) * theta_dot_reconstructed

# Animation (same as provided)
fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot(2, 3, (1, 2))
traj_line, = ax1.plot([], [], 'b-', label='Trajectory')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Inverse Kinematics Animated Trajectory')
ax1.grid(True)
ax1.axis('equal')
ax1.legend()

vehicle = plt.Rectangle((0, 0), L, W, fill=True, color='lightgreen', alpha=0.5)
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
ax3.plot(t, x_dot, 'b-', label='x_dot')
ax3.plot(t, y_dot, 'r-', label='y_dot')
ax3.plot(t, theta_dot_reconstructed, 'g-', label='theta_dot')
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
