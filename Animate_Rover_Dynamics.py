# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:50:35 2025

@author: bentl
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RoverDynamics:
    def __init__(self, r=0.25, L=1.718, W=1.0, m=100.0, I_z=33.0, I_w=0.31, C_f=0.1, C_a=0.01, b=0.1, k_delta=10, dt=0.001):
        """Initialize rover parameters."""
        self.r = r  # Wheel radius (m)
        self.L = L  # Wheelbase (m)
        self.W = W  # Track width (m)
        self.m = m  # Mass (kg)
        self.I_z = I_z  # Moment of inertia about z-axis (kg·m²)
        self.I_w = I_w  # Wheel inertia (kg·m²)
        self.C_f = C_f  # Friction coefficient
        self.C_a = C_a  # Drag coefficient
        self.b = b  # Damping coefficient
        self.k_delta = k_delta  # Steering alignment stiffness
        self.dt = dt  # Time step (s)

    def dynamics(self, t, state, T_r, T_l, delta_cmd):
        """Compute the time derivative of the state vector using the forward dynamics model."""
        x_G, y_G, theta, theta_dot, x_dot_G, y_dot_G, omega_r, omega_l, delta = state

        # Body frame velocity from wheel speeds
        x_dot_b = (self.r / 2) * (omega_r + omega_l) * np.cos(delta)
        y_dot_b = (self.r / 2) * (omega_r + omega_l) * np.sin(delta)

        # Global frame velocities
        x_dot_G_new = x_dot_b * np.cos(theta) - y_dot_b * np.sin(theta)
        y_dot_G_new = x_dot_b * np.sin(theta) + y_dot_b * np.cos(theta)

        # Compute body frame accelerations
        x_ddot_b = (1/self.m) * ((T_r + T_l)/self.r - self.C_f * x_dot_b - self.C_a * x_dot_b**2)
        y_ddot_b = -x_dot_b * theta_dot + (self.r / (2 * self.L)) * (omega_r + omega_l) * np.tan(delta)

        # Global frame accelerations
        x_ddot_G = x_ddot_b * np.cos(theta) - y_ddot_b * np.sin(theta) - theta_dot * (x_dot_b * np.sin(theta) + y_dot_b * np.cos(theta))
        y_ddot_G = x_ddot_b * np.sin(theta) + y_ddot_b * np.cos(theta) + theta_dot * (x_dot_b * np.cos(theta) - y_dot_b * np.sin(theta))

        # Yaw acceleration (theta_ddot)
        theta_ddot = (self.W / (2 * self.r * self.I_z)) * (T_r - T_l) + (self.k_delta / self.I_z) * ((self.r / self.L) * (omega_r + omega_l) * np.tan(delta) - theta_dot)

        # Wheel dynamics
        omega_r_dot = (T_r - self.b * omega_r) / self.I_w
        omega_l_dot = (T_l - self.b * omega_l) / self.I_w

        # Steering dynamics: Use a proportional controller to track delta_cmd
        k_steer = 10.0  # Steering gain
        delta_dot = k_steer * (delta_cmd - delta)

        # State derivative
        dstate_dt = np.array([
            x_dot_G_new,     # dx_G/dt
            y_dot_G_new,     # dy_G/dt
            theta_dot,       # dtheta/dt
            theta_ddot,      # d(theta_dot)/dt
            x_ddot_G,        # d(x_dot_G)/dt
            y_ddot_G,        # d(y_dot_G)/dt
            omega_r_dot,     # domega_r/dt
            omega_l_dot,     # domega_l/dt
            delta_dot        # ddelta/dt
        ])
        return dstate_dt, x_dot_b, y_dot_b, x_ddot_G, y_ddot_G, theta_ddot

    def rk4_step(self, t, state, T_r, T_l, delta_cmd):
        """Perform one RK4 integration step."""
        k1, x_dot_b1, y_dot_b1, x_ddot_G1, y_ddot_G1, theta_ddot1 = self.dynamics(t, state, T_r, T_l, delta_cmd)
        k2, x_dot_b2, y_dot_b2, x_ddot_G2, y_ddot_G2, theta_ddot2 = self.dynamics(t + self.dt/2, state + self.dt/2 * k1, T_r, T_l, delta_cmd)
        k3, x_dot_b3, y_dot_b3, x_ddot_G3, y_ddot_G3, theta_ddot3 = self.dynamics(t + self.dt/2, state + self.dt/2 * k2, T_r, T_l, delta_cmd)
        k4, x_dot_b4, y_dot_b4, x_ddot_G4, y_ddot_G4, theta_ddot4 = self.dynamics(t + self.dt, state + self.dt * k3, T_r, T_l, delta_cmd)
        next_state = state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x_dot_b = (x_dot_b1 + 2*x_dot_b2 + 2*x_dot_b3 + x_dot_b4) / 6
        y_dot_b = (y_dot_b1 + 2*y_dot_b2 + 2*y_dot_b3 + y_dot_b4) / 6
        x_ddot_G = (x_ddot_G1 + 2*x_ddot_G2 + 2*x_ddot_G3 + x_ddot_G4) / 6
        y_ddot_G = (y_ddot_G1 + 2*y_ddot_G2 + 2*y_ddot_G3 + y_ddot_G4) / 6
        theta_ddot = (theta_ddot1 + 2*theta_ddot2 + 2*theta_ddot3 + theta_ddot4) / 6
        return next_state, x_dot_b, y_dot_b, x_ddot_G, y_ddot_G, theta_ddot

def generate_control_inputs(t, rover, target_velocity=3.0):
    """
    Generate control inputs for the rover to achieve a target velocity.
    """
    n_steps = len(t)
    T_r = np.zeros(n_steps)
    T_l = np.zeros(n_steps)
    delta = 0.2* np.sin(2 * np.pi * t / 10) + 1e-6  # Sinusoidal steering input

    # Desired wheel angular velocity for target_velocity
    omega_desired = target_velocity / rover.r  # 10 rad/s for v = 3 m/s, r = 0.3 m

    # Initial torque to overcome damping and accelerate wheels
    alpha = 100.0  # Desired angular acceleration
    for i in range(n_steps):
        theta_dot = (target_velocity * np.tan(delta[i])) / rover.L
        v_r = target_velocity + (theta_dot * rover.W / 2)
        v_l = target_velocity - (theta_dot * rover.W / 2)
        omega_r_desired = v_r / rover.r
        omega_l_desired = v_l / rover.r

        # Increase torque to achieve desired omega
        T_r[i] = rover.I_w * alpha + rover.b * omega_r_desired
        T_l[i] = rover.I_w * alpha + rover.b * omega_l_desired

    # Clip torques
    T_r = np.clip(T_r, -10.0, 10.0)
    T_l = np.clip(T_l, -10.0, 10.0)

    return T_r, T_l, delta

def simulate_rover(rover, t, T_r_init, T_l_init, delta_cmd):
    """
    Simulate the rover dynamics with feedback control.
    """
    n_steps = len(t)
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_history = [initial_state]
    x_dot_G_history = [0.0]
    y_dot_G_history = [0.0]
    T_r_history = []
    T_l_history = []

    target_velocity = 3.0
    k_omega = 2.0

    for i in range(n_steps - 1):
        current_state = state_history[-1]
        omega_r, omega_l, delta, theta = current_state[6], current_state[7], current_state[8], current_state[2]

        # Desired wheel velocities
        theta_dot = (target_velocity * np.tan(delta)) / rover.L
        v_r = target_velocity + (theta_dot * rover.W / 2)
        v_l = target_velocity - (theta_dot * rover.W / 2)
        omega_r_desired = v_r / rover.r
        omega_l_desired = v_l / rover.r

        # Feedback control
        error_r = omega_r_desired - omega_r
        error_l = omega_l_desired - omega_l
        T_r = T_r_init[i] + k_omega * error_r
        T_l = T_l_init[i] + k_omega * error_l

        # Clip torques
        T_r = np.clip(T_r, -10.0, 10.0)
        T_l = np.clip(T_l, -10.0, 10.0)

        T_r_history.append(T_r)
        T_l_history.append(T_l)

        # Step dynamics
        next_state, x_dot_b, y_dot_b, x_ddot_G, y_ddot_G, theta_ddot = rover.rk4_step(t[i], current_state, T_r, T_l, delta_cmd[i])

        # Compute velocities
        x_dot_G = x_dot_b * np.cos(theta) - y_dot_b * np.sin(theta)
        y_dot_G = x_dot_b * np.sin(theta) + y_dot_b * np.cos(theta)

        x_dot_G_history.append(x_dot_G)
        y_dot_G_history.append(y_dot_G)
        state_history.append(next_state)

        if i % 1000 == 0:
            print(f"Step {i}, t = {t[i]:.1f}, omega_r={omega_r:.2f}, omega_l={omega_l:.2f}, v={x_dot_G:.2f} m/s")

    return (np.array(state_history), np.array(x_dot_G_history), np.array(y_dot_G_history),
            np.array(T_r_history), np.array(T_l_history))

# Simulation setup
rover = RoverDynamics(dt=0.001)
t_final = 10.0
t = np.arange(0, t_final + rover.dt, rover.dt)
n_steps = len(t)

# Generate control inputs
T_r_init, T_l_init, delta_cmd = generate_control_inputs(t, rover, target_velocity=3.0)

# Run simulation
state_history, x_dot_G_history, y_dot_G_history, T_r_history, T_l_history = simulate_rover(
    rover, t, T_r_init, T_l_init, delta_cmd)

# Remove the last two elements from the arrays
# Adjust the number of steps to exclude the last two elements
n_steps_adjusted = n_steps - 2

# Adjust time array
t = t[:n_steps_adjusted]

# Adjust state history and related arrays
state_history = state_history[:n_steps_adjusted]
x_dot_G_history = x_dot_G_history[:n_steps_adjusted]
y_dot_G_history = y_dot_G_history[:n_steps_adjusted]
T_r_history = T_r_history[:n_steps_adjusted-1]  # T_r_history has one less element than state_history
T_l_history = T_l_history[:n_steps_adjusted-1]  # T_l_history has one less element than state_history

# Extract states
x_G = state_history[:, 0]
y_G = state_history[:, 1]
theta = state_history[:, 2]
theta_dot = state_history[:, 3]
omega_r = state_history[:, 6]  # Already adjusted by state_history slicing
omega_l = state_history[:, 7]  # Already adjusted by state_history slicing
delta = state_history[:, 8]

# Set up the figure and subplots for animation
fig = plt.figure(figsize=(15, 15))

# Row 1: Animation and Orientation
# Top Left: Animation of the vehicle
ax1 = plt.subplot(3, 3, (1, 2))
# Initialize an empty trajectory line that will be updated dynamically
traj_line, = ax1.plot([], [], 'b-', label='Actual Trajectory')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Rover Trajectory')
ax1.grid(True)
ax1.axis('equal')
ax1.legend()

# Initialize the vehicle (rectangle with wheels)
vehicle_length = rover.L
vehicle_width = rover.W
vehicle = plt.Rectangle((0, 0), vehicle_length, vehicle_width, fill=True, color='lightblue', alpha=0.5)
ax1.add_patch(vehicle)

# Initialize wheels
wheel_length = 0.2
wheel_width = 0.1
rear_left_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, fill=True, color='gray')
rear_right_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, fill=True, color='gray')
front_left_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, fill=True, color='gray')
front_right_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, fill=True, color='gray')
ax1.add_patch(rear_left_wheel)
ax1.add_patch(rear_right_wheel)
ax1.add_patch(front_left_wheel)
ax1.add_patch(front_right_wheel)

# Center point of the vehicle
center_point, = ax1.plot([], [], 'ro')

# Top Right: Orientation (theta) - Static
ax2 = plt.subplot(3, 3, 3)
ax2.plot(t, theta, 'r-', label='theta')  # Plot the full data
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Theta (rad)')
ax2.set_title('Orientation')
ax2.grid(True)
ax2.legend()

# Row 2: Velocities, Steering Angle, Torques, Wheel Angular Velocities
# Bottom Left: Velocities (x_dot_G, y_dot_G, theta_dot) - Static
ax3 = plt.subplot(3, 3, 4)
ax3.plot(t, x_dot_G_history, 'b-', label='x_dot_G')
ax3.plot(t, y_dot_G_history, 'r-', label='y_dot_G')
ax3.plot(t, theta_dot, 'g-', label='theta_dot')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocities (m/s or rad/s)')
ax3.set_title('Velocities (Global Frame)')
ax3.grid(True)
ax3.legend()

# Bottom Middle Left: Steering Angle (delta) - Static
ax4 = plt.subplot(3, 3, 5)
ax4.plot(t, delta, 'g-', label='delta')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Delta (rad)')
ax4.set_title('Steering Angle')
ax4.grid(True)
ax4.legend()

# Bottom Middle Right: Torques (T_r, T_l) - Static
ax5 = plt.subplot(3, 3, 6)
ax5.plot(t[:-1], T_r_history, 'm-', label='T_r')
ax5.plot(t[:-1], T_l_history, 'c-', label='T_l')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Torque (N·m)')
ax5.set_title('Torques')
ax5.grid(True)
ax5.legend()

# Bottom Right: Wheel Angular Velocities (omega_r, omega_l) - Static
ax6 = plt.subplot(3, 3, 9)
ax6.plot(t, omega_r, 'b-', label='omega_r')
ax6.plot(t, omega_l, 'r-', label='omega_l')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Wheel Angular Velocities (rad/s)')
ax6.set_title('Wheel Angular Velocities')
ax6.grid(True)
ax6.legend()

# Animation update function
def update(frame):
    frame_idx = frame * 100
    if frame_idx >= n_steps_adjusted:
        frame_idx = n_steps_adjusted - 1
    
    # Get current states
    x = x_G[frame_idx]
    y = y_G[frame_idx]
    th = theta[frame_idx]
    steer = delta[frame_idx]
    
    # Calculate transformation matrix for vehicle body
    cos_th = np.cos(th)
    sin_th = np.sin(th)
    
    # Update vehicle body position and orientation
    vehicle.set_xy((x - vehicle_length/2 * cos_th + vehicle_width/2 * sin_th,
                    y - vehicle_length/2 * sin_th - vehicle_width/2 * cos_th))
    vehicle.set_angle(np.degrees(th))
    
    # Update wheel positions with proper transformations
    # Rear wheels (no steering, just body orientation)
    rear_left_x = x - (vehicle_length/2) * cos_th + (vehicle_width/2 + wheel_width/2) * sin_th
    rear_left_y = y - (vehicle_length/2) * sin_th - (vehicle_width/2 + wheel_width/2) * cos_th
    rear_left_wheel.set_xy((rear_left_x - wheel_length/2 * cos_th + wheel_width/2 * sin_th,
                           rear_left_y - wheel_length/2 * sin_th - wheel_width/2 * cos_th))
    rear_left_wheel.set_angle(np.degrees(th))
    
    rear_right_x = x - (vehicle_length/2) * cos_th - (vehicle_width/2 + wheel_width/2) * sin_th
    rear_right_y = y - (vehicle_length/2) * sin_th + (vehicle_width/2 + wheel_width/2) * cos_th
    rear_right_wheel.set_xy((rear_right_x - wheel_length/2 * cos_th - wheel_width/2 * sin_th,
                            rear_right_y - wheel_length/2 * sin_th + wheel_width/2 * cos_th))
    rear_right_wheel.set_angle(np.degrees(th))
    
    # Front wheels (include steering angle)
    cos_th_steer = np.cos(th + steer)
    sin_th_steer = np.sin(th + steer)
    
    front_left_x = x + (vehicle_length/2) * cos_th + (vehicle_width/2 + wheel_width/2) * sin_th
    front_left_y = y + (vehicle_length/2) * sin_th - (vehicle_width/2 + wheel_width/2) * cos_th
    front_left_wheel.set_xy((front_left_x - wheel_length/2 * cos_th_steer + wheel_width/2 * sin_th_steer,
                            front_left_y - wheel_length/2 * sin_th_steer - wheel_width/2 * cos_th_steer))
    front_left_wheel.set_angle(np.degrees(th + steer))
    
    front_right_x = x + (vehicle_length/2) * cos_th - (vehicle_width/2 + wheel_width/2) * sin_th
    front_right_y = y + (vehicle_length/2) * sin_th + (vehicle_width/2 + wheel_width/2) * cos_th
    front_right_wheel.set_xy((front_right_x - wheel_length/2 * cos_th_steer - wheel_width/2 * sin_th_steer,
                             front_right_y - wheel_length/2 * sin_th_steer + wheel_width/2 * cos_th_steer))
    front_right_wheel.set_angle(np.degrees(th + steer))
    
    # Update center point
    center_point.set_data([x], [y])
    
    # Update the trajectory line to show only the path up to the current position
    traj_line.set_data(x_G[:frame_idx+1], y_G[:frame_idx+1])
    
    # Autoscaling for the trajectory plot
    x_data = x_G[:frame_idx+1]
    y_data = y_G[:frame_idx+1]
    if len(x_data) > 0:
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        x_padding = (x_max - x_min) * 0.1 + 1.0
        y_padding = (y_max - y_min) * 0.1 + 1.0
        ax1.set_xlim(x_min - x_padding, x_max + x_padding)
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)

    return (vehicle, rear_left_wheel, rear_right_wheel, front_left_wheel, front_right_wheel,
            center_point, traj_line)

# Create the animation
n_frames = n_steps_adjusted // 100
ani = FuncAnimation(fig, update, frames=range(n_frames), interval=50, blit=True)

# Create a new static figure for all vehicle state variables (similar to inverse dynamics)
fig1, axs = plt.subplots(5, 2, figsize=(15, 15))

# Row 0: Trajectory and x_G
axs[0, 0].plot(x_G, y_G, 'b-', label='Actual Trajectory')
axs[0, 0].set_xlabel('X Position (m)')
axs[0, 0].set_ylabel('Y Position (m)')
axs[0, 0].set_title('Trajectory')
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].plot(t, x_G, 'b-', label='x_G')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('X Position (m)')
axs[0, 1].set_title('X Position')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Row 1: y_G and theta
axs[1, 0].plot(t, y_G, 'r-', label='y_G')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Y Position (m)')
axs[1, 0].set_title('Y Position')
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].plot(t, theta, 'g-', label='Theta')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Theta (rad)')
axs[1, 1].set_title('Orientation')
axs[1, 1].grid(True)
axs[1, 1].legend()

# Row 2: Velocities and Delta
axs[2, 0].plot(t, x_dot_G_history, 'b-', label='x_dot_G')
axs[2, 0].plot(t, y_dot_G_history, 'r-', label='y_dot_G', linestyle='dashed')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('Velocities (m/s)')
axs[2, 0].set_title('Global Velocities')
axs[2, 0].grid(True)
axs[2, 0].legend()

axs[2, 1].plot(t, delta, 'g-', label='delta')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Delta (rad)')
axs[2, 1].set_title('Steering Angle')
axs[2, 1].grid(True)
axs[2, 1].legend()

# Row 3: Wheel Angular Velocities and Torques
axs[3, 0].plot(t, omega_r, 'b-', label='omega_r')
axs[3, 0].plot(t, omega_l, 'r-', label='omega_l', linestyle='dashed')
axs[3, 0].set_xlabel('Time (s)')
axs[3, 0].set_ylabel('Wheel Angular Velocities (rad/s)')
axs[3, 0].set_title('Wheel Angular Velocities')
axs[3, 0].grid(True)
axs[3, 0].legend()

axs[3, 1].plot(t[:-1], T_r_history, 'm-', label='T_r')
axs[3, 1].plot(t[:-1], T_l_history, 'c-', label='T_l', linestyle='dashed')
axs[3, 1].set_xlabel('Time (s)')
axs[3, 1].set_ylabel('Torque (N·m)')
axs[3, 1].set_title('Torques')
axs[3, 1].grid(True)
axs[3, 1].legend()

# Row 4: Theta_dot and Empty
axs[4, 0].plot(t, theta_dot, 'g-', label='Theta_dot')
axs[4, 0].set_xlabel('Time (s)')
axs[4, 0].set_ylabel('Theta_dot (rad/s)')
axs[4, 0].set_title('Yaw Rate')
axs[4, 0].grid(True)
axs[4, 0].legend()

axs[4, 1].axis('off')  # Turn off the last subplot

plt.tight_layout()

# Show the animation and static plots
plt.show()