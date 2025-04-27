import matplotlib
matplotlib.use('TkAgg')  # Force an interactive backend for animations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# If you're using Jupyter Notebook, uncomment the following line and comment out matplotlib.use('TkAgg')
# %matplotlib notebook

class RoverDynamics:
    def __init__(self, r=0.5, L=1.718, W=1.0, m=10.0, I_z=2.0, I_w=0.01, 
                 C_f=0.1, C_a=0.01, b=0.1, k_delta=2.0, dt=0.001):
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

def generate_desired_trajectory(rover, t, target_velocity=3.0):
    """
    Generate desired trajectory using forward dynamics with specified theta_dot and target velocity.
    
    Parameters:
    - rover: RoverDynamics instance
    - t: Array of time steps
    - target_velocity: Desired velocity in m/s (default: 3.0)
    
    Returns:
    - Desired states and derivatives
    """
    n_steps = len(t)
    # Define a sinusoidal desired yaw rate (theta_dot) to achieve a similar trajectory
    theta_dot_des = 0.1 * np.sin(2 * np.pi * t / 10)  # Amplitude 0.1 rad/s, period 10s

    # Initial state: [x_G, y_G, theta, theta_dot, x_dot_G, y_dot_G, omega_r, omega_l, delta]
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_history = [initial_state]
    x_dot_b_history = [0.0]
    y_dot_b_history = [0.0]
    x_ddot_G_history = [0.0]
    y_ddot_G_history = [0.0]
    theta_ddot_history = [0.0]
    delta_cmd_history = [0.0]

    # Simple control to achieve target velocity
    k_omega = 0.5  # Feedback control gain
    for i in range(n_steps - 1):
        current_state = state_history[-1]
        omega_r, omega_l, delta, theta = current_state[6], current_state[7], current_state[8], current_state[2]
        x_dot_b = (rover.r / 2) * (omega_r + omega_l) * np.cos(delta)

        # Desired forward velocity
        V = target_velocity

        # Desired yaw rate at this time step
        theta_dot_cmd = theta_dot_des[i]

        # Compute desired wheel speeds using x_b and theta_dot
        # x_b = (r/2) * (omega_r + omega_l)
        # theta_dot = (r/W) * (omega_r - omega_l)
        # Solve for omega_r, omega_l:
        # omega_r + omega_l = (2/r) * x_b
        # omega_r - omega_l = (W/r) * theta_dot
        x_b_des = V  # Approximate x_b_des as V (adjust based on delta in dynamics)
        omega_r_des = (2 * x_b_des / rover.r + (rover.W / rover.r) * theta_dot_cmd) / 2
        omega_l_des = (2 * x_b_des / rover.r - (rover.W / rover.r) * theta_dot_cmd) / 2

        # Compute delta using the thesis equation: delta = arctan(2L/W * (omega_r - omega_l) / (omega_r + omega_l))
        if omega_r_des + omega_l_des != 0:  # Avoid division by zero
            delta_cmd = np.arctan2(2 * rover.L * (omega_r_des - omega_l_des), rover.W * (omega_r_des + omega_l_des))
        else:
            delta_cmd = 0.0
        delta_cmd_history.append(delta_cmd)

        # Feedback control for torques
        error_r = omega_r_des - omega_r
        error_l = omega_l_des - omega_l
        T_r = rover.b * omega_r_des + k_omega * error_r
        T_l = rover.b * omega_l_des + k_omega * error_l
        T_r = np.clip(T_r, -5.0, 5.0)
        T_l = np.clip(T_l, -5.0, 5.0)

        # Step the dynamics
        next_state, x_dot_b, y_dot_b, x_ddot_G, y_ddot_G, theta_ddot = rover.rk4_step(t[i], current_state, T_r, T_l, delta_cmd)
        state_history.append(next_state)
        x_dot_b_history.append(x_dot_b)
        y_dot_b_history.append(y_dot_b)
        x_ddot_G_history.append(x_ddot_G)
        y_ddot_G_history.append(y_ddot_G)
        theta_ddot_history.append(theta_ddot)

    # Convert to arrays
    state_history = np.array(state_history)
    x_G = state_history[:, 0]
    y_G = state_history[:, 1]
    theta = state_history[:, 2]
    theta_dot = state_history[:, 3]
    x_dot_G = state_history[:, 4]
    y_dot_G = state_history[:, 5]
    omega_r = state_history[:, 6]
    omega_l = state_history[:, 7]
    delta = state_history[:, 8]
    x_dot_b = np.array(x_dot_b_history)
    y_dot_b = np.array(y_dot_b_history)
    x_ddot_G = np.array(x_ddot_G_history)
    y_ddot_G = np.array(y_ddot_G_history)
    theta_ddot = np.array(theta_ddot_history)
    delta_cmd = np.array(delta_cmd_history)

    return (x_G, y_G, theta, theta_dot, x_dot_G, y_dot_G, 
            omega_r, omega_l, delta, x_dot_b, y_dot_b, 
            x_ddot_G, y_ddot_G, theta_ddot, delta_cmd)

def compute_inverse_dynamics(rover, t, x_G_des, y_G_des, theta_des, theta_dot_des, 
                            x_dot_G_des, y_dot_G_des, delta_des, x_dot_b_des, 
                            y_dot_b_des, x_ddot_G_des, y_ddot_G_des, theta_ddot_des):
    """
    Compute control inputs using the provided inverse dynamics equations.
    
    Parameters:
    - rover: RoverDynamics instance
    - t: Array of time steps
    - Desired states and derivatives
    
    Returns:
    - delta_cmd: Steering angle command
    - T_r_id: Right wheel torque
    - T_l_id: Left wheel torque
    """
    n_steps = len(t)
    delta_cmd = delta_des.copy()
    T_r_id = np.zeros(n_steps)
    T_l_id = np.zeros(n_steps)

    for i in range(n_steps):
        # Compute torques using the inverse dynamics equations
        term1 = rover.r * rover.m * (x_ddot_G_des[i] * np.cos(theta_des[i]) + y_ddot_G_des[i] * np.sin(theta_des[i]))
        term2 = rover.r * rover.C_f * x_dot_b_des[i]
        term3 = rover.r * rover.C_a * x_dot_b_des[i]**2
        term4 = (2 * rover.I_z * rover.r / rover.W) * (theta_ddot_des[i] + (rover.k_delta / rover.I_z) * 
                                                       (theta_dot_des[i] - (x_dot_b_des[i] * np.tan(delta_des[i]) / rover.L)))

        T_r_id[i] = 0.5 * (term1 + term2 + term3 + term4)
        T_l_id[i] = 0.5 * (term1 + term2 + term3 - term4)

        # Clip torques to reasonable values
        T_r_id[i] = np.clip(T_r_id[i], -5.0, 5.0)
        T_l_id[i] = np.clip(T_l_id[i], -5.0, 5.0)

    return delta_cmd, T_r_id, T_l_id

def simulate_rover(rover, t, delta_cmd, T_r_id, T_l_id):
    """
    Simulate the rover dynamics over time with given control inputs.
    
    Parameters:
    - rover: RoverDynamics instance
    - t: Array of time steps
    - delta_cmd: Steering angle command
    - T_r_id: Right wheel torque
    - T_l_id: Left wheel torque
    
    Returns:
    - state_history: Array of state vectors over time
    - x_dot_G_history: Array of x velocities
    - y_dot_G_history: Array of y velocities
    - T_r_history: Array of applied right torques
    - T_l_history: Array of applied left torques
    """
    n_steps = len(t)
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_history = [initial_state]
    x_dot_G_history = [0.0]
    y_dot_G_history = [0.0]
    T_r_history = []
    T_l_history = []

    for i in range(n_steps - 1):
        current_state = state_history[-1]
        omega_r, omega_l, delta, theta = current_state[6], current_state[7], current_state[8], current_state[2]

        T_r = T_r_id[i]
        T_l = T_l_id[i]
        T_r_history.append(T_r)
        T_l_history.append(T_l)

        next_state, x_dot_b, y_dot_b, x_ddot_G, y_ddot_G, theta_ddot = rover.rk4_step(t[i], current_state, T_r, T_l, delta_cmd[i])

        x_dot_G = x_dot_b * np.cos(theta) - y_dot_b * np.sin(theta)
        y_dot_G = x_dot_b * np.sin(theta) + y_dot_b * np.cos(theta)

        x_dot_G_history.append(x_dot_G)
        y_dot_G_history.append(y_dot_G)
        state_history.append(next_state)

        if i % 1000 == 0:
            x_G, y_G, theta, _, _, _, omega_r, omega_l, delta = current_state
            print(f"Step {i}, t = {t[i]:.1f}, x_G={x_G:.2f}, y_G={y_G:.2f}, theta={np.rad2deg(theta):.1f}°, delta={np.rad2deg(delta):.1f}°")

    return (np.array(state_history), np.array(x_dot_G_history), np.array(y_dot_G_history),
            np.array(T_r_history), np.array(T_l_history))

# Simulation setup
rover = RoverDynamics(dt=0.001)
t_final = 10.0
t = np.arange(0, t_final + rover.dt, rover.dt)
n_steps = len(t)

# Generate desired trajectory using forward dynamics
(x_G_des, y_G_des, theta_des, theta_dot_des, x_dot_G_des, y_dot_G_des, 
 omega_r_des, omega_l_des, delta_des, x_dot_b_des, y_dot_b_des, 
 x_ddot_G_des, y_ddot_G_des, theta_ddot_des, delta_cmd) = generate_desired_trajectory(rover, t, target_velocity=3.0)

# Compute control inputs using inverse dynamics
delta_cmd, T_r_id, T_l_id = compute_inverse_dynamics(
    rover, t, x_G_des, y_G_des, theta_des, theta_dot_des, 
    x_dot_G_des, y_dot_G_des, delta_des, x_dot_b_des, 
    y_dot_b_des, x_ddot_G_des, y_ddot_G_des, theta_ddot_des)

# Run simulation with inverse dynamics torques
state_history, x_dot_G_history, y_dot_G_history, T_r_history, T_l_history = simulate_rover(
    rover, t, delta_cmd, T_r_id, T_l_id)

# Extract states
x_G = state_history[:, 0]
y_G = state_history[:, 1]
theta = state_history[:, 2]
theta_dot = state_history[:, 3]
omega_r = state_history[:, 6]
omega_l = state_history[:, 7]
delta = state_history[:, 8]

# Set up the figure for animation only
fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
traj_line, = ax1.plot([], [], 'b-', label='Trajectory', linewidth=2)
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Rover Trajectory (Animated)')
ax1.grid(True)
ax1.axis('equal')
ax1.legend()

# Initialize the vehicle (rectangle with wheels)
vehicle_length = rover.L
vehicle_width = rover.W
vehicle = plt.Rectangle((0, 0), vehicle_length, vehicle_width, fill=True, color='skyblue', alpha=0.7, label='Rover')
ax1.add_patch(vehicle)

# Initialize wheels
wheel_length = 0.2
wheel_width = 0.1
rear_left_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, fill=True, color='darkgray', alpha=0.9)
rear_right_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, fill=True, color='darkgray', alpha=0.9)
front_left_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, fill=True, color='darkgray', alpha=0.9)
front_right_wheel = plt.Rectangle((0, 0), wheel_length, wheel_width, fill=True, color='darkgray', alpha=0.9)
ax1.add_patch(rear_left_wheel)
ax1.add_patch(rear_right_wheel)
ax1.add_patch(front_left_wheel)
ax1.add_patch(front_right_wheel)

# Center point of the vehicle
center_point, = ax1.plot([], [], 'ro', label='Center of Gravity')
ax1.legend()

def update(frame):
    """Update function for the animation, called for each frame."""
    frame_idx = frame * 100
    if frame_idx >= n_steps:
        frame_idx = n_steps - 1
    
    # Get current states
    x = x_G[frame_idx]
    y = y_G[frame_idx]
    th = theta[frame_idx]
    steer = delta[frame_idx]
    
    # Update the subplot title with the current time
    current_time = t[frame_idx]
    ax1.set_title(f'Rover Trajectory (Animated) - t = {current_time:.1f}s')
    
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
n_frames = n_steps // 100
ani = FuncAnimation(fig, update, frames=range(n_frames), interval=50, blit=True)

# Display the animation figure and give it time to start
plt.show(block=False)
plt.pause(0.1)  # Small pause to allow the animation to initialize

# Now create the static plots in a separate figure
fig1, axs = plt.subplots(5, 2, figsize=(15, 15))

# Trajectory (x_G vs y_G)
axs[0, 0].plot(x_G, y_G, 'b-', label='Actual Trajectory')
axs[0, 0].set_xlabel('X Position (m)')
axs[0, 0].set_ylabel('Y Position (m)')
axs[0, 0].set_title('Rover Trajectory')
axs[0, 0].grid(True)
axs[0, 0].axis('equal')
axs[0, 0].legend()

# Position (x_G, y_G)
axs[0, 1].plot(t, x_G, label='x_G')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('X Position (m)')
axs[0, 1].set_title('Global X Position')
axs[0, 1].grid()
axs[0, 1].legend()

axs[1, 0].plot(t, y_G, label='y_G')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Y Position (m)')
axs[1, 0].set_title('Global Y Position')
axs[1, 0].grid()
axs[1, 0].legend()

# Orientation (theta)
axs[1, 1].plot(t, theta, label='Theta')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Theta (rad)')
axs[1, 1].set_title('Orientation (Theta)')
axs[1, 1].grid()
axs[1, 1].legend()

# Velocity components
axs[2, 0].plot(t, x_dot_G_history, label='x_dot_G')
axs[2, 0].plot(t, y_dot_G_history, label='y_dot_G', linestyle='dashed')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('Velocity (m/s)')
axs[2, 0].set_title('Global Velocities')
axs[2, 0].grid()
axs[2, 0].legend()

# Steering angle
axs[2, 1].plot(t, delta, label='Steering Angle (delta)')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Delta (rad)')
axs[2, 1].set_title('Steering Angle Over Time')
axs[2, 1].grid()
axs[2, 1].legend()

# Wheel angular velocities
axs[3, 0].plot(t, omega_r, label='omega_r')
axs[3, 0].plot(t, omega_l, label='omega_l', linestyle='dashed')
axs[3, 0].set_xlabel('Time (s)')
axs[3, 0].set_ylabel('Angular Velocity (rad/s)')
axs[3, 0].set_title('Wheel Angular Velocities')
axs[3, 0].grid()
axs[3, 0].legend()

# Torques applied
axs[3, 1].plot(t[:-1], T_r_history, label='T_r')
axs[3, 1].plot(t[:-1], T_l_history, linestyle='dashed', label='T_l')
axs[3, 1].set_xlabel('Time (s)')
axs[3, 1].set_ylabel('Torque (N·m)')
axs[3, 1].set_title('Applied Torques')
axs[3, 1].grid()
axs[3, 1].legend()

# Theta_dot (angular velocity)
axs[4, 0].plot(t, theta_dot, label='Theta_dot')
axs[4, 0].set_xlabel('Time (s)')
axs[4, 0].set_ylabel('Theta_dot (rad/s)')
axs[4, 0].set_title('Angular Velocity Over Time')
axs[4, 0].grid()
axs[4, 0].legend()

# Leave the last subplot empty
axs[4, 1].axis('off')

plt.tight_layout()
plt.show()