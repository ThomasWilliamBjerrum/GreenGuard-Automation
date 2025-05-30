import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RoverDynamics:
    def __init__(self, r=0.25, L=1.718, W=1.0, m=100.0, I_z=33.0, I_w=0.31, C_f=0.1, C_a=0.01, b=0.1, k_delta=10, dt=0.001):
        """Initialize rover parameters."""
        self.r = r
        self.L = L
        self.W = W
        self.m = m
        self.I_z = I_z
        self.I_w = I_w
        self.C_f = C_f
        self.C_a = C_a
        self.b = b
        self.k_delta = k_delta
        self.dt = dt

    def dynamics(self, t, state, T_r, T_l, delta):
        """Compute the time derivative of the state vector."""
        x_G, y_G, theta, theta_dot, x_dot_G, y_dot_G, omega_r, omega_l = state

        # Body frame velocity from wheel speeds
        x_dot_b = (self.r / 2) * (omega_r + omega_l) * np.cos(delta)
        y_dot_b = (self.r / 2) * (omega_r + omega_l) * np.sin(delta)

        # Global frame velocities (for state update)
        x_dot_G_new = x_dot_b * np.cos(theta) - y_dot_b * np.sin(theta)
        y_dot_G_new = x_dot_b * np.sin(theta) + y_dot_b * np.cos(theta)

        # Compute body frame accelerations
        x_ddot_b = (1/self.m) * ((T_r + T_l)/self.r - self.C_f * x_dot_b - self.C_a * x_dot_b**2)
        y_ddot_b = -x_dot_b * theta_dot + (self.r / (2 * self.L)) * (omega_r + omega_l) * np.tan(delta)

        # Global frame accelerations
        x_ddot_G = x_ddot_b * np.cos(theta) - y_ddot_b * np.sin(theta) - theta_dot * (x_dot_b * np.sin(theta) + y_dot_b * np.cos(theta))
        y_ddot_G = x_ddot_b * np.sin(theta) + y_ddot_b * np.cos(theta) + theta_dot * (x_dot_b * np.cos(theta) - y_dot_b * np.sin(theta))

        # Yaw acceleration
        theta_ddot = (self.W / (2 * self.r * self.I_z)) * (T_r - T_l) + (self.k_delta / self.I_z) * ((self.r / self.L) * (omega_r + omega_l) * np.tan(delta) - theta_dot)

        # Wheel dynamics
        omega_r_dot = (T_r - self.b * omega_r) / self.I_w
        omega_l_dot = (T_l - self.b * omega_l) / self.I_w

        # State derivative
        dstate_dt = np.array([
            x_dot_G_new,
            y_dot_G_new,
            theta_dot,
            theta_ddot,
            x_ddot_G,
            y_ddot_G,
            omega_r_dot,
            omega_l_dot
        ])
        return dstate_dt, x_dot_b, y_dot_b, x_ddot_G, y_ddot_G, theta_ddot

    def rk4_step(self, t, state, T_r, T_l, delta):
        """Perform one RK4 integration step."""
        k1, x_dot_b1, y_dot_b1, x_ddot_G1, y_ddot_G1, theta_ddot1 = self.dynamics(t, state, T_r, T_l, delta)
        k2, x_dot_b2, y_dot_b2, x_ddot_G2, y_ddot_G2, theta_ddot2 = self.dynamics(t + self.dt/2, state + self.dt/2 * k1, T_r, T_l, delta)
        k3, x_dot_b3, y_dot_b3, x_ddot_G3, y_ddot_G3, theta_ddot3 = self.dynamics(t + self.dt/2, state + self.dt/2 * k2, T_r, T_l, delta)
        k4, x_dot_b4, y_dot_b4, x_ddot_G4, y_ddot_G4, theta_ddot4 = self.dynamics(t + self.dt, state + self.dt * k3, T_r, T_l, delta)
        next_state = state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x_dot_b = (x_dot_b1 + 2*x_dot_b2 + 2*x_dot_b3 + x_dot_b4) / 6
        y_dot_b = (y_dot_b1 + 2*y_dot_b2 + 2*y_dot_b3 + y_dot_b4) / 6
        x_ddot_G = (x_ddot_G1 + 2*x_ddot_G2 + 2*x_ddot_G3 + x_ddot_G4) / 6
        y_ddot_G = (y_ddot_G1 + 2*y_ddot_G2 + 2*y_ddot_G3 + y_ddot_G4) / 6
        theta_ddot = (theta_ddot1 + 2*theta_ddot2 + 2*theta_ddot3 + theta_ddot4) / 6
        return next_state, x_dot_b, y_dot_b, x_ddot_G, y_ddot_G, theta_ddot

def generate_control_inputs(t, rover, target_velocity=3.0):
    """Generate control inputs for the rover."""
    n_steps = len(t)
    T_r = np.zeros(n_steps)
    T_l = np.zeros(n_steps)
    delta = 0.2 * np.sin(2 * np.pi * t / 10) + 1e-6

    for i in range(n_steps):
        x_dot_b_des = target_velocity
        omega_sum_des = (2 * x_dot_b_des) / (rover.r * np.cos(delta[i]))
        theta_dot_des = (x_dot_b_des * np.tan(delta[i])) / rover.L
        omega_r_des = 0.5 * omega_sum_des * (1 + (rover.W / (2 * rover.L)) * np.tan(delta[i]))
        omega_l_des = 0.5 * omega_sum_des * (1 - (rover.W / (2 * rover.L)) * np.tan(delta[i]))
        x_ddot_b_des = 0.0
        theta_ddot_des = 0.0
        omega_r_dot_des = 0.0
        omega_l_dot_des = 0.0
        T_sum = rover.r * (rover.m * x_ddot_b_des + rover.C_f * x_dot_b_des + rover.C_a * x_dot_b_des**2)
        T_diff = (2 * rover.r * rover.I_z / rover.W) * (theta_ddot_des - (rover.k_delta / rover.I_z) * ((rover.r / rover.L) * (omega_r_des + omega_l_des) * np.tan(delta[i]) - theta_dot_des))
        T_r[i] = 0.5 * (T_sum + T_diff)
        T_l[i] = 0.5 * (T_sum - T_diff)
        T_r[i] += rover.I_w * omega_r_dot_des + rover.b * omega_r_des
        T_l[i] += rover.I_w * omega_l_dot_des + rover.b * omega_l_des
        T_r[i] = np.clip(T_r[i], -10.0, 10.0)
        T_l[i] = np.clip(T_l[i], -10.0, 10.0)
    return T_r, T_l, delta

def simulate_rover(rover, t, T_r, T_l, delta):
    """Simulate the rover dynamics."""
    n_steps = len(t)
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_history = [initial_state]
    x_dot_G_history = [0.0]
    y_dot_G_history = [0.0]
    x_ddot_G_history = [0.0]
    y_ddot_G_history = [0.0]
    theta_ddot_history = [0.0]
    T_r_history = []
    T_l_history = []

    for i in range(n_steps - 1):
        current_state = state_history[-1]
        theta = current_state[2]
        T_r_history.append(T_r[i])
        T_l_history.append(T_l[i])
        next_state, x_dot_b, y_dot_b, x_ddot_G, y_ddot_G, theta_ddot = rover.rk4_step(t[i], current_state, T_r[i], T_l[i], delta[i])
        # Use velocities from the state vector, which are integrated with accelerations
        x_dot_G = current_state[4]  # state[4] is x_dot_G
        y_dot_G = current_state[5]  # state[5] is y_dot_G
        x_dot_G_history.append(x_dot_G)
        y_dot_G_history.append(y_dot_G)
        x_ddot_G_history.append(x_ddot_G)
        y_ddot_G_history.append(y_ddot_G)
        theta_ddot_history.append(theta_ddot)
        state_history.append(next_state)
        if i % 1000 == 0:
            x_G, y_G, theta, _, _, _, omega_r, omega_l = current_state
            print(f"Step {i}, t = {t[i]:.1f}, x_G={x_G:.2f}, y_G={y_G:.2f}, theta={np.rad2deg(theta):.1f}°, delta={np.rad2deg(delta[i]):.1f}°")
    return (np.array(state_history), np.array(x_dot_G_history), np.array(y_dot_G_history),
            np.array(T_r_history), np.array(T_l_history), delta,
            np.array(x_ddot_G_history), np.array(y_ddot_G_history), np.array(theta_ddot_history))

# Simulation setup
rover = RoverDynamics(dt=0.001)
t_final = 10.0
t = np.arange(0, t_final + rover.dt, rover.dt)
n_steps = len(t)
n_steps_adjusted = n_steps

# Generate control inputs
T_r, T_l, delta = generate_control_inputs(t, rover, target_velocity=3.0)

# Run simulation
state_history, x_dot_G_history, y_dot_G_history, T_r_history, T_l_history, delta, x_ddot_G_history, y_ddot_G_history, theta_ddot_history = simulate_rover(
    rover, t, T_r, T_l, delta)

# Extract states
x_G = state_history[:, 0]
y_G = state_history[:, 1]
theta = state_history[:, 2]
theta_dot = state_history[:, 3]
omega_r = state_history[:, 6]
omega_l = state_history[:, 7]

# Debug: Numerically differentiate velocities to compare with accelerations
x_ddot_G_numerical = np.diff(x_dot_G_history) / rover.dt
y_ddot_G_numerical = np.diff(y_dot_G_history) / rover.dt
theta_ddot_numerical = np.diff(theta_dot) / rover.dt
print("Sample comparison at t=5s (index 5000):")
idx = 5000
print(f"x_ddot_G (computed): {x_ddot_G_history[idx]:.4f}, x_ddot_G (numerical): {x_ddot_G_numerical[idx-1]:.4f}")
print(f"y_ddot_G (computed): {y_ddot_G_history[idx]:.4f}, y_ddot_G (numerical): {y_ddot_G_numerical[idx-1]:.4f}")
print(f"theta_ddot (computed): {theta_ddot_history[idx]:.4f}, theta_ddot (numerical): {theta_ddot_numerical[idx-1]:.4f}")

# Set up the figure and subplots for animation
fig = plt.figure(figsize=(15, 15))

# Row 1: Animation and Orientation
ax1 = plt.subplot(3, 3, (1, 2))
traj_line, = ax1.plot([], [], 'b-', label='Actual Trajectory')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Rover Trajectory')
ax1.grid(True)
ax1.axis('equal')
ax1.legend()

# Initialize the vehicle
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

# Center point
center_point, = ax1.plot([], [], 'ro')

# Top Right: Orientation (theta)
ax2 = plt.subplot(3, 3, 3)
ax2.plot(t, theta, 'r-', label='theta')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Theta (rad)')
ax2.set_title('Orientation')
ax2.grid(True)
ax2.legend()

# Row 2: Velocities
ax3 = plt.subplot(3, 3, 4)
ax3.plot(t, x_dot_G_history, 'b-', label='x_dot_G')
ax3.plot(t, y_dot_G_history, 'r-', label='y_dot_G')
ax3.plot(t, theta_dot, 'g-', label='theta_dot')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocities (m/s or rad/s)')
ax3.set_title('Velocities (Global Frame)')
ax3.grid(True)
ax3.legend()

# Accelerations
ax7 = plt.subplot(3, 3, 7)
ax7.plot(t, x_ddot_G_history, 'b-', label='x_ddot_G')
ax7.plot(t, y_ddot_G_history, 'r-', label='y_ddot_G')
ax7.plot(t, theta_ddot_history, 'g-', label='theta_ddot')
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Accelerations (m/s² or rad/s²)')
ax7.set_title('Accelerations (Global Frame)')
ax7.grid(True)
ax7.legend()

# Steering Angle
ax4 = plt.subplot(3, 3, 5)
ax4.plot(t, delta, 'g-', label='delta')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Delta (rad)')
ax4.set_title('Steering Angle')
ax4.grid(True)
ax4.legend()

# Torques
ax5 = plt.subplot(3, 3, 6)
ax5.plot(t[:-1], T_r_history, 'm-', label='T_r')
ax5.plot(t[:-1], T_l_history, 'c-', label='T_l')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Torque (N·m)')
ax5.set_title('Torques')
ax5.grid(True)
ax5.legend()

# Wheel Angular Velocities
ax6 = plt.subplot(3, 3, 9)
ax6.plot(t, omega_r, 'b-', label='omega_r')
ax6.plot(t, omega_l, 'r-', label='omega_l')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Wheel Angular Velocities (rad/s)')
ax6.set_title('Wheel Angular Velocities')
ax6.grid(True)
ax6.legend()

# Power Consumption
ax8 = plt.subplot(3, 3, 8)
power_line, = ax8.plot([], [], 'm-', label='Total Power')
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('Power (W)')
ax8.set_title('Power Consumption')
ax8.grid(True)
ax8.legend()

# Animation update function
def update(frame):
    frame_idx = frame * 100
    if frame_idx >= n_steps_adjusted:
        frame_idx = n_steps_adjusted - 1
    
    x = x_G[frame_idx]
    y = y_G[frame_idx]
    th = theta[frame_idx]
    steer = delta[frame_idx]
    current_state = state_history[frame_idx]
    omega_r_current = current_state[6]
    omega_l_current = current_state[7]
    T_r_current = T_r_history[frame_idx] if frame_idx < len(T_r_history) else T_r_history[-1]
    T_l_current = T_l_history[frame_idx] if frame_idx < len(T_l_history) else T_l_history[-1]
    power_current = T_r_current * omega_r_current + T_l_current * omega_l_current
    
    cos_th = np.cos(th)
    sin_th = np.sin(th)
    vehicle.set_xy((x - vehicle_length/2 * cos_th + vehicle_width/2 * sin_th,
                    y - vehicle_length/2 * sin_th - vehicle_width/2 * cos_th))
    vehicle.set_angle(np.degrees(th))
    
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
    
    center_point.set_data([x], [y])
    traj_line.set_data(x_G[:frame_idx+1], y_G[:frame_idx+1])
    
    power_data = [T_r_history[i] * omega_r[i] + T_l_history[i] * omega_l[i] for i in range(min(frame_idx + 1, len(T_r_history)))]
    power_line.set_data(t[:len(power_data)], power_data)
    ax8.set_xlim(0, t[-1])
    ax8.set_ylim(min(power_data) - 1 if power_data else -10, max(power_data) + 1 if power_data else 10)
    
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
            center_point, traj_line, power_line)

# Create the animation
n_frames = n_steps_adjusted // 100
ani = FuncAnimation(fig, update, frames=range(n_frames), interval=50, blit=True)

# Static figure for all vehicle state variables
fig1, axs = plt.subplots(5, 2, figsize=(15, 15))
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
axs[4, 0].plot(t, theta_dot, 'g-', label='Theta_dot')
axs[4, 0].set_xlabel('Time (s)')
axs[4, 0].set_ylabel('Theta_dot (rad/s)')
axs[4, 0].set_title('Yaw Rate')
axs[4, 0].grid(True)
axs[4, 0].legend()
power = T_r_history * omega_r[:-1] + T_l_history * omega_l[:-1]
axs[4, 1].plot(t[:-1], power, 'm-', label='Total Power')
axs[4, 1].set_xlabel('Time (s)')
axs[4, 1].set_ylabel('Power (W)')
axs[4, 1].set_title('Power Consumption')
axs[4, 1].grid(True)
axs[4, 1].legend()

plt.tight_layout()
plt.savefig('rover_dynamics_plots.png')
plt.show()