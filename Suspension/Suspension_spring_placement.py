import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import scipy.signal

# Suspension parameters
r = 0.25  # Wheel radius (m)
b = 0.140  # Upright length (AB) (m)
a = 0.184  # Upper arm length (MB) (m)
c = 0.231  # Lower arm length (NA) (m)
d = 0.160  # Vertical distance between chassis pivots (m)

# Spring-damper parameters (reduced damping for better oscillation)
k_spring = 1332  # N/m
c_damper = 170    # Reduced to 50 Ns/m
m_unsprung = 10  # kg
m_sprung = 15    # kg
k_tire = 50000    # Reduced to 5000 N/m

# Mount offset distances
spring_mount_lower = 0.05  # 5cm from A along lower arm (m)
spring_mount_upper_positions = np.linspace(0.10 * a, 0.99 * a, 90)  # 10% to 99% along upper arm
optimal_pos = 0.10 * a  # Fixed at 10% for animation

# Chassis pivots
x_N = -0.33
z_N = 0.129  # Adjusted for ±75 mm travel
x_M = -0.28
z_M = z_N + d

# Terrain input
A_road = 0.075  # 7.5 cm amplitude (m)
f_road = 2.0    # Frequency of 2 Hz
t_total = 5     # 5 seconds
fps = 60
t = np.linspace(0, t_total, int(t_total * fps))
delta_t = t[1] - t[0]  # Time step

# Initial spring length calculation at 10% position
initial_z_wheel_center = r
initial_z_B = initial_z_wheel_center + b / 2
initial_z_A = initial_z_wheel_center - b / 2
initial_delta_z_MB = initial_z_B - z_M
initial_disc_MB = a**2 - initial_delta_z_MB**2
initial_x_B = x_M + np.sqrt(initial_disc_MB) if initial_disc_MB >= 0 else x_M
initial_delta_z_NA = initial_z_A - z_N
initial_disc_NA = c**2 - initial_delta_z_NA**2
initial_x_A = x_N + np.sqrt(initial_disc_NA) if initial_disc_NA >= 0 else x_N
initial_ux = (initial_x_B - x_M) / a
initial_uz = (initial_z_B - z_M) / a
initial_x_spring_upper = x_M + initial_ux * optimal_pos
initial_z_spring_upper = z_M + initial_uz * optimal_pos
initial_lx = (initial_x_A - x_N) / c
initial_lz = (initial_z_A - z_N) / c
initial_x_spring_lower = initial_x_A - initial_lx * spring_mount_lower
initial_z_spring_lower = initial_z_A - initial_lz * spring_mount_lower
initial_spring_length = np.sqrt((initial_x_spring_upper - initial_x_spring_lower)**2 + (initial_z_spring_upper - initial_z_spring_lower)**2)

# Setup figure for animation
fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.set_aspect('equal')
ax1.set_xlim(-0.6, 0.4)
ax1.set_ylim(0, 0.8)
ax1.plot(x_N, z_N, 'ko', label='N')
ax1.text(x_N, z_N, 'N', fontsize=12, ha='right')
ax1.plot(x_M, z_M, 'ko', label='M')
ax1.text(x_M, z_M, 'M', fontsize=12, ha='right')

upper_arm_line, = ax1.plot([], [], 'r-', lw=2, label='Upper Arm')
lower_arm_line, = ax1.plot([], [], 'b-', lw=2, label='Lower Arm')
upright_line, = ax1.plot([], [], 'g-', lw=2, label='Upright')
spring_line, = ax1.plot([], [], 'm--', lw=2, label='Spring-Damper')
pointA_dot, = ax1.plot([], [], 'go', label='A (Lower)')
pointA_text = ax1.text(0, 0, 'A', fontsize=12, ha='left', color='green')
pointB_dot, = ax1.plot([], [], 'ro', label='B (Upper)')
pointB_text = ax1.text(0, 0, 'B', fontsize=12, ha='left', color='red')
wheel_patch = Rectangle((0, 0), 0.1, 2 * r, color='gray', alpha=0.5)
ax1.add_patch(wheel_patch)
terrain_line, = ax1.plot([], [], 'k--', lw=1)
ax1.legend()
r_w_text = ax1.text(-0.55, 0.75, 'R_w = 0.000', fontsize=12)

# Data storage
z_spring_upper_records = {optimal_pos: [initial_z_spring_upper]}
spring_length_records = {optimal_pos: [initial_spring_length]}
R_w_records = {optimal_pos: []}
spring_velocity_records = {optimal_pos: [0.0]}
z_wheel_record = [initial_z_wheel_center]

# For dynamic R_w plotting and displacement simulation
r_w_time_history = []
r_w_value_history = []

# Initial values
last_x_B = initial_x_B
last_x_A = initial_x_A
last_x_spring_lower = initial_x_spring_lower
last_z_spring_lower = initial_z_spring_lower
last_spring_length = initial_spring_length

# Update function
def update(frame, data_collection=False):
    global last_x_B, last_x_A, last_x_spring_lower, last_z_spring_lower, last_spring_length

    t_current = t[frame]
    z_terrain = A_road * np.sin(2 * np.pi * f_road * t_current)

    # Double wishbone kinematics
    z_B = z_terrain + r + b / 2
    z_A = z_terrain + r - b / 2
    delta_z_MB = z_B - z_M
    disc_MB = a**2 - delta_z_MB**2
    if disc_MB >= 0:
        x_B = x_M + np.sqrt(disc_MB)
    else:
        x_B = last_x_B
        z_B = z_M + np.sqrt(a**2 - (x_B - x_M)**2) if (a**2 - (x_B - x_M)**2) >= 0 else z_B

    delta_z_NA = z_A - z_N
    disc_NA = c**2 - delta_z_NA**2
    if disc_NA >= 0:
        x_A = x_N + np.sqrt(disc_NA)
    else:
        x_A = last_x_A
        z_A = z_N + np.sqrt(c**2 - (x_A - x_N)**2) if (c**2 - (x_A - x_N)**2) >= 0 else z_A

    # Align upright with left side of wheel
    x_wheel_center = x_A
    z_wheel_center = z_terrain + r
    x_left_tire = x_wheel_center
    x_B = x_left_tire
    x_A = x_left_tire

    upper_arm_line.set_data([x_M, x_B], [z_M, z_B])
    lower_arm_line.set_data([x_N, x_A], [z_N, z_A])
    upright_line.set_data([x_A, x_B], [z_A, z_B])
    pointA_dot.set_data([x_A], [z_A])
    pointA_text.set_position((x_A + 0.02, z_A))
    pointB_dot.set_data([x_B], [z_B])
    pointB_text.set_position((x_B + 0.02, z_B))
    wheel_patch.set_xy((x_wheel_center - 0.05, z_wheel_center - r))
    terrain_line.set_data([x_wheel_center - 0.1, x_wheel_center + 0.1], [z_terrain, z_terrain])

    lx = (x_A - x_N) / c
    lz = (z_A - z_N) / c
    x_spring_lower = x_A - lx * spring_mount_lower
    z_spring_lower = z_A - lz * spring_mount_lower

    ux = (x_B - x_M) / a
    uz = (z_B - z_M) / a
    x_spring_upper = x_M + ux * optimal_pos
    z_spring_upper = z_M + uz * optimal_pos
    if data_collection:
        z_spring_upper_records[optimal_pos].append(z_spring_upper)

    spring_length = np.sqrt((x_spring_upper - x_spring_lower)**2 + (z_spring_upper - z_spring_lower)**2)
    if data_collection:
        spring_length_records[optimal_pos].append(spring_length)

    # Compute R_w for the current timestep
    current_r_w = 0.0
    if len(z_wheel_record) > 1 and len(spring_length_records[optimal_pos]) > 1:
        delta_l_spring = spring_length - spring_length_records[optimal_pos][-2]  # Use -2 to compare with previous
        delta_z_wheel = z_wheel_center - z_wheel_record[-1]
        if abs(delta_z_wheel) > 1e-6:
            current_r_w = abs(delta_l_spring / delta_z_wheel)
        else:
            current_r_w = R_w_records[optimal_pos][-1] if R_w_records[optimal_pos] else 0.0
    else:
        delta_l_spring = spring_length - initial_spring_length
        delta_z_wheel = z_wheel_center - initial_z_wheel_center
        if abs(delta_z_wheel) > 1e-6:
            current_r_w = abs(delta_l_spring / delta_z_wheel)
        else:
            current_r_w = 0.0

    # Store R_w for the plot and displacement simulation
    if data_collection:
        R_w_records[optimal_pos].append(current_r_w)
        r_w_time_history.append(t_current)
        r_w_value_history.append(current_r_w)
        print(f"Frame {frame}, t = {t_current:.2f}s, R_w = {current_r_w:.3f}")  # Debug print

    # Update R_w text in the corner
    r_w_text.set_text(f'R_w = {current_r_w:.3f}')

    if data_collection:
        z_wheel_record.append(z_wheel_center)

    spring_line.set_data([x_spring_upper, x_spring_lower], [z_spring_upper, z_spring_lower])

    ax1.set_title(f"f_terrain = {f_road:.1f} Hz, Spring placement at 10%")

    return upper_arm_line, lower_arm_line, upright_line, spring_line, pointA_dot, pointA_text, pointB_dot, pointB_text, wheel_patch, terrain_line, r_w_text

# Kinematic analysis for optimization plot
spring_length_kinematic = {pos: [] for pos in spring_mount_upper_positions}
R_w_kinematic = {pos: [] for pos in spring_mount_upper_positions}
spring_velocity_kinematic = {pos: [] for pos in spring_mount_upper_positions}
z_wheel_kinematic = [initial_z_wheel_center]
last_x_B = initial_x_B
last_x_A = initial_x_A
last_x_spring_lower = initial_x_spring_lower
last_z_spring_lower = initial_z_spring_lower
last_spring_length = initial_spring_length

for frame in range(len(t)):
    t_current = t[frame]
    z_terrain = A_road * np.sin(2 * np.pi * f_road * t_current)
    z_wheel_center = z_terrain + r
    z_wheel_kinematic.append(z_wheel_center)
    z_B = z_wheel_center + b / 2
    z_A = z_wheel_center - b / 2
    delta_z_MB = z_B - z_M
    disc_MB = a**2 - delta_z_MB**2
    if disc_MB >= 0:
        x_B = x_M + np.sqrt(disc_MB)
    else:
        x_B = last_x_B
        z_B = z_M + np.sqrt(a**2 - (x_B - x_M)**2) if (a**2 - (x_B - x_M)**2) >= 0 else z_B

    delta_z_NA = z_A - z_N
    disc_NA = c**2 - delta_z_NA**2
    if disc_NA >= 0:
        x_A = x_N + np.sqrt(disc_NA)
    else:
        x_A = last_x_A
        z_A = z_N + np.sqrt(c**2 - (x_A - x_N)**2) if (c**2 - (x_A - x_N)**2) >= 0 else z_A

    lx = (x_A - x_N) / c
    lz = (z_A - z_N) / c
    x_spring_lower = x_A - lx * spring_mount_lower
    z_spring_lower = z_A - lz * spring_mount_lower

    for spring_mount_pos in spring_mount_upper_positions:
        ux = (x_B - x_M) / a
        uz = (z_B - z_M) / a
        x_spring_upper = x_M + ux * spring_mount_pos
        z_spring_upper = z_M + uz * spring_mount_pos
        spring_length = np.sqrt((x_spring_upper - x_spring_lower)**2 + (z_spring_upper - z_spring_lower)**2)
        spring_length_kinematic[spring_mount_pos].append(spring_length)

        if len(spring_length_kinematic[spring_mount_pos]) > 1 and len(z_wheel_kinematic) > 1:
            delta_l_spring = spring_length_kinematic[spring_mount_pos][-1] - spring_length_kinematic[spring_mount_pos][-2]
            delta_z_wheel = z_wheel_kinematic[-1] - z_wheel_kinematic[-2]
            if abs(delta_z_wheel) > 1e-6:
                R_w = abs(delta_l_spring / delta_z_wheel)
                R_w_kinematic[spring_mount_pos].append(R_w)
            else:
                R_w_kinematic[spring_mount_pos].append(R_w_kinematic[spring_mount_pos][-1] if R_w_kinematic[spring_mount_pos] else 0.0)

        if len(spring_length_kinematic[spring_mount_pos]) > 1:
            delta_l_spring_t = spring_length_kinematic[spring_mount_pos][-1] - spring_length_kinematic[spring_mount_pos][-2]
            v_spring = delta_l_spring_t / delta_t
            spring_velocity_kinematic[spring_mount_pos].append(v_spring)

    last_x_B = x_B
    last_x_A = x_A
    last_spring_length = spring_length
    last_x_spring_lower = x_spring_lower
    last_z_spring_lower = z_spring_lower

# Compute average R_w and total energy dissipation
avg_R_w = {pos: np.mean(R_w_kinematic[pos]) for pos in spring_mount_upper_positions}
total_Ed = {pos: sum(c_damper * v**2 * delta_t for v in spring_velocity_kinematic[pos]) for pos in spring_mount_upper_positions}

# Normalize R_w and Ed
max_R_w = max(avg_R_w.values())
max_Ed = max(total_Ed.values())
normalized_R_w = {pos: avg_R_w[pos] / max_R_w for pos in spring_mount_upper_positions}
normalized_Ed = {pos: total_Ed[pos] / max_Ed for pos in spring_mount_upper_positions}

# Compute weighted sum
weighted_sum = {pos: 0.5 * normalized_R_w[pos] + 0.5 * normalized_Ed[pos] for pos in spring_mount_upper_positions}

# Data collection for displacement simulation
for frame in range(len(t)):
    update(frame, data_collection=True)

# State-space simulation using recorded R_w
M = np.array([[m_sprung, 0], [0, m_unsprung]])
A = np.zeros((4, 4))
B = np.zeros((4, 1))
C = np.block([[np.eye(2), np.zeros((2, 2))]])
D = np.zeros((2, 1))

# Time-varying effective parameters
k_eff_t = np.array([k_spring * r_w**2 for r_w in r_w_value_history])
c_eff_t = np.array([c_damper * r_w**2 for r_w in r_w_value_history])

# Construct state-space matrices dynamically
z_s = np.zeros_like(t)
z_u = np.zeros_like(t)
for i in range(len(t) - 1):
    K = np.array([[k_eff_t[i], -k_eff_t[i]], [-k_eff_t[i], k_eff_t[i] + k_tire]])
    C_mat = np.array([[c_eff_t[i], -c_eff_t[i]], [-c_eff_t[i], c_eff_t[i]]])
    A[0:2, 2:4] = np.eye(2)
    A[2:4, 0:2] = -np.linalg.inv(M) @ K
    A[2:4, 2:4] = -np.linalg.inv(M) @ C_mat
    B[3] = 1 / m_unsprung  # Force input to unsprung mass
    sys = scipy.signal.StateSpace(A, B, C, D)
    t_segment = t[i:i+2]
    u_segment = k_tire * A_road * np.sin(2 * np.pi * f_road * t_segment)
    t_out, y_out, _ = scipy.signal.lsim(sys, u_segment, t_segment, X0=np.array([z_s[i], z_u[i], 0, 0]))
    z_s[i+1] = y_out[1, 0]
    z_u[i+1] = y_out[1, 1]

# Animation
ani = FuncAnimation(fig, update, frames=len(t), interval=1000/fps, blit=True)
plt.tight_layout()
plt.show()

# Plot R_w vs time
fig2, ax2 = plt.subplots()
ax2.plot(r_w_time_history, r_w_value_history, 'b-', label=f'R_w at 10% Mounting')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Motion Ratio (R_w)')
ax2.set_title('Motion Ratio (R_w) vs Time at 10% Spring Mounting Position')
ax2.grid(True)
ax2.legend()
plt.show()

# Plot R_w and Ed vs position
fig3, ax3 = plt.subplots()
positions = [pos / a * 100 for pos in spring_mount_upper_positions]
ax3.plot(positions, [normalized_R_w[pos] for pos in spring_mount_upper_positions], 'b-o', label='Normalized R_w')
ax3.plot(positions, [normalized_Ed[pos] for pos in spring_mount_upper_positions], 'g-^', label='Normalized Ed')
ax3.plot(positions, [weighted_sum[pos] for pos in spring_mount_upper_positions], 'r--', label='Weighted Sum (0.5 R_w + 0.5 Ed)')
ax3.axvline(10, color='k', linestyle='--', label='Best at 10%')
ax3.set_xlabel('Mounting Position (% of Upper Arm Length)')
ax3.set_ylabel('Normalized Values')
ax3.set_title('Normalized R_w, Ed, and Weighted Sum vs Mounting Position')
ax3.grid(True)
ax3.legend()
plt.show()

# Plot displacements vs time
fig4, ax4 = plt.subplots()
ax4.plot(t, A_road * np.sin(2 * np.pi * f_road * t), 'k--', label='Terrain Input (z_terrain)')
ax4.plot(t, z_s, 'b-', label='Sprung Displacement (z_s)')
ax4.plot(t, z_u, 'g-', label='Unsprung Displacement (z_u)')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Displacement (m)')
ax4.set_title('Terrain Input and Mass Displacements vs Time')
ax4.grid(True)
ax4.legend()
plt.show()