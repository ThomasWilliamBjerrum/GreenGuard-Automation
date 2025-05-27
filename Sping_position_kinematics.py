import numpy as np
import matplotlib.pyplot as plt

# === Suspension parameters ===
r = 0.25  # Wheel radius (m)
b = 0.140  # Upright length (AB) (m)
a = 0.184  # Upper arm length (MB) (m)
c = 0.231  # Lower arm length (NA) (m)
d = 0.160  # Vertical distance between chassis pivots (m)

# Spring-damper parameters
k_spring = 3500  # N/m
c_damper = 450   # Ns/m

# Mount offset distances
spring_mount_lower = 0.05  # 5cm from A along lower arm (m)
spring_mount_upper_positions = np.linspace(0.00 * a, 0.99 * a, 100)  # 100 positions from 0% to 99%

# Chassis pivots
x_N = -0.33
z_N = 0.20
x_M = -0.28
z_M = z_N + d

# === Terrain input ===
A_road = 0.075  # 7.5 cm amplitude (m)
f_sweep = np.linspace(0.5, 5, 10)  # Sweeping frequencies (Hz)
t_total = 20
fps = 60
t = np.linspace(0, t_total, int(t_total * fps))

# Initial spring length (using last position as reference)
initial_z_wheel_center = r
initial_z_B = initial_z_wheel_center + b / 2
initial_z_A = initial_z_wheel_center - b / 2
initial_delta_z_MB = initial_z_B - z_M
initial_disc_MB = a**2 - initial_delta_z_MB**2
initial_x_B = x_M + np.sqrt(initial_disc_MB) if initial_disc_MB >= 0 else x_M
initial_delta_z_NA = initial_z_A - z_N
initial_disc_NA = c**2 - initial_delta_z_NA**2
initial_x_A = x_N + np.sqrt(initial_disc_NA) if initial_disc_NA >= 0 else x_N
initial_ux = (initial_x_B - x_M) / a if not np.isnan(initial_x_B) else 0
initial_uz = (initial_z_B - z_M) / a
initial_x_spring_upper = x_M + initial_ux * spring_mount_upper_positions[-1]
initial_z_spring_upper = z_M + initial_uz * spring_mount_upper_positions[-1]
initial_lx = (initial_x_A - x_N) / c if not np.isnan(initial_x_A) else 0
initial_lz = (initial_z_A - z_N) / c
initial_x_spring_lower = initial_x_A - initial_lx * spring_mount_lower if not np.isnan(initial_x_A) else x_N
initial_z_spring_lower = initial_z_A - initial_lz * spring_mount_lower
initial_spring_length = np.sqrt((initial_x_spring_upper - initial_x_spring_lower)**2 + (initial_z_spring_upper - initial_z_spring_lower)**2)

# Data recording for kinematic analysis
z_spring_upper_kinematic = {pos: [] for pos in spring_mount_upper_positions}
z_spring_lower_kinematic = {pos: [] for pos in spring_mount_upper_positions}
spring_length_kinematic = {pos: [] for pos in spring_mount_upper_positions}
R_w_kinematic = {pos: [] for pos in spring_mount_upper_positions}
spring_velocity_kinematic = {pos: [] for pos in spring_mount_upper_positions}
z_wheel_kinematic = []

# Last valid values for fallback
last_x_B = x_M + a
last_x_A = x_N + c
last_x_spring_lower = last_x_A - spring_mount_lower
last_z_spring_lower = z_N
last_spring_length = initial_spring_length
last_R_w = {pos: 0 for pos in spring_mount_upper_positions}
last_x_wheel_center = (last_x_A + last_x_B) / 2

# === Kinematic Analysis ===
for frame in range(len(t)):
    t_current = t[frame]
    freq_idx = int(frame / len(t) * len(f_sweep))
    f_road = f_sweep[freq_idx % len(f_sweep)]
    
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

    delta_z_NA = z_A - z_N
    disc_NA = c**2 - delta_z_NA**2
    if disc_NA >= 0:
        x_A = x_N + np.sqrt(disc_NA)
    else:
        x_A = last_x_A

    x_wheel_center = (x_A + x_B) / 2

    lx = (x_A - x_N) / c if not np.isnan(x_A) else (last_x_A - x_N) / c
    lz = (z_A - z_N) / c
    x_spring_lower = x_A - lx * spring_mount_lower
    z_spring_lower = z_A - lz * spring_mount_lower

    for spring_mount_pos in spring_mount_upper_positions:
        ux = (x_B - x_M) / a if not np.isnan(x_B) else (last_x_B - x_M) / a
        uz = (z_B - z_M) / a
        x_spring_upper = x_M + ux * spring_mount_pos
        z_spring_upper = z_M + uz * spring_mount_pos

        # Compute spring length for this position
        spring_length = np.sqrt((x_spring_upper - x_spring_lower)**2 + (z_spring_upper - z_spring_lower)**2)

        # Record positions of both ends
        z_spring_upper_kinematic[spring_mount_pos].append(z_spring_upper)
        z_spring_lower_kinematic[spring_mount_pos].append(z_spring_lower)

        # Record spring length
        if np.isnan(spring_length):
            spring_length = last_spring_length
        spring_length_kinematic[spring_mount_pos].append(spring_length)

        # Compute relative vertical displacement of the spring
        if len(z_spring_upper_kinematic[spring_mount_pos]) > 1:
            prev_z_s_upper = z_spring_upper_kinematic[spring_mount_pos][-2]
            prev_z_s_lower = z_spring_lower_kinematic[spring_mount_pos][-2]
            delta_z_s = (z_spring_upper - z_spring_lower) - (prev_z_s_upper - prev_z_s_lower)
        else:
            delta_z_s = 0.0

        # Compute wheel vertical displacement
        if len(z_wheel_kinematic) > 1:
            delta_z_wheel = z_wheel_kinematic[-1] - z_wheel_kinematic[-2]
        else:
            delta_z_wheel = 0.0

        # Store displacements for later accumulation (temporary storage per frame)
        if frame == 0:
            z_spring_upper_kinematic[spring_mount_pos].append(z_spring_upper)
            z_spring_lower_kinematic[spring_mount_pos].append(z_spring_lower)
            delta_z_s = 0.0
            delta_z_wheel = 0.0

    # Update last valid values
    if not np.isnan(x_B):
        last_x_B = x_B
    if not np.isnan(x_A):
        last_x_A = x_A
    if not np.isnan(spring_length):
        last_spring_length = spring_length
    if not np.isnan(x_spring_lower):
        last_x_spring_lower = x_spring_lower
    last_z_spring_lower = z_spring_lower
    last_x_wheel_center = x_wheel_center

# Compute accumulated displacements and R_w
total_delta_z_s = {pos: 0.0 for pos in spring_mount_upper_positions}
total_delta_z_wheel = 0.0
for frame in range(1, len(t)):
    for spring_mount_pos in spring_mount_upper_positions:
        prev_z_s_upper = z_spring_upper_kinematic[spring_mount_pos][frame - 1]
        prev_z_s_lower = z_spring_lower_kinematic[spring_mount_pos][frame - 1]
        curr_z_s_upper = z_spring_upper_kinematic[spring_mount_pos][frame]
        curr_z_s_lower = z_spring_lower_kinematic[spring_mount_pos][frame]
        delta_z_s = (curr_z_s_upper - curr_z_s_lower) - (prev_z_s_upper - prev_z_s_lower)
        total_delta_z_s[spring_mount_pos] += abs(delta_z_s)

    prev_z_wheel = z_wheel_kinematic[frame - 1]
    curr_z_wheel = z_wheel_kinematic[frame]
    delta_z_wheel = curr_z_wheel - prev_z_wheel
    total_delta_z_wheel += abs(delta_z_wheel)

# Compute R_w for each position
avg_R_w = {pos: total_delta_z_s[pos] / total_delta_z_wheel if total_delta_z_wheel > 0 else 0.0 for pos in spring_mount_upper_positions}

# Compute energy dissipation
spring_velocity_kinematic = {pos: [] for pos in spring_mount_upper_positions}
for frame in range(1, len(t)):
    delta_t = t[frame] - t[frame - 1]
    for spring_mount_pos in spring_mount_upper_positions:
        prev_z_s_upper = z_spring_upper_kinematic[spring_mount_pos][frame - 1]
        prev_z_s_lower = z_spring_lower_kinematic[spring_mount_pos][frame - 1]
        curr_z_s_upper = z_spring_upper_kinematic[spring_mount_pos][frame]
        curr_z_s_lower = z_spring_lower_kinematic[spring_mount_pos][frame]
        delta_z_s_t = (curr_z_s_upper - curr_z_s_lower) - (prev_z_s_upper - prev_z_s_lower)
        v_spring = delta_z_s_t / delta_t if delta_t > 0 else 0
        spring_velocity_kinematic[spring_mount_pos].append(v_spring)

energy_dissipation = {}
for pos in spring_mount_upper_positions:
    valid_v_spring = [v for v in spring_velocity_kinematic[pos] if not np.isnan(v)]
    if valid_v_spring:
        v_spring_rms = np.sqrt(np.mean(np.array(valid_v_spring)**2))
        energy_dissipation[pos] = c_damper * v_spring_rms**2 * t_total
    else:
        energy_dissipation[pos] = 0.0

# Normalize R_w and energy dissipation
max_R_w = max(avg_R_w.values()) if avg_R_w.values() else 1.0
max_energy = max(energy_dissipation.values()) if energy_dissipation.values() else 1.0
normalized_R_w = {pos: avg_R_w[pos] / max_R_w for pos in spring_mount_upper_positions}
normalized_energy = {pos: energy_dissipation[pos] / max_energy for pos in spring_mount_upper_positions}

# Combined objective function (50/50 weighting)
w1 = 0.5  # Weight for normalized R_w
w2 = 0.5  # Weight for normalized energy
combined_objective = {pos: w1 * normalized_R_w[pos] + w2 * normalized_energy[pos] for pos in spring_mount_upper_positions}

# Find the optimal position
optimal_pos = max(combined_objective, key=combined_objective.get)
print(f"Optimal spring placement (maximizing R_w and energy dissipation): {optimal_pos / a * 100:.1f}% along upper arm MB")
print(f"Average R_w at optimal position: {avg_R_w[optimal_pos]:.3f} (max normalized to 1.0)")
print(f"Energy dissipation at optimal position: {energy_dissipation[optimal_pos]:.2f} J (proportional)")

# === Plot Normalized R_w and Energy Dissipation ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([pos / a * 100 for pos in spring_mount_upper_positions], [normalized_R_w[pos] for pos in spring_mount_upper_positions], 'b-o', label=f'Normalized R_w (max={max_R_w:.3f})')
ax.plot([pos / a * 100 for pos in spring_mount_upper_positions], [normalized_energy[pos] for pos in spring_mount_upper_positions], 'g-^', label='Normalized Energy Dissipation')
ax.axvline(optimal_pos / a * 100, color='r', linestyle='--', label=f'Optimal: {optimal_pos / a * 100:.1f}%')
ax.set_xlabel('Mounting Position (% of Upper Arm Length)')
ax.set_ylabel('Normalized Values')
ax.set_title('Normalized R_w and Energy Dissipation vs Mounting Position')
ax.grid(True)
ax.legend()
ax.text(0.5, -0.2, f'Note: Normalized R_w is scaled to 1.0 where max avg_R_w = {max_R_w:.3f}', transform=ax.transAxes, ha='center')

plt.show()