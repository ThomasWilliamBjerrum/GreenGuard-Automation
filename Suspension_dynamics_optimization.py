import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from matplotlib.patches import Rectangle
import os

# Step 1: Define Rover Parameters
m = 200  # Total mass in kg
g = 9.81  # Gravitational acceleration in m/s^2
r = 0.25  # Wheel radius in m
wheel_diameter = 0.5  # Wheel diameter in m
wheel_width = 0.2  # Wheel width (for front view) in m
ground_clearance = 0.1  # Ground clearance in m
wheel_travel = 0.15  # Total wheel travel in m

# Assumptions
f_n = 1.5    # Natural frequency in Hz
zeta = 0.5   # Damping ratio
m_unsprung = 10  # Unsprung mass per wheel in kg

# Step 2: Load Calculations
W = m * g  # Total weight
W_axle = W / 2  # Weight per axle
W_wheel = W_axle / 2  # Weight per wheel
W_unsprung = m_unsprung * g  # Unsprung weight per wheel
W_sprung = W_wheel - W_unsprung  # Sprung weight per wheel
m_sprung = W_sprung / g  # Sprung mass per wheel

print(f"Total Weight: {W:.1f} N")
print(f"Weight per Wheel: {W_wheel:.1f} N")
print(f"Sprung Mass per Wheel: {m_sprung:.1f} kg")

# Step 3: Suspension Geometry (Before Rotation)
x_wheel = 0  # Wheel center at x = 0
z_wheel = r  # Wheel center at z = 0.25 m
b = 0.14  # Length of upright AB (140 mm)
wheel_left_edge = x_wheel - wheel_width / 2  # Left edge of the wheel

# Upright AB at the left edge of the wheel
x_A = x_B = wheel_left_edge
z_A = z_wheel + b / 2
z_B = z_wheel - b / 2

# Initial arm lengths
a_initial = 0.20  # Upper arm length (MB)
c_initial = 0.24  # Lower arm length (NA)
d = 0.16  # Vertical separation between M and N

# Position M and N to the left of AB
x_N = x_A - c_initial  # Lower pivot N
z_N = 0.20  # Adjusted to balance camber
x_M = x_N  # Upper pivot M
z_M = z_N + d

# Step 4: Camber Angle Calculation (Kinematic Approach)
def calculate_camber_kinematic(a, c, d, b, delta_z, z_N_local, z_M_local):
    z_B = z_wheel - b / 2
    z_A = z_wheel + b / 2
    z_B_new = z_B + delta_z
    z_A_new = z_A + delta_z
    
    discriminant_NA = c**2 - (z_A_new - z_N_local)**2
    if discriminant_NA < 0:
        return np.nan, np.nan, np.nan, np.nan
    delta_x_NA = np.sqrt(discriminant_NA)
    x_A = x_N + delta_x_NA
    
    discriminant_MB = a**2 - (z_B_new - z_M_local)**2
    if discriminant_MB < 0:
        return np.nan, np.nan, np.nan, np.nan
    delta_x_MB = np.sqrt(discriminant_MB)
    x_B = x_M + delta_x_MB
    
    theta3 = np.arctan2(z_A_new - z_B_new, x_A - x_B) - np.pi / 2
    
    return np.degrees(theta3), x_A, z_A_new, x_B, z_B_new

# Step 4.5: Compute initial camber variation
delta_z_values = np.linspace(-0.075, 0.075, 100)
camber_angles_initial = []
for dz in delta_z_values:
    result = calculate_camber_kinematic(a_initial, c_initial, d, b, dz, z_N, z_M)
    if len(result) != 5 or np.any(np.isnan(result)):
        camber_angles_initial.append(np.nan)
        continue
    theta3, _, _, _, _ = result
    camber_angles_initial.append(theta3)

# Step 5: Optimization
def objective(params):
    z_N_offset, z_M_offset, a_offset, c_offset = params
    z_N_local = z_N + z_N_offset
    z_M_local = z_N_local + d + z_M_offset
    a = a_initial + a_offset
    c = c_initial + c_offset
    delta_z_values = np.linspace(-0.075, 0.075, 50)
    camber_angles = []
    R_w_values = []
    S_a_values = []
    S_b_values = []
    horizontal_penalty = 0
    parallelism_penalty = 0
    height_penalty = 0
    negative_Rw_penalty = 0
    Rw_var_penalty = 0
    motion_ratio_penalty = 0

    # Check geometric feasibility
    if a <= 0 or c <= 0 or z_M_local <= z_N_local:
        return 1e14

    for delta_z in delta_z_values:
        result = calculate_camber_kinematic(a, c, d, b, delta_z, z_N_local, z_M_local)
        if len(result) != 5 or np.any(np.isnan(result)):
            return 1e12
        theta3, x_A_new, z_A_new, x_B_new, z_B_new = result
        camber_angles.append(theta3)

        # Compute motion ratios
        phi = np.arctan2(z_A_new - z_B_new, x_A_new - x_B_new)
        theta = np.arctan2(x_A_new - x_B_new, z_A_new - z_B_new)
        denom = np.sqrt(np.cos(phi)**2 + np.cos(theta)**2 * np.sin(phi)**2)
        if denom == 0:
            return 1e12
        R_w = (np.cos(theta) * np.cos(phi)) / denom
        S_a = (np.cos(theta) * np.sin(phi)) / denom
        S_b = (-np.cos(phi) * np.sin(theta)) / denom
        R_w_values.append(R_w)
        S_a_values.append(S_a)
        S_b_values.append(S_b)

        # Strong penalties for R_w
        if R_w <= 0:
            negative_Rw_penalty += (0.3 - R_w) * 5e12
        elif R_w < 0.3:
            negative_Rw_penalty += (0.3 - R_w) * 1e8  # Stronger penalty for low R_w
        if R_w > 0.7:
            negative_Rw_penalty += (R_w - 0.7) * 1e8

        # Geometric penalties
        theta_MB = np.arctan2(z_B_new - z_M_local, x_B_new - x_M)
        theta_NA = np.arctan2(z_A_new - z_N_local, x_A_new - x_N)
        horizontal_penalty += (np.abs(theta_MB) + np.abs(theta_NA)) * 1e3
        parallelism_penalty += np.abs(theta_MB - theta_NA) * 5e3

        t_values = np.linspace(0, 1, 20)
        for t in t_values:
            z_MB = (1 - t) * z_M_local + t * z_B_new
            z_NA = (1 - t) * z_N_local + t * z_A_new
            if z_MB < z_NA:
                height_penalty += (z_NA - z_MB) * 5e5  # Reduced to balance

    if not camber_angles:
        return 1e14
    camber_var = np.max(camber_angles) - np.nanmin(camber_angles)
    camber_mean = np.nanmean(camber_angles)
    camber_mean_penalty = np.abs(camber_mean) * 5e6  # Stronger
    camber_var_penalty = 10000 * camber_var  # Stronger

    # Penalize motion ratio variations
    if R_w_values:
        Rw_var_penalty = (np.nanmax(R_w_values) - np.nanmin(R_w_values)) * 5e5  # Stronger
    if S_a_values and S_b_values:
        motion_ratio_penalty = (np.nanmax(S_a_values) - np.nanmin(S_a_values)) * 1e5  # Stronger
        if np.any(np.array(S_b_values) < 0):
            motion_ratio_penalty += np.sum(-np.array(S_b_values)[np.array(S_b_values) < 0]) * 2e5  # Stronger

    total_penalty = (camber_var_penalty + camber_mean_penalty + horizontal_penalty +
                     parallelism_penalty + height_penalty + negative_Rw_penalty +
                     Rw_var_penalty + motion_ratio_penalty)

    # Log penalty contributions
    print(f"Camber Var: {camber_var_penalty:.2e}, Camber Mean: {camber_mean_penalty:.2e}, "
          f"Horizontal: {horizontal_penalty:.2e}, Parallelism: {parallelism_penalty:.2e}, "
          f"Height: {height_penalty:.2e}, Negative R_w: {negative_Rw_penalty:.2e}, "
          f"R_w Var: {Rw_var_penalty:.2e}, Motion Ratio: {motion_ratio_penalty:.2e}")

    return total_penalty

# Bounds for differential_evolution
bounds = [
    (-0.005, 0.005),  # z_N_offset (extremely tight)
    (-0.005, 0.005),  # z_M_offset (extremely tight)
    (-0.02, 0.02),    # a_offset (tighter)
    (-0.02, 0.02)     # c_offset (tighter)
]

# Perform optimization with error handling
try:
    result = differential_evolution(objective, bounds, maxiter=1000, popsize=30)  # Increased popsize
    z_N_offset, z_M_offset, a_offset, c_offset = result.x
    z_N_opt = z_N + z_N_offset
    z_M_opt = z_N_opt + d + z_M_offset
    a_opt = a_initial + a_offset
    c_opt = c_initial + c_offset
    print(f"Optimized z_N: {z_N_opt:.3f} m, z_M: {z_M_opt:.3f} m")
    print(f"Optimized a: {a_opt:.3f} m, c: {c_opt:.3f} m")
except Exception as e:
    print(f"Optimization failed: {e}")
    exit(1)

# Step 6: Spring and Damping Constants
k = (2 * np.pi * f_n)**2 * m_sprung
c_damp = 2 * zeta * np.sqrt(k * m_sprung)
print(f"Spring Constant k: {k:.2f} N/m")
print(f"Damping Constant c: {c_damp:.2f} Ns/m")

# Step 7: Plotting
delta_z_values = np.linspace(-0.075, 0.075, 100)
camber_angles = []
phi_angles = []
theta_angles = []
S_a_values = []
S_b_values = []
R_w_values = []

try:
    for dz in delta_z_values:
        result = calculate_camber_kinematic(a_opt, c_opt, d, b, dz, z_N_opt, z_M_opt)
        if len(result) != 5 or np.any(np.isnan(result)):
            camber_angles.append(np.nan)
            phi_angles.append(np.nan)
            theta_angles.append(np.nan)
            S_a_values.append(np.nan)
            S_b_values.append(np.nan)
            R_w_values.append(np.nan)
            continue
        theta3, x_A, z_A_new, x_B, z_B_new = result
        camber_angles.append(theta3)
        
        phi = np.arctan2(z_A_new - z_B_new, x_A - x_B)
        theta = np.arctan2(x_A - x_B, z_A_new - z_B_new)
        phi_angles.append(np.degrees(phi))
        theta_angles.append(np.degrees(theta))
        
        denom = np.sqrt(np.cos(phi)**2 + np.cos(theta)**2 * np.sin(phi)**2)
        if denom == 0:
            S_a_values.append(np.nan)
            S_b_values.append(np.nan)
            R_w_values.append(np.nan)
            continue
        S_a = (np.cos(theta) * np.sin(phi)) / denom
        S_b = (-np.cos(phi) * np.sin(theta)) / denom
        R_w = (np.cos(theta) * np.cos(phi)) / denom
        S_a_values.append(S_a)
        S_b_values.append(S_b)
        R_w_values.append(R_w)
except Exception as e:
    print(f"Error in data collection for plotting: {e}")
    exit(1)

# Ensure the working directory is writable
output_dir = os.getcwd()
print(f"Saving plots to: {output_dir}")

# Plot 1: Camber Angle Variation
try:
    plt.figure(figsize=(10, 6))
    plt.plot(delta_z_values * 1000, camber_angles_initial, 'r--', label='Camber Angle (Initial)')
    plt.plot(delta_z_values * 1000, camber_angles, 'b-', label='Camber Angle (Optimized)')
    plt.xlabel('Wheel Displacement (mm)')
    plt.ylabel('Camber Angle (degrees)')
    plt.title('Camber Angle Variation: Initial vs. Optimized')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'camber_angle_variation.png'))
    plt.show()
except Exception as e:
    print(f"Failed to generate Camber Angle Variation plot: {e}")

# Plot 2: Theta and Phi vs. Wheel Displacement
try:
    plt.figure(figsize=(10, 6))
    plt.plot(delta_z_values * 1000, phi_angles, 'r-', label=r'$\phi$ (degrees)')
    plt.plot(delta_z_values * 1000, theta_angles, 'g-', label=r'$\theta$ (degrees)')
    plt.xlabel('Wheel Displacement (mm)')
    plt.ylabel('Angle (degrees)')
    plt.title('Upright Orientation Angles Over Wheel Travel')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'theta_phi_angles.png'))
    plt.show()
except Exception as e:
    print(f"Failed to generate Theta and Phi plot: {e}")

# Plot 3: Motion Ratios vs. Wheel Displacement
try:
    plt.figure(figsize=(10, 6))
    plt.plot(delta_z_values * 1000, S_a_values, 'b-', label=r'$S_a$')
    plt.plot(delta_z_values * 1000, S_b_values, 'r-', label=r'$S_b$')
    plt.plot(delta_z_values * 1000, R_w_values, 'g-', label=r'$R_w$')
    plt.xlabel('Wheel Displacement (mm)')
    plt.ylabel('Motion Ratio')
    plt.title('Motion Ratios Over Wheel Travel')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'motion_ratios.png'))
    plt.show()
except Exception as e:
    print(f"Failed to generate Motion Ratios plot: {e}")

# Plot 4: Effective Spring and Damping Rates
try:
    k_wheel = [k * (rw**2) if not np.isnan(rw) and rw > 0 else np.nan for rw in R_w_values]
    c_wheel = [c_damp * (rw**2) if not np.isnan(rw) and rw > 0 else np.nan for rw in R_w_values]

    plt.figure(figsize=(10, 6))
    plt.plot(delta_z_values * 1000, k_wheel, 'b-', label='Effective Spring Rate (N/m)')
    plt.plot(delta_z_values * 1000, c_wheel, 'r-', label='Effective Damping Rate (Ns/m)')
    plt.xlabel('Wheel Displacement (mm)')
    plt.ylabel('Rate')
    plt.title('Effective Spring and Damping Rates at the Wheel')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'spring_damping_rates.png'))
    plt.show()
except Exception as e:
    print(f"Failed to generate Spring and Damping Rates plot: {e}")

# Step 8: Detailed Suspension Geometry Schematic (Rotated 90° CCW)
try:
    z_N = z_N_opt
    z_M = z_M_opt
    z_A = z_wheel + b / 2
    z_B = z_wheel - b / 2
    z_A_nom = z_A
    z_B_nom = z_B

    result = calculate_camber_kinematic(a_opt, c_opt, d, b, 0, z_N_opt, z_M_opt)
    if len(result) != 5 or np.any(np.isnan(result)):
        x_A_nom = x_B_nom = wheel_left_edge
    else:
        _, x_A_nom, _, x_B_nom, _ = result

    points = {
        'N': (x_N, z_N),
        'M': (x_M, z_M),
        'A': (x_A_nom, z_A_nom),
        'B': (x_B_nom, z_B_nom),
        'O': (x_wheel, z_wheel)
    }

    rotated_points = {key: (-z, x) for key, (x, z) in points.items()}
    x_N_rot, z_N_rot = rotated_points['N']
    x_M_rot, z_M_rot = rotated_points['M']
    x_A_rot, z_A_rot = rotated_points['A']
    x_B_rot, z_B_rot = rotated_points['B']
    x_wheel_rot, z_wheel_final = rotated_points['O']

    final_points = {key: (-z, x) for key, (x, z) in rotated_points.items()}
    x_N_final, z_N_final = final_points['N']
    x_M_final, z_M_final = final_points['M']
    x_A_final, z_A_final = final_points['A']
    x_B_final, z_B_final = final_points['B']
    x_wheel_final, z_wheel_final = final_points['O']

    x_A_final = x_B_final = x_wheel_final - wheel_width / 2
    z_A_final = z_wheel_final + b / 2
    z_B_final = z_wheel_final - b / 2

    delta_z_MB = z_M_final - z_B_final
    discriminant_MB = a_opt**2 - delta_z_MB**2
    if discriminant_MB < 0:
        x_M_final = x_B_final - a_opt * 0.9
    else:
        delta_x_MB = np.sqrt(discriminant_MB)
        x_M_final = x_B_final - delta_x_MB

    delta_z_NA = z_N_final - z_A_final
    discriminant_NA = c_opt**2 - delta_z_NA**2
    if discriminant_NA < 0:
        x_N_final = x_A_final - c_opt * 0.9
    else:
        delta_x_NA = np.sqrt(discriminant_NA)
        x_N_final = x_A_final - delta_x_NA

    if z_wheel_final < 0:
        z_wheel_final = -z_wheel_final
        z_A_final = -z_A_final
        z_B_final = -z_B_final
        z_M_final = -z_M_final
        z_N_final = -z_N_final

    plt.figure(figsize=(12, 8))
    plt.axhline(y=0, color='k', linestyle='--', label='Ground')
    plt.axhline(y=ground_clearance, color='gray', linestyle='--', label='Chassis Bottom (0.1 m)')
    plt.plot([x_N_final, x_M_final], [z_N_final, z_M_final], 'k-', label='Chassis', linewidth=2)
    plt.plot([x_N_final, x_A_final], [z_N_final, z_A_final], 'b-', label='Lower Arm', linewidth=2)
    plt.plot([x_M_final, x_B_final], [z_M_final, z_B_final], 'r-', label='Upper Arm', linewidth=2)
    plt.plot([x_B_final, x_A_final], [z_B_final, z_A_final], 'g-', label='Upright', linewidth=2)

    x_mid_AB = (x_A_final + x_B_final) / 2
    z_mid_AB = (z_A_final + z_B_final) / 2
    plt.plot([x_wheel_final, x_mid_AB], [z_wheel_final, z_mid_AB], 'k-', linewidth=2, label='Wheel to Upright Link')

    wheel_rect = Rectangle(
        (x_wheel_final - wheel_width / 2, z_wheel_final - wheel_diameter / 2),
        wheel_width, wheel_diameter, edgecolor='k', facecolor='gray', alpha=0.5, label='Wheel'
    )
    plt.gca().add_patch(wheel_rect)

    plt.plot(x_N_final, z_N_final, 'ko')
    plt.plot(x_M_final, z_M_final, 'ko')
    plt.plot(x_A_final, z_A_final, 'ko')
    plt.plot(x_B_final, z_B_final, 'ko')
    plt.plot(x_wheel_final, z_wheel_final, 'ko')

    plt.text(x_N_final, z_N_final + 0.02, 'N', fontsize=12, ha='center')
    plt.text(x_M_final, z_M_final + 0.02, 'M', fontsize=12, ha='center')
    plt.text(x_A_final, z_A_final + 0.02, 'A', fontsize=12, ha='center')
    plt.text(x_B_final, z_B_final - 0.02, 'B', fontsize=12, ha='center')
    plt.text(x_wheel_final, z_wheel_final + 0.02, 'O', fontsize=12, ha='center')

    plt.text((x_M_final + x_B_final) / 2, (z_M_final + z_B_final) / 2, f'a = {a_opt:.3f} m', fontsize=10, color='r')
    plt.text((x_N_final + x_A_final) / 2, (z_N_final + z_A_final) / 2, f'c = {c_opt:.3f} m', fontsize=10, color='b')
    plt.text((x_N_final + x_M_final) / 2, (z_N_final + z_M_final) / 2, f'd = {d:.3f} m', fontsize=10, color='k')
    plt.text((x_A_final + x_B_final) / 2, (z_A_final + z_B_final) / 2, f'b = {b:.3f} m', fontsize=10, color='g')

    legend_table = (
        f"Values:\n"
        f"a: {a_opt:.3f} m\n"
        f"c: {c_opt:.3f} m\n"
        f"d: {d:.3f} m\n"
        f"b: {b:.3f} m"
    )

    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.title('Optimized Double Wishbone Suspension Geometry (Rotated 90° CCW)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title=legend_table)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'suspension_geometry.png'))
    plt.show()
except Exception as e:
    print(f"Failed to generate Suspension Geometry plot: {e}")

# Step 9: Report Result Parameters
try:
    print("\nResult Parameters:")
    print(f"Initial Camber Variation: {np.nanmax(camber_angles_initial) - np.nanmin(camber_angles_initial):.2f} degrees")
    print(f"Optimized Camber Variation: {np.nanmax(camber_angles) - np.nanmin(camber_angles):.2f} degrees")
    print(f"Mean Optimized Camber: {np.nanmean(camber_angles):.2f} degrees")
    print(f"Phi Range: {np.nanmin(phi_angles):.2f} to {np.nanmax(phi_angles):.2f} degrees")
    print(f"Theta Range: {np.nanmin(theta_angles):.2f} to {np.nanmax(theta_angles):.2f} degrees")
    print(f"S_a Range: {np.nanmin(S_a_values):.3f} to {np.nanmax(S_a_values):.3f}")
    print(f"S_b Range: {np.nanmin(S_b_values):.3f} to {np.nanmax(S_b_values):.3f}")
    print(f"R_w Range: {np.nanmin(R_w_values):.3f} to {np.nanmax(R_w_values):.3f}")
    print(f"Effective Spring Rate Range: {np.nanmin(k_wheel):.0f} to {np.nanmax(k_wheel):.0f} N/m")
    print(f"Effective Damping Rate Range: {np.nanmin(c_wheel):.0f} to {np.nanmax(c_wheel):.0f} Ns/m")
except Exception as e:
    print(f"Failed to generate result parameters: {e}")