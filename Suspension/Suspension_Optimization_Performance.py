import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------------
# Constants and Setup 
# -------------------------------
m = 100
g = 9.81
r = 0.25
wheel_diameter = 0.5
wheel_width = 0.2
f_n = 1.5
zeta = 0.6
m_unsprung = 10
W = m * g
W_axle = W / 2
W_wheel = W_axle / 2
W_unsprung = m_unsprung * g
W_sprung = W_wheel - W_unsprung
m_sprung = W_sprung / g
k = (2 * np.pi * f_n)**2 * m_sprung
c_damp = 2 * zeta * np.sqrt(k * m_sprung)
x_wheel = 0
z_wheel = r
wheel_left_edge = x_wheel - wheel_width / 2
d = 0.160
b = 0.140

# -------------------------------
# Geometry + Camber Function
# -------------------------------
def calculate_camber_kinematic(a, c, d, b, delta_z, z_N, z_M, x_N, x_M):
    z_A0 = z_wheel + b / 2
    z_B0 = z_wheel - b / 2
    z_A = z_A0 + delta_z
    z_B = z_B0 + delta_z
    Dna = c**2 - (z_A - z_N)**2
    if Dna < 0:
        return (np.nan,) * 7
    dx_NA = np.sqrt(Dna)
    x_A = x_N + dx_NA
    Dmb = a**2 - (z_B - z_M)**2
    if Dmb < 0:
        return (np.nan,) * 7
    dx_MB = np.sqrt(Dmb)
    x_B = x_M + dx_MB
    theta3 = np.arctan2(z_A - z_B, x_A - x_B) - np.pi / 2
    m1 = (z_A - z_N) / (x_A - x_N) if x_A != x_N else np.inf
    m2 = (z_B - z_M) / (x_B - x_M) if x_B != x_M else np.inf
    if m1 == m2:
        rc = np.nan
    else:
        if m1 == np.inf:
            x_rc = x_N
            z_rc = m2 * (x_rc - x_M) + z_M
        elif m2 == np.inf:
            x_rc = x_M
            z_rc = m1 * (x_rc - x_N) + z_N
        else:
            x_rc = (z_M - z_N + m1 * x_N - m2 * x_M) / (m1 - m2)
            z_rc = m1 * (x_rc - x_N) + z_N
        rc = z_rc
    x_mid = (x_A + x_B) / 2
    return np.degrees(theta3), x_A, z_A, x_B, z_B, rc, x_mid

init = dict(a=0.200, c=0.200, d=d, b=b)
opt = dict(a=0.184, c=0.231, d=d, b=b)

def pivots(geom):
    a, c, d, b = geom['a'], geom['c'], geom['d'], geom['b']
    zN = 0.129
    zM = zN + d
    delta_z = (z_wheel + b / 2) - zN
    dx = np.sqrt(c**2 - delta_z**2)
    xN = wheel_left_edge - dx
    return xN, xN, zN, zM

xN_i, xM_i, zN_i, zM_i = pivots(init)
xN_o, xM_o, zN_o, zM_o = pivots(opt)

# -------------------------------
# Metric Computation
# -------------------------------
delta_z_values = np.linspace(-0.075, 0.075, 1000)

def compute_metrics(geom, xN, xM, zN, zM):
    camber, rc, theta, S_a, S_b, R_w, lateral_disp = [], [], [], [], [], [], []
    for dz in delta_z_values:
        t3, xA, zA, xB, zB, h_rc, x_mid = calculate_camber_kinematic(
            geom['a'], geom['c'], geom['d'], geom['b'], dz, zN, zM, xN, xM)
        camber.append(t3)
        rc.append(h_rc)
        lateral_disp.append(x_mid - wheel_left_edge if not np.isnan(x_mid) else np.nan)
        if not np.isnan(t3):
            theta_val = np.arctan2(xA - xB, zA - zB)
            theta_deg = np.degrees(theta_val)
            theta.append(theta_deg)
            phi_rad = 0
            theta_rad = np.radians(theta_deg)
            denom = np.sqrt(np.cos(phi_rad)**2 + np.cos(theta_rad)**2 * np.sin(phi_rad)**2)
            if denom != 0:
                S_a_val = np.sin(theta_rad)
                S_b_val = (-np.cos(phi_rad) * np.sin(theta_rad)) / denom
                R_w_val = max((np.cos(theta_rad) * np.cos(phi_rad)) / denom, 0)
                S_a.append(S_a_val)
                S_b.append(S_b_val)
                R_w.append(R_w_val)
            else:
                S_a.append(np.nan)
                S_b.append(np.nan)
                R_w.append(np.nan)
        else:
            theta.append(np.nan)
            S_a.append(np.nan)
            S_b.append(np.nan)
            R_w.append(np.nan)
    return (np.array(camber), np.array(rc), np.array(theta),
            np.array(S_a), np.array(S_b), np.array(R_w), np.array(lateral_disp))

cam_i, rc_i, theta_i, S_a_i, S_b_i, R_w_i, lat_disp_i = compute_metrics(init, xN_i, xM_i, zN_i, zM_i)
cam_o, rc_o, theta_o, S_a_o, S_b_o, R_w_o, lat_disp_o = compute_metrics(opt, xN_o, xM_o, zN_o, zM_o)

k_wheel_i = [k * (rw**2) if not np.isnan(rw) else np.nan for rw in R_w_i]
k_wheel_o = [k * (rw**2) if not np.isnan(rw) else np.nan for rw in R_w_o]
c_wheel_i = [c_damp * (rw**2) if not np.isnan(rw) else np.nan for rw in R_w_i]
c_wheel_o = [c_damp * (rw**2) if not np.isnan(rw) else np.nan for rw in R_w_o]

# -------------------------------
# Compute and Print Comparison
# -------------------------------
def compute_comparison(camber, rc, lateral_disp):
    camber_variation = np.ptp(camber[~np.isnan(camber)])
    mean_camber = np.nanmean(camber)
    max_lateral_disp = np.nanmax(lateral_disp[~np.isnan(lateral_disp)]) * 1000  # Convert to mm
    rc_variation = np.ptp(rc[~np.isnan(rc)])  # Convert to m
    return camber_variation, mean_camber, max_lateral_disp, rc_variation

cam_var_i, mean_cam_i, max_lat_i, rc_var_i = compute_comparison(cam_i, rc_i, lat_disp_i)
cam_var_o, mean_cam_o, max_lat_o, rc_var_o = compute_comparison(cam_o, rc_o, lat_disp_o)

print(f"Initial Configuration: (a={init['a']:.3f}, c={init['c']:.3f}, d={init['d']:.3f}, b={init['b']:.3f}) m")
print(f"• Initial Camber Variation: {cam_var_i:.2f} degrees")
print(f"• Initial Mean Camber: {mean_cam_i:.2f} degrees")
print(f"• Initial Max Lateral Displacement: {max_lat_i:.0f} mm")
print(f"• Initial Roll Center Height Variation: {rc_var_i:.3f} m")
print(f"Optimized Configuration: (a={opt['a']:.3f}, c={opt['c']:.3f}, d={opt['d']:.3f}, b={opt['b']:.3f}) m")
print(f"• Optimized Camber Variation: {cam_var_o:.2f} degrees")
print(f"• Optimized Mean Camber: {mean_cam_o:.2f} degrees")
print(f"• Optimized Max Lateral Displacement: {max_lat_o:.0f} mm")
print(f"• Optimized Roll Center Height Variation: {rc_var_o:.3f} m")

# -------------------------------
# Plot 1: 2x2 Subplots with Legends
# -------------------------------
X = delta_z_values * 1000  # mm
fig1, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(X, cam_i, 'r--', label='Initial')
axs[0, 0].plot(X, cam_o, 'b-', label='Optimized')
axs[0, 0].set_title('Camber Angle vs Wheel Travel')
axs[0, 0].set_ylabel('Camber (°)')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(X, rc_i * 1000, 'r--', label='Initial')
axs[0, 1].plot(X, rc_o * 1000, 'b-', label='Optimized')
axs[0, 1].set_title('Roll Center Height vs Wheel Travel')
axs[0, 1].set_ylabel('RC Height (mm)')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].plot(X, lat_disp_i * 1000, 'r--', label='Initial')
axs[1, 0].plot(X, lat_disp_o * 1000, 'b-', label='Optimized')
axs[1, 0].set_title('Lateral Wheel Displacement vs Wheel Travel')
axs[1, 0].set_ylabel('Lateral Disp (mm)')
axs[1, 0].set_xlabel('Wheel Travel (mm)')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(X, R_w_i, 'r--', label='Rw Initial')
axs[1, 1].plot(X, R_w_o, 'b-', label='Rw Optimized')
axs[1, 1].plot(X, S_a_i, 'r:', label='Sa Initial')
axs[1, 1].plot(X, S_a_o, 'b:', label='Sa Optimized')
axs[1, 1].plot(X, S_b_i, 'r-.', label='Sb Initial')
axs[1, 1].plot(X, S_b_o, 'b-.', label='Sb Optimized')
axs[1, 1].set_title('Motion Ratios vs Wheel Travel')
axs[1, 1].set_ylabel('Ratio')
axs[1, 1].set_xlabel('Wheel Travel (mm)')
axs[1, 1].legend()
axs[1, 1].grid(True)

for ax in axs.flat:
    ax.set_xlim([-75, 75])

plt.tight_layout()

# -------------------------------
# Plot 2: Spring and Damping Rate
# -------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(X, k_wheel_i, 'r--', label='Spring Initial')
ax2.plot(X, k_wheel_o, 'b-', label='Spring Optimized')
ax2.plot(X, c_wheel_i, 'r-.', label='Damping Initial')
ax2.plot(X, c_wheel_o, 'b-.', label='Damping Optimized')
ax2.set_title('Spring and Damping Rate vs Wheel Travel')
ax2.set_ylabel('Rate (N/m or Ns/m)')
ax2.set_xlabel('Wheel Travel (mm)')
ax2.legend()
ax2.grid(True)
ax2.set_xlim([-75, 75])

plt.tight_layout()
plt.show()