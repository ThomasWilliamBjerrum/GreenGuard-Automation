import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# === Suspension Parameters ===
# (Using the same parameters as in the original code)
m_sprung = 15  # kg (sprung mass)
m_unsprung = 10  # kg (unsprung mass)
k_spring = 1332  # N/m (spring stiffness)
c_damper = 170   # Ns/m (damping coefficient)
k_tire = 50000   # N/m (tire stiffness)

# Motion ratio (R_w) at optimal position (10% of upper arm, as set in the code)
R_w_optimal = 0.65  # Approximate value based on earlier computation (avg_R_w at 10%)
k_eff = k_spring * R_w_optimal**2  # Effective spring constant
c_eff = c_damper * R_w_optimal**2  # Effective damping coefficient

# === 2DoF Dynamic Model Setup ===
# Mass matrix
M = np.array([[m_sprung, 0], [0, m_unsprung]])

# Stiffness matrix with tire stiffness
K = np.array([[k_eff, -k_eff], [-k_eff, k_eff + k_tire]])

# Damping matrix
C = np.array([[c_eff, -c_eff], [-c_eff, c_eff]])

# State-space representation: A = [[0, I], [-M^-1*K, -M^-1*C]]
M_inv = np.linalg.inv(M)
A = np.block([[np.zeros((2, 2)), np.eye(2)], [-M_inv @ K, -M_inv @ C]])

# Input matrix B (force on unsprung mass from terrain)
B = np.block([[np.zeros((2, 1))], [M_inv @ np.array([[0], [k_tire]])]])

# Eigenvalue Analysis (to get natural frequencies for plotting)
eigvals, eigvecs = np.linalg.eig(A)
omega_n = np.abs(eigvals.imag[::2])  # Imaginary parts for oscillatory modes
valid_omega_n = omega_n[omega_n > 1e-6]
valid_eigvals_real = -eigvals.real[::2][omega_n > 1e-6]
zeta = [r / w if w > 1e-6 else 0 for r, w in zip(valid_eigvals_real, valid_omega_n)]
f_n = valid_omega_n / (2 * np.pi)  # Natural frequencies in Hz

# === Frequency Response for Both Transfer Functions ===
# Frequency range in rad/s
w = np.logspace(-1, 2, 500)

# Define SISO system for z_s (sprung mass displacement)
C_siso_zs = np.array([[1, 0, 0, 0]])  # Observe z_s
D_siso = np.zeros((1, 1))
sys_zs = scipy.signal.StateSpace(A, B, C_siso_zs, D_siso)
w, H_zs = scipy.signal.freqresp(sys_zs, w)

# Define SISO system for z_u (unsprung mass displacement)
C_siso_zu = np.array([[0, 1, 0, 0]])  # Observe z_u
sys_zu = scipy.signal.StateSpace(A, B, C_siso_zu, D_siso)
w, H_zu = scipy.signal.freqresp(sys_zu, w)

# Magnitude for z_s and z_u (in dB)
mag_zs = 20 * np.log10(np.abs(H_zs))
mag_zu = 20 * np.log10(np.abs(H_zu))

# Phase for z_s and z_u (in degrees)
phase_zs = np.angle(H_zs, deg=True)
phase_zu = np.angle(H_zu, deg=True)

# === Plotting ===
fig_bode, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Magnitude plot
ax1.semilogx(w / (2 * np.pi), mag_zs, label='Sprung (z_s)', color='blue')
ax1.semilogx(w / (2 * np.pi), mag_zu, label='Unsprung (z_u)', color='green')
ax1.axvline(f_n[0], color='red', linestyle='--', label=f'f_n1 = {f_n[0]:.2f} Hz')
ax1.axvline(f_n[1], color='purple', linestyle='--', label=f'f_n2 = {f_n[1]:.2f} Hz')
ax1.set_ylabel('Magnitude (dB)')
ax1.grid(True, which='both')
ax1.legend()
ax1.set_title('Bode Plot: Terrain to Sprung and Unsprung Displacements')

# Phase plot
ax2.semilogx(w / (2 * np.pi), phase_zs, label='Sprung (z_s)', color='blue')
ax2.semilogx(w / (2 * np.pi), phase_zu, label='Unsprung (z_u)', color='green')
ax2.axvline(f_n[0], color='red', linestyle='--')
ax2.axvline(f_n[1], color='purple', linestyle='--')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Phase (deg)')
ax2.grid(True, which='both')
ax2.legend()

plt.tight_layout()
plt.show()