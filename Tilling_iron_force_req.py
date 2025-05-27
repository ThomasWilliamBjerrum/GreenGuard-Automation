# -*- coding: utf-8 -*-
"""
Spyder Editor

Greenguard Automation
Author: Thomas Bjerrum

This script seeks to estimate optimal tilling dimentional parameters utilizing
the universal earthmoving equation.  
"""
# importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Defining variabels

gamma = 1400 # [kg/m^3] specific density
g = 9.82 # [m/s^2] gravitational acc.
d = 0.035 # [m] tillage depth (1.5 inch is preferable)
C = 13 # [Pa] soil cohesive sthrength (Red Brown Earth exp. value)
q = 0 # [Pa] surchage pressure (zero for this work)
C_a = 0 # [Pa] soil adhesive strength (assumed zero)

# tool dimensions
w = 0.200 # [m] tool width  
alpha = 25 # [degrees] rake angle
delta = 70 # [degrees] soil to metal friction angle: Assumed 2/3 soil friction 
# angle Phi.(Red Brown Earth exp. value)


N_gamma = 1 # soil density factor (1 unless working with unusual soils)
N_c = 1.5 # soil cohesive strength factor (normally in range: [0.5;1.5])
N_q = 1 # surcharge factor (usually 1, but in the case of additional pressure, it may increase.)
N_ca = 1 # soil adhesion factor (Can vary from 0 (no adhesion) to 1.5 (for sticky wet soils).)

# Analysis

# convert units
alpha_rad = np.radians(alpha)
delta_rad = np.radians(delta)

# force 
P = (gamma*g**d**2*N_gamma + C*d*N_c + q*d*N_q + C_a*d*N_ca)*w

# draft force
H = P*np.sin(alpha + delta_rad) + C_a*d*w*(1/np.tan(alpha_rad))

# vertical down force
V = P*np.cos(alpha + delta_rad) - C_a*d*w

# resulting force
F_R = np.sqrt(V**2 + H**2)



# Print Output
print(f"Draft Force (H): {H:.2f} N")
print(f"Vertical Down Force (V): {V:.2f} N")
print(f"Resulting Force (F_R): {F_R:.2f} N")



