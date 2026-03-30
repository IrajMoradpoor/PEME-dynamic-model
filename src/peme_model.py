"""
Dynamic PEM Electrolyzer Model
Author: Iraj Moradpoor

Modeling PEM electrolyzer performance under variable load
at optimal operating temperature (80°C).
"""

import numpy as np
import pandas as pd

# Constants
F = 96485.3
R = 8.314
n = 2
M_H2 = 2

A_cell = 200      # cm2
N_cell = 50

alpha_an = 0.5
alpha_ca = 0.5

lambda_h = 22
i_L = 6.0
i_rated = 2.5

p_H2 = 30
p_O2 = 2
a_H2O = 1

t_el = 0.1
rho_el = 7.5e-3
t_mem = 322e-4

deltaG_rev = 237200
k = 2.16e6
E_act = 76000

T = 353.15  # 80°C

# Current density range
i_array = np.linspace(0.1, 2.5, 50)

# Model Functions
def U_ocv():
    q = (p_H2 * np.sqrt(p_O2)) / a_H2O
    return (deltaG_rev/(n*F)) + (R*T/(n*F))*np.log(q)

def U_act(i):
    i0_an = k*np.exp(-E_act/(R*T))
    return (R*T/F)*np.arcsinh(i/(2*i0_an))

def sigma_mem():
    return (0.005139*lambda_h - 0.00326)*np.exp(1268*(1/303 - 1/T))

def U_ohmic(i):
    R_el = t_el*rho_el/A_cell
    R_mem = t_mem/(sigma_mem()*A_cell)
    return (2*R_el + R_mem)*i*A_cell

def U_conc(i):
    return (R*T/(alpha_an*n*F))*np.log(i_L/(i_L - i))

def U_cell(i):
    return U_ocv() + U_act(i) + U_ohmic(i) + U_conc(i)

def H2_rate(i):
    return N_cell*i*A_cell*M_H2/(n*F)*1e-3

def spc_kWh_per_kg(i):
    H2_kg_s = H2_rate(i)
    P = N_cell*U_cell(i)*i*A_cell
    return (P/H2_kg_s)/36e5

def Load_fraction(i):
    return i/i_rated

# Simulation
results = []

for i in i_array:
    results.append([
        Load_fraction(i),
        spc_kWh_per_kg(i),
        U_cell(i),
        i
    ])

df = pd.DataFrame(results, columns=[
    "Load Fraction",
    "SPC (kWh/kgH2)",
    "Cell Voltage (V)",
    "Current Density (A/cm2)"
])
