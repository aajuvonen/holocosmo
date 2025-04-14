# HC-001-MOD: Dynamic holographic dark energy
# Created on 2025-04-14T17:26:52.029854
"""
Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
Simulates the cosmological evolution of a holographic universe model
in which the vacuum energy density is dynamically determined by the
inverse area of a causal horizon.

Implements a simple ODE model capturing:
  - Horizon size evolution R(t)
  - Matter energy density rho_m(t)
  - Scale factor a(t)
  - Derived quantities: effective vacuum energy, w_eff, Hubble rate

Outputs:
---------
- Timestamped CSV: data/processed/YYYYMMDD-HHMM_holographic_output.csv
- Timestamped Figures: 5 PNG plots in data/figures/

Scientific Reference:
----------------------
"A Unified Holographic Solution to the Vacuum Catastrophe and a Dynamic Model of Dark Energy" (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from datetime import datetime
import os

# -------------------------------------------------------
# Timestamp for file naming
# -------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d-%H%M_")

# -------------------------------------------------------
# ODE System: dR/dt, d(rho_m)/dt, da/dt
# -------------------------------------------------------
def holographic_odes(t, y):
    R, rho_m, a = y
    R = max(R, 1e-10)
    rho_eff = 3.0 / (R * R)
    H = np.sqrt(max((rho_m + rho_eff) / 3.0, 0.0))
    dR = H * R - 1.0
    drho_m = -3.0 * H * rho_m
    da = H * a
    return [dR, drho_m, da]

# -------------------------------------------------------
# Initial Conditions
# -------------------------------------------------------
R0 = 50.0
rho_m0 = 1e-2
a0 = 1.0
y0 = [R0, rho_m0, a0]

t_span = (0.0, 1000.0)
t_eval = np.linspace(*t_span, 2000)

# -------------------------------------------------------
# Integrate System
# -------------------------------------------------------
sol = solve_ivp(
    holographic_odes, t_span, y0, t_eval=t_eval,
    rtol=1e-10, atol=1e-12
)

# -------------------------------------------------------
# Extract Solutions and Derived Quantities
# -------------------------------------------------------
t = sol.t
R = sol.y[0]
rho_m_raw = sol.y[1]
a_raw = sol.y[2]
rho_m = np.maximum(rho_m_raw, 1e-40)

rho_eff = 3.0 / (R * R)
H = np.sqrt((rho_m + rho_eff) / 3.0)
drho_eff_dt = np.gradient(rho_eff, t)
p_eff = -rho_eff - drho_eff_dt / (3.0 * H)
w_eff = p_eff / rho_eff
density_ratio = rho_m / rho_eff

# -------------------------------------------------------
# Normalize to Redshift
# -------------------------------------------------------
a_today = a_raw[-1]
a = a_raw / a_today
z = 1.0 / a - 1.0
z = np.clip(z, 0, 10)

# ΛCDM comparison (same H0 for visual normalization)
Omega_m0 = 0.3
Omega_Lambda0 = 0.7
H0 = H[-1]
H_LCDM = H0 * np.sqrt(Omega_m0 * (1 + z)**3 + Omega_Lambda0)

# -------------------------------------------------------
# Save CSV Output
# -------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
output_csv = f"data/processed/{timestamp}holographic_output.csv"
df = pd.DataFrame({
    't': t,
    'a': a,
    'z': z,
    'R': R,
    'H': H,
    'rho_m': rho_m,
    'rho_eff': rho_eff,
    'w_eff': w_eff,
    'rho_m_over_rho_eff': density_ratio
})
df.to_csv(output_csv, index=False)
print(f"CSV saved to '{output_csv}'")

# -------------------------------------------------------
# Save Figures
# -------------------------------------------------------
os.makedirs("data/figures", exist_ok=True)

# 1. Hubble Parameter vs Redshift
plt.figure()
plt.plot(z, H, label='Holographic H(z)')
plt.plot(z, H_LCDM, '--', label='ΛCDM H(z)')
plt.xlabel("Redshift z")
plt.ylabel("Hubble parameter H")
plt.title("Hubble Parameter: Holographic vs ΛCDM")
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"data/figures/{timestamp}holographic_hubble_vs_lcdm.svg")

# 2. Effective Equation of State w_eff(t)
plt.figure()
plt.plot(t, w_eff)
plt.axhline(-1, color='gray', linestyle='--')
plt.xlabel("Time")
plt.ylabel("Effective w(t)")
plt.title("Equation of State w_eff(t)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"data/figures/{timestamp}holographic_weff_vs_time.svg")

# 3. Energy Densities: Matter and Vacuum
plt.figure()
plt.plot(t, rho_m, label="ρ_m (matter)")
plt.plot(t, rho_eff, label="ρ_eff (vacuum)")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Energy Densities")
plt.title("Matter vs Vacuum Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"data/figures/{timestamp}holographic_energy_components.svg")

# 4. Matter-to-Vacuum Ratio
plt.figure()
plt.plot(t, density_ratio)
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("ρ_m / ρ_eff")
plt.title("Matter-to-Vacuum Energy Ratio Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"data/figures/{timestamp}holographic_matter_to_vacuum_ratio.svg")

# 5. w_eff vs Redshift
plt.figure()
plt.plot(z, w_eff)
plt.xlabel("Redshift z")
plt.ylabel("w_eff")
plt.title("w_eff vs Redshift")
plt.gca().invert_xaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"data/figures/{timestamp}holographic_weff_vs_z.svg")

print("All figures saved to data/figures/")