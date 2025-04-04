import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

def holographic_odes(t, y):
    R, rho_m, a = y
    R = max(R, 1e-10)
    rho_eff = 3.0 / (R * R)
    H = np.sqrt(max((rho_m + rho_eff) / 3.0, 0.0))
    dR = H * R - 1.0
    drho_m = -3.0 * H * rho_m
    da = H * a
    return [dR, drho_m, da]

# --------------------------
# Initial conditions
# --------------------------
R0 = 50.0
rho_m0 = 1e-2
a0 = 1.0
y0 = [R0, rho_m0, a0]

t_span = (0.0, 1000.0)
t_eval = np.linspace(*t_span, 2000)

# --------------------------
# Integration
# --------------------------
sol = solve_ivp(
    holographic_odes, t_span, y0, t_eval=t_eval,
    rtol=1e-10, atol=1e-12
)

# Extract variables
t = sol.t
R = sol.y[0]
rho_m_raw = sol.y[1]
a_raw = sol.y[2]
rho_m = np.maximum(rho_m_raw, 1e-40)

# Derived quantities
rho_eff = 3.0 / (R * R)
H = np.sqrt((rho_m + rho_eff) / 3.0)
drho_eff_dt = np.gradient(rho_eff, t)
p_eff = -rho_eff - drho_eff_dt / (3.0 * H)
w_eff = p_eff / rho_eff
density_ratio = rho_m / rho_eff

# --------------------------
# Redshift normalization
# --------------------------
a_today = a_raw[-1]             # Final value of a(t)
a = a_raw / a_today             # Normalize so that a_final = 1
z = 1.0 / a - 1.0
z = np.clip(z, 0, 10)           # Avoid negative redshift

# --------------------------
# ΛCDM comparison
# --------------------------
Omega_m0 = 0.3
Omega_Lambda0 = 0.7
H0 = H[-1]  # Normalize to match "today"
H_LCDM = H0 * np.sqrt(Omega_m0 * (1 + z)**3 + Omega_Lambda0)

# --------------------------
# CSV Export
# --------------------------
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
df.to_csv("holographic_output.csv", index=False)
print("✅ Output saved to 'holographic_output.csv'.")

# --------------------------
# Plots
# --------------------------
plt.figure()
plt.plot(z, H, label='Holographic H(z)')
plt.plot(z, H_LCDM, '--', label='ΛCDM H(z)')
plt.xlabel("Redshift z")
plt.ylabel("Hubble parameter H")
plt.title("Hubble Parameter: Holographic vs ΛCDM")
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(t, w_eff)
plt.axhline(-1, color='gray', linestyle='--')
plt.xlabel("Time")
plt.ylabel("Effective w(t)")
plt.title("Equation of State w_eff(t)")
plt.grid(True)

plt.figure()
plt.plot(t, rho_m, label="ρ_m (matter)")
plt.plot(t, rho_eff, label="ρ_eff (vacuum)")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Energy Densities")
plt.title("Matter vs Vacuum Energy")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t, density_ratio)
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("ρ_m / ρ_eff")
plt.title("Matter-to-Vacuum Energy Ratio Over Time")
plt.grid(True)

plt.figure()
plt.plot(z, w_eff)
plt.xlabel("Redshift z")
plt.ylabel("w_eff")
plt.title("w_eff vs Redshift")
plt.gca().invert_xaxis()
plt.grid(True)

plt.show()