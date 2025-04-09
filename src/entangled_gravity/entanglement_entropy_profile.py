import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1) Entanglement entropy profile S(r)
# ---------------------------------------------------
def S(r, S0=1.0, r0=1.0, alpha=1.0):
    return S0 * (1 + r / r0) ** (-alpha)

# ---------------------------------------------------
# 2) Effective potential: Phi(r) = -kappa * S(r)
# ---------------------------------------------------
def Phi(r, kappa=1.0, S0=1.0, r0=1.0, alpha=1.0):
    return -kappa * S(r, S0, r0, alpha)

# ---------------------------------------------------
# 3) Rotation velocity: v(r) = sqrt( r * dPhi/dr )
# ---------------------------------------------------
def v_rot(r, kappa=1.0, S0=1.0, r0=1.0, alpha=1.0):
    dS_dr = -alpha * S0 / r0 * (1.0 + r / r0) ** (-(alpha + 1))
    dPhi_dr = -kappa * dS_dr
    val = r * dPhi_dr
    return np.sqrt(np.maximum(val, 0.0))

# ---------------------------------------------------
# 4) Demo: Plotting S(r), v(r), and residuals
# ---------------------------------------------------
if __name__ == "__main__":
    # Radial array
    r_vals = np.linspace(0.1, 50, 500)

    # Model parameters
    S0 = 1.0
    r0 = 5.0
    alpha = 1.0
    kappa = 5.0

    # Generate profiles
    s_vals = S(r_vals, S0, r0, alpha)
    v_model = v_rot(r_vals, kappa, S0, r0, alpha)

    # For demonstration: make up some synthetic observations
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 2, size=len(v_model))
    v_obs = v_model + noise
    v_err = np.full_like(v_obs, 2.0)  # constant uncertainty

    # Residuals
    residuals = v_obs - v_model
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"RMSE of fit: {rmse:.2f} km/s")

    # --- Plot 1: Entropy ---
    plt.figure(figsize=(6, 5))
    plt.plot(r_vals, s_vals, label='Entropy S(r)')
    plt.yscale('log')
    plt.xlabel('Radius r')
    plt.ylabel('Entropy S(r) [log scale]')
    plt.title('Entanglement Entropy Profile')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Rotation + Residuals ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Top: velocity
    ax1.errorbar(r_vals, v_obs, yerr=v_err, fmt='o', label='Observed $V_{obs}$', color='black', markersize=3)
    ax1.plot(r_vals, v_model, label='Model $v(r)$', color='red')
    ax1.set_ylabel("Velocity [km/s]")
    ax1.set_title(f'Rotation Curve and Residuals\n(kappa={kappa}, r0={r0}, alpha={alpha})')
    ax1.legend()
    ax1.grid(True)

    # Bottom: residuals
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.errorbar(r_vals, residuals, yerr=v_err, fmt='o', color='darkred', markersize=3)
    ax2.set_xlabel("Radius [kpc]")
    ax2.set_ylabel("Residual")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
