import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1) Entanglement entropy profile S(r)
# ---------------------------------------------------
def S(r, S0=1.0, r0=1.0, alpha=1.0):
    """
    Entanglement entropy profile, decaying more slowly than typical mass density
    if alpha < 2 (standard mass distributions might have alpha >= 2).
    """
    return S0 * (1 + r / r0) ** (-alpha)

# ---------------------------------------------------
# 2) Effective potential: Phi(r) = kappa * S(r)
# ---------------------------------------------------
def Phi(r, kappa=1.0, S0=1.0, r0=1.0, alpha=1.0):
    # Make the potential negative, akin to standard gravity
    return -kappa * S(r, S0, r0, alpha)

# ---------------------------------------------------
# 3) Rotation velocity: v(r) = sqrt( r * dPhi/dr )
# ---------------------------------------------------
def v_rot(r, kappa=1.0, S0=1.0, r0=1.0, alpha=1.0):
    # derivative of Phi is also negative * negative => positive
    dS_dr = -alpha * S0 / r0 * (1.0 + r / r0) ** (-(alpha + 1))
    dPhi_dr = -kappa * dS_dr   # extra minus sign
    val = r * dPhi_dr
    return np.sqrt(np.maximum(val, 0.0))

# ---------------------------------------------------
# 4) Demo: Plotting both S(r) and v(r)
# ---------------------------------------------------
if __name__ == "__main__":
    # Set up a radial array
    r_vals = np.linspace(0.1, 50, 500)

    # Example parameters
    S0 = 1.0
    r0 = 5.0
    alpha = 1.0
    kappa = 5.0

    # Compute S(r) and v(r)
    s_vals = S(r_vals, S0, r0, alpha)
    v_vals = v_rot(r_vals, kappa, S0, r0, alpha)

    # --- Plot Entropy ---
    plt.figure(figsize=(6,5))
    plt.plot(r_vals, s_vals, label='Entropy S(r)')
    plt.yscale('log')  # Only valid if s_vals > 0
    plt.xlabel('Radius r')
    plt.ylabel('Entropy S(r) [log scale]')
    plt.title('Entanglement Entropy Profile')
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Plot Rotation Velocity ---
    plt.figure(figsize=(6,5))
    plt.plot(r_vals, v_vals, 'r-', label='v(r) from Entanglement Potential')
    plt.xlabel('Radius r')
    plt.ylabel('Rotational Velocity v(r)')
    plt.title('Rotation Curve from Entanglement Entropy Gradient')
    plt.grid(True)
    plt.legend()
    plt.show()