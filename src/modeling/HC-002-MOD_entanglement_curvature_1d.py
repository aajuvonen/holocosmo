# HC-002-MOD: Entanglement curvature 1D
# Created on 2025-04-14T19:35:45.004753
"""
Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
This toy model visualizes how curvature and directional "forces" may arise from
entanglement structure. Hessian eigenvalues reflect local curvature and serve as
a proxy for emergent geometric responses.

The scipt demonstrates curvature extraction from a synthetic entropy field S(x, y)
by constructing the Hessian matrix at each point and analyzing its eigenvalues.
This provides directional information about curvature structure.

Inputs:
--------
No input files — the entropy field is defined analytically.

Outputs:
--------
- Contour plots of:
  • Entropy field S(x, y)
  • Entanglement curvature tensor components (E_xx, E_xy, E_yy)
  • Principal and secondary curvature eigenvalue fields

Paper Reference:
-----------------
- HC-002-DOC: A Toy Model Indicating Inverse-Square-Law Emergence from Quantum Entanglement
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Grid Setup ---
N = 100
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# --- Define Synthetic Entropy Field ---
S = np.exp(- (X**2 + Y**2))  # centered Gaussian bump

# --- Compute Gradients & Hessian ---
S_x = np.gradient(S, dx, axis=1)
S_y = np.gradient(S, dy, axis=0)
S_xx = np.gradient(S_x, dx, axis=1)
S_yy = np.gradient(S_y, dy, axis=0)
S_xy = np.gradient(S_x, dy, axis=0)
S_yx = np.gradient(S_y, dx, axis=1)

# --- Curvature Tensor Components ---
E_xx = S_xx
E_yy = S_yy
E_xy = 0.5 * (S_xy + S_yx)  # symmetrized off-diagonal

# --- Compute Eigenvalues of Hessian (Curvature Tensor) ---
eig1 = np.zeros_like(S)
eig2 = np.zeros_like(S)

for i in range(N):
    for j in range(N):
        H = np.array([[E_xx[i, j], E_xy[i, j]],
                      [E_xy[i, j], E_yy[i, j]]])
        vals, _ = np.linalg.eig(H)
        eig1[i, j] = np.max(vals)
        eig2[i, j] = np.min(vals)

# --- Plot: Entropy Field ---
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, S, levels=50, cmap='viridis')
plt.colorbar(label='S(x, y)')
plt.title('Synthetic Entropy Field')
plt.xlabel('x'); plt.ylabel('y')
plt.show()

# --- Plot: Curvature Components ---
for label, field in zip(["E_xx", "E_xy", "E_yy"], [E_xx, E_xy, E_yy]):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, field, levels=50, cmap='viridis')
    plt.colorbar(label=label)
    plt.title(f'Entanglement Curvature Component {label}')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

# --- Plot: Principal Curvatures ---
for label, field in zip(["Principal", "Secondary"], [eig1, eig2]):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, field, levels=50, cmap='viridis')
    plt.colorbar(label=f'{label} Eigenvalue')
    plt.title(f'{label} Eigenvalue of Curvature Tensor')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

