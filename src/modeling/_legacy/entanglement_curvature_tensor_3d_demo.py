"""
entanglement_curvature_tensor_3d_demo.py

Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
Simulates a 2x2x2 lattice of spins in a 3D transverse-field Ising model,
introducing a slight impurity and calculating:
- Local entanglement entropy at each site
- Discrete Laplacian of the entropy (a proxy for entanglement curvature)

Inspired by:
"Toy Model in Three Dimensions Indicating Inverse-Square-Law Emergence from Quantum Entanglement"
→ see: doc/papers/toy_model_3d_inverse_square_entanglement.pdf

Inputs:
--------
- None (all fields constructed synthetically)

Outputs:
---------
- Printed values of entropy and Laplacian
- Slice plots of:
  • Entropy field S(x, y) at z = 0
  • Laplacian ΔS(x, y) at z = 0

Scientific Context:
-------------------
This model illustrates how curvature-like structure may emerge in entanglement
patterns on small spin lattices. The Laplacian of entanglement entropy plays the
role of an effective scalar curvature related to information gradients.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# --- Helper functions ---
def kron_N(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def pauli_operator(N, site, op):
    return kron_N([op if i == site else I2 for i in range(N)])

# --- Lattice setup ---
Lx, Ly, Lz = 2, 2, 2
N = Lx * Ly * Lz  # 8 spins

def lattice_index(i, j, k):
    return i + Lx * (j + Ly * k)

def neighbors(i, j, k):
    neigh = []
    for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < Lx and 0 <= nj < Ly and 0 <= nk < Lz:
            neigh.append((ni, nj, nk))
    return neigh

def build_3D_hamiltonian(J=1.0, h0=1.0, impurity_site=(0,0,0), h_imp=1.2):
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    # Interactions
    for k in range(Lz):
        for j in range(Ly):
            for i in range(Lx):
                idx = lattice_index(i, j, k)
                for (ni, nj, nk) in neighbors(i, j, k):
                    if (i, j, k) < (ni, nj, nk):  # avoid double-counting
                        idx_neighbor = lattice_index(ni, nj, nk)
                        H -= J * (pauli_operator(N, idx, sz) @ pauli_operator(N, idx_neighbor, sz))

    # Transverse field
    for k in range(Lz):
        for j in range(Ly):
            for i in range(Lx):
                idx = lattice_index(i, j, k)
                h_here = h_imp if (i, j, k) == impurity_site else h0
                H -= h_here * pauli_operator(N, idx, sx)

    return H

# --- Hamiltonian and ground state ---
J, h0, h_imp = 1.0, 1.0, 1.2
H = build_3D_hamiltonian(J, h0, impurity_site=(0,0,0), h_imp=h_imp)
eigs, vecs = la.eigh(H)
ground_state = vecs[:, 0]

# --- Entropy field ---
def single_site_entropy(psi, site, dims):
    psi_tensor = psi.reshape(dims)
    axes_to_trace = tuple(i for i in range(len(dims)) if i != site)
    rho = np.tensordot(psi_tensor, psi_tensor.conj(), axes=(axes_to_trace, axes_to_trace))
    rho = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(rho)
    eps = 1e-12
    return -np.sum(evals * np.log(evals + eps)).real

dims = [2] * N
S_field = np.zeros((Lx, Ly, Lz))
for k in range(Lz):
    for j in range(Ly):
        for i in range(Lx):
            idx = lattice_index(i, j, k)
            S_field[i, j, k] = single_site_entropy(ground_state, idx, dims)

print("Local Entanglement Entropy Field S(x,y,z):")
print(S_field)

# --- Discrete Laplacian ---
def discrete_laplacian(field):
    lap = np.zeros_like(field)
    nx, ny, nz = field.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                s_center = field[i, j, k]
                neighbor_sum, count = 0.0, 0
                for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        neighbor_sum += field[ni, nj, nk]
                        count += 1
                lap[i, j, k] = neighbor_sum - count * s_center
    return lap

lap_S = discrete_laplacian(S_field)
print("\nDiscrete Laplacian of S (Curvature Proxy):")
print(lap_S)

# --- Visualization: slice at z=0 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(S_field[:, :, 0], cmap='viridis', origin='lower',
                     extent=(0, Ly-1, 0, Lx-1))
axes[0].set_title("Entropy Field S(x,y) at z=0")
axes[0].set_xlabel("y")
axes[0].set_ylabel("x")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(lap_S[:, :, 0], cmap='plasma', origin='lower',
                     extent=(0, Ly-1, 0, Lx-1))
axes[1].set_title("Laplacian of S(x,y) at z=0")
axes[1].set_xlabel("y")
axes[1].set_ylabel("x")
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
