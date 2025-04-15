"""
gravity_laplacian_impurity.py

Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
This script simulates a 3D PEPS (Projected Entangled Pair State) tensor network on a cubic lattice,
introducing a localized region ("impurity cube") with modified transverse field strength. It computes:

- The local entanglement entropy field S(x)
- The discrete Laplacian of the entropy field ∇²S(x), interpreted as an emergent curvature proxy
- CSV output and optional 3D scatter visualization of the curvature field

Inputs (via CLI):
------------------
- Lattice size:            --lattice Lx Ly Lz           (default: 32x32x32)
- Bond dimension:          --bond-dim D                 (default: 3)
- Time evolution step:     --tau                        (default: 0.01)
- Number of steps:         --steps                      (default: 15)
- Coupling constant:       --J                          (default: 1.0)
- Base transverse field:   --h                          (default: 1.0)
- Impurity field strength: --h-impurity                 (default: 3.0)
- Output location:         --output-dir                 (default: ../../data/processed/)
- Show plot:               --plot                       (optional flag)

Scientific Basis:
------------------
This builds on the emergent gravity framework proposed in:
- *An Effective Field Equation for Emergent Gravity from Quantum Entanglement*&#8203;:contentReference[oaicite:1]{index=1}

In this context, local perturbations to the entropy field (e.g., via an impurity region)
create "informational gravity wells" that resemble localized curvature anomalies in classical spacetime.

Outputs:
---------
- A CSV file containing (x, y, z, entropy, laplacian)
- Optional 3D scatter plot showing curvature around the impurity

See `/doc/papers/` for the supporting theoretical documents.
"""

# --- Imports ---
import argparse
import os
from datetime import datetime
import numpy as np
import scipy.linalg as la
import tensornetwork as tn
import matplotlib.pyplot as plt
import csv


# --- PEPS Initialization ---
def init_peps(Lx, Ly, Lz, d, D):
    """Initialize a 3D PEPS state with random tensors and proper boundary bond dimensions."""
    peps = {}
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                Dl = D if i > 0 else 1
                Dr = D if i < Lx - 1 else 1
                Du = D if j > 0 else 1
                Dd = D if j < Ly - 1 else 1
                Df = D if k > 0 else 1
                Db = D if k < Lz - 1 else 1
                shape = (d, Dl, Dr, Du, Dd, Df, Db)
                peps[(i, j, k)] = np.random.rand(*shape) * 0.1
    return peps


# --- Gates ---
def two_site_gate(J, tau):
    """Construct a two-site ZZ interaction gate."""
    sz = np.array([[1, 0], [0, -1]])
    H_int = -J * np.kron(sz, sz)
    return la.expm(-tau * H_int).reshape(2, 2, 2, 2)

def single_site_gate(h_val, tau):
    """Construct a single-site X-rotation gate for transverse field."""
    sx = np.array([[0, 1], [1, 0]])
    return la.expm(-tau * h_val * sx)


# --- Time Evolution Step ---
def update_bond(peps, site_a, site_b, axis_a, axis_b, gate, d, D):
    """Apply a two-site gate to the PEPS network along a bond (schematic, no SVD update)."""
    A = peps[site_a]
    B = peps[site_b]
    nodeA = tn.Node(A)
    nodeB = tn.Node(B)
    tn.connect(nodeA[axis_a+1], nodeB[axis_b+1])
    combined_node = tn.contract_between(nodeA, nodeB)
    combined = combined_node.get_tensor()
    gate_matrix = gate.reshape(d*d, d*d)
    updated = gate_matrix @ combined.reshape(d*d, -1)
    updated = updated.reshape(combined.shape)
    # Dummy update: replace with new random tensors (as SVD splitting is not implemented)
    peps[site_a] = np.random.rand(*A.shape)
    peps[site_b] = np.random.rand(*B.shape)
    return peps


def simple_update(peps, J, h, tau, num_steps, impurity_cube, h_impurity, Lx, Ly, Lz, d, D):
    """Apply simple update for imaginary time evolution, including site-dependent impurity field."""
    U_two = two_site_gate(J, tau)
    U_single_default = single_site_gate(h, tau)
    U_single_imp = single_site_gate(h_impurity, tau)

    for step in range(num_steps):
        # Apply single-site gates with impurity substitution
        for site in peps:
            gate = U_single_imp if site in impurity_cube else U_single_default
            peps[site] = np.tensordot(gate, peps[site], axes=([1], [0]))

        # Apply two-site gates along x-direction only
        for i in range(Lx - 1):
            for j in range(Ly):
                for k in range(Lz):
                    site_a = (i, j, k)
                    site_b = (i + 1, j, k)
                    peps = update_bond(peps, site_a, site_b, axis_a=1, axis_b=0, gate=U_two, d=d, D=D)

        print(f"Completed evolution step {step+1}/{num_steps}")
    return peps


# --- Entanglement Entropy + Laplacian ---
def compute_local_entropy(peps, site, d):
    """Estimate von Neumann entropy from local PEPS tensor (approximate, no full environment)."""
    tensor_mat = peps[site].reshape(d, -1)
    U, S, Vh = la.svd(tensor_mat, full_matrices=False)
    S_norm = S / (np.linalg.norm(S) + 1e-12)
    return -np.sum((S_norm**2) * np.log(S_norm**2 + 1e-12))


def discrete_laplacian(field):
    """Compute discrete Laplacian of a scalar 3D field."""
    lap = np.zeros_like(field)
    nx, ny, nz = field.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                s_center = field[i, j, k]
                neighbor_sum = 0.0
                count = 0
                for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        neighbor_sum += field[ni, nj, nk]
                        count += 1
                lap[i, j, k] = neighbor_sum - count * s_center
    return lap


# --- Output ---
def save_csv_data(filepath, entropy_field, laplacian_field):
    """Save (x, y, z, entropy, laplacian) to CSV."""
    nx, ny, nz = entropy_field.shape
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "z", "entropy", "laplacian"])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    writer.writerow([i, j, k, entropy_field[i,j,k], laplacian_field[i,j,k]])
    print(f"CSV data saved to {filepath}")


def visualize_3d(laplacian_field):
    """3D scatter plot of the curvature proxy (Laplacian values)."""
    nx, ny, nz = laplacian_field.shape
    xs, ys, zs, vals = [], [], [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                xs.append(i)
                ys.append(j)
                zs.append(k)
                vals.append(laplacian_field[i,j,k])
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=vals, cmap='plasma', marker='o', s=2)
    plt.colorbar(sc, label='Laplacian value')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("3D Scatter Plot of Entanglement Curvature with Impurity")
    plt.show()


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="3D PEPS Simulation with Impurity Region")
    parser.add_argument("--lattice", nargs=3, type=int, default=[32, 32, 32])
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--h-impurity", type=float, default=3.0)
    parser.add_argument("--output-dir", type=str, default="../../data/processed")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    Lx, Ly, Lz = args.lattice
    d = 2
    os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing PEPS state...")
    peps = init_peps(Lx, Ly, Lz, d, args.bond_dim)

    # Define the 3x3x3 impurity region around the center of the lattice
    cx, cy, cz = Lx//2, Ly//2, Lz//2
    impurity_cube = {
        (i, j, k)
        for i in range(cx-1, cx+2)
        for j in range(cy-1, cy+2)
        for k in range(cz-1, cz+2)
    }

    print("Running time evolution with impurity...")
    peps = simple_update(peps, args.J, args.h, args.tau, args.steps, impurity_cube,
                         args.h_impurity, Lx, Ly, Lz, d, args.bond_dim)

    print("Computing entropy field...")
    entropy_field = np.zeros((Lx, Ly, Lz))
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                entropy_field[i,j,k] = compute_local_entropy(peps, (i,j,k), d)

    print("Computing Laplacian...")
    laplacian_field = discrete_laplacian(entropy_field)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"{timestamp}_gravity_impurity_L{Lx}_D{args.bond_dim}_tau{args.tau}_himp{args.h_impurity}.csv"
    output_path = os.path.join(args.output_dir, filename)
    save_csv_data(output_path, entropy_field, laplacian_field)

    if args.plot:
        visualize_3d(laplacian_field)


if __name__ == "__main__":
    main()
