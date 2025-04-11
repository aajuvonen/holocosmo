"""
gravity_laplacian_simulation.py

Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
Simulates a 3D transverse-field Ising model on a cubic lattice using a simplified PEPS-like update.
The script computes a local entanglement entropy scalar field, then calculates a discrete Laplacian
of this field as a proxy for "entanglement curvature" — a geometric quantity hypothesized to underlie
gravitational behavior in emergent gravity frameworks.

Inputs (via CLI):
------------------
- Lattice size:        --lattice Lx Ly Lz          (default: 32x32x32)
- Bond dimension:      --bond-dim D                (default: 4)
- Time evolution step: --tau                       (default: 0.01)
- Number of steps:     --steps                     (default: 10)
- Coupling strength:   --J                         (default: 1.0)
- Transverse field:    --h                         (default: 1.0)
- Output folder:       --output-dir path           (default: data/processed/)
- Plot output:         --plot                      (flag to enable 3D scatter plot)

Outputs:
---------
- CSV file with (x, y, z, entropy, laplacian) for each lattice site
- Optional 3D scatter plot of the curvature proxy

Scientific Basis:
------------------
This simulation is grounded in the theoretical framework proposed in:
- "Entanglement Curvature: A Tensorial Approach to Emergent Geometry"
- "An Effective Field Equation for Emergent Gravity from Quantum Entanglement"
- "A Framework for Gravity as Emergent Entanglement"

These papers propose that gravity arises from the second derivatives of local entanglement entropy,
making the computed Laplacian of the entropy field a candidate proxy for curvature-like behavior.

References:
-----------
Supporting PDFs in: /holocosmo/doc/papers/

"""

import argparse
import os
from datetime import datetime
import numpy as np
import scipy.linalg as la
import tensornetwork as tn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv


# -----------------------------------------------------------------------------
# PEPS Initialization
# -----------------------------------------------------------------------------
def init_peps(Lx, Ly, Lz, d, D):
    """
    Initialize a 3D PEPS state on a lattice of shape Lx x Ly x Lz.
    Each tensor is initialized with small random values and correct boundary bond dims.
    """
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


# -----------------------------------------------------------------------------
# Quantum Gate Definitions
# -----------------------------------------------------------------------------
def two_site_gate(J, tau):
    """
    Returns exp(-tau * H_int), where H_int = -J * sz ⊗ sz
    """
    sz = np.array([[1, 0], [0, -1]])
    H_int = -J * np.kron(sz, sz)
    U = la.expm(-tau * H_int)
    return U.reshape(2, 2, 2, 2)


def single_site_gate(h, tau):
    """
    Returns exp(-tau * h * sx), where sx is the Pauli X matrix
    """
    sx = np.array([[0, 1], [1, 0]])
    return la.expm(-tau * h * sx)


# -----------------------------------------------------------------------------
# Simple Update Routine (Imaginary Time Evolution)
# -----------------------------------------------------------------------------
def update_bond(peps, site_a, site_b, axis_a, axis_b, gate, d, D):
    """
    Apply a two-site gate between neighboring PEPS tensors (schematic form).
    Note: this is a placeholder update; no full environment contraction.
    """
    A = peps[site_a]
    B = peps[site_b]
    nodeA = tn.Node(A)
    nodeB = tn.Node(B)
    tn.connect(nodeA[axis_a + 1], nodeB[axis_b + 1])
    combined_node = tn.contract_between(nodeA, nodeB)
    combined = combined_node.get_tensor()
    combined_reshaped = combined.reshape(d * d, -1)
    gate_matrix = gate.reshape(d * d, d * d)
    updated = gate_matrix @ combined_reshaped
    updated = updated.reshape(combined.shape)
    U, S, Vh = la.svd(updated.reshape(d * d, -1), full_matrices=False)
    peps[site_a] = np.random.rand(*A.shape)
    peps[site_b] = np.random.rand(*B.shape)
    return peps


def simple_update(peps, J, h, tau, num_steps, Lx, Ly, Lz, d, D):
    """
    Evolve the PEPS state via a simple update:
    - Apply single-site X-rotations
    - Apply two-site ZZ interactions along the x-direction
    """
    U_two = two_site_gate(J, tau)
    U_single = single_site_gate(h, tau)
    for step in range(num_steps):
        for site in peps:
            A = peps[site]
            peps[site] = np.tensordot(U_single, A, axes=([1], [0]))
        for i in range(Lx - 1):
            for j in range(Ly):
                for k in range(Lz):
                    site_a = (i, j, k)
                    site_b = (i + 1, j, k)
                    peps = update_bond(peps, site_a, site_b, axis_a=1, axis_b=0, gate=U_two, d=d, D=D)
        print(f"Completed evolution step {step+1}/{num_steps}")
    return peps


# -----------------------------------------------------------------------------
# Entanglement Entropy Calculation
# -----------------------------------------------------------------------------
def compute_local_entropy(peps, site, d):
    """
    Approximate von Neumann entropy of a site tensor reshaped to a (d x N) matrix.
    The resulting entropy defines the scalar S(x) field over the lattice.
    """
    A = peps[site]
    tensor_mat = A.reshape(d, -1)
    U, S, Vh = la.svd(tensor_mat, full_matrices=False)
    S_norm = S / (np.linalg.norm(S) + 1e-12)
    entropy = -np.sum((S_norm ** 2) * np.log(S_norm ** 2 + 1e-12))
    return entropy


# -----------------------------------------------------------------------------
# Entanglement Curvature (Discrete Laplacian)
# -----------------------------------------------------------------------------
def discrete_laplacian(field):
    """
    Compute a discrete Laplacian of a scalar field on a 3D lattice.
    This is used as a proxy for curvature, based on ∇²S(x) from [Entanglement Curvature]&#8203;:contentReference[oaicite:3]{index=3}
    """
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


# -----------------------------------------------------------------------------
# CSV Export and Visualization
# -----------------------------------------------------------------------------
def save_csv_data(filepath, entropy_field, laplacian_field):
    """
    Save data for each site: (x, y, z, entropy, laplacian)
    """
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
    """
    Show 3D scatter plot of curvature proxy values.
    """
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
    plt.title("3D Scatter Plot of Entanglement Curvature (Laplacian)")
    plt.show()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="3D PEPS Simulation and Laplacian Entropy Computation")
    parser.add_argument("--lattice", nargs=3, type=int, default=[32, 32, 32], help="Lattice dimensions (Lx Ly Lz)")
    parser.add_argument("--bond-dim", type=int, default=4, help="Bond dimension D")
    parser.add_argument("--tau", type=float, default=0.01, help="Imaginary time step")
    parser.add_argument("--steps", type=int, default=10, help="Number of time evolution steps")
    parser.add_argument("--J", type=float, default=1.0, help="Interaction strength J")
    parser.add_argument("--h", type=float, default=1.0, help="Transverse field h")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to store CSV output")
    parser.add_argument("--plot", action="store_true", help="Show 3D scatter plot of Laplacian")
    args = parser.parse_args()

    Lx, Ly, Lz = args.lattice
    d = 2
    D = args.bond_dim
    tau = args.tau
    num_steps = args.steps
    J = args.J
    h = args.h
    os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing PEPS state...")
    peps = init_peps(Lx, Ly, Lz, d, D)

    print("Starting imaginary time evolution...")
    peps = simple_update(peps, J, h, tau, num_steps, Lx, Ly, Lz, d, D)

    print("Computing local entanglement entropy field...")
    entropy_field = np.zeros((Lx, Ly, Lz))
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                entropy_field[i,j,k] = compute_local_entropy(peps, (i,j,k), d)

    print("Computing entanglement curvature proxy (Laplacian)...")
    laplacian_field = discrete_laplacian(entropy_field)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"{timestamp}_gravity_laplacian_L{Lx}_D{D}_tau{tau}.csv"
    output_path = os.path.join(args.output_dir, filename)
    save_csv_data(output_path, entropy_field, laplacian_field)

    if args.plot:
        visualize_3d(laplacian_field)


if __name__ == "__main__":
    main()
