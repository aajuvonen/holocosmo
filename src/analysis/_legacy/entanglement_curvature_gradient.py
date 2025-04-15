"""
entanglement_curvature_gradient.py

Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
Visualizes the gradient (∇Laplacian) of the curvature field computed from a PEPS simulation.
It produces a quiver plot for a fixed z-slice of the 3D gradient field.

Inputs (via CLI):
------------------
--input FILE.csv       Input CSV with columns: x, y, z, entropy, laplacian
--lattice-size INT     Lattice dimension (assumes cubic, default: 32)
--slice-z INT          Z-slice to visualize (default: 16)

Outputs:
---------
- A matplotlib quiver plot showing ∇Laplacian on the given z-slice

Scientific Context:
------------------
Gradient fields of the Laplacian serve as a directional curvature signal and
can be interpreted as effective gravitational field directions in emergent geometry frameworks.

"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize ∇Laplacian from PEPS data")
    parser.add_argument("--input", type=str, required=True, help="CSV file with x,y,z,entropy,laplacian")
    parser.add_argument("--lattice-size", type=int, default=32, help="Assumed lattice size (cubic)")
    parser.add_argument("--slice-z", type=int, default=16, help="Z-slice index to visualize")
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.input)
    laplacian_values = df['laplacian'].values
    L = args.lattice_size

    # Reshape flat Laplacian list into a 3D field (assumes row order matches spatial order)
    lap_field = laplacian_values.reshape((L, L, L))

    print("Computing gradients...")
    grad_x, grad_y, grad_z = np.gradient(lap_field)

    print(f"Visualizing curvature gradient on z-slice {args.slice_z}...")
    X, Y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    U = grad_x[:, :, args.slice_z]
    V = grad_y[:, :, args.slice_z]

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, -U, -V, color='darkred', scale=50, width=0.002)
    plt.title(f"Curvature Gradient Field (z = {args.slice_z})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
