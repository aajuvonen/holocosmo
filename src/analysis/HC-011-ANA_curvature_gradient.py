# HC-011-ANA: Curvature gradient
# Created on 2025-04-15T19:14:28.877452
"""
Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
Gradient fields of the Laplacian serve as a directional curvature signal and
can be interpreted as effective gravitational field directions in emergent geometry frameworks.

Visualizes the gradient (∇Laplacian) of the curvature field computed from a PEPS simulation.
It produces a quiver plot for a fixed z-slice of the 3D gradient field.

Inputs (via CLI):
------------------
--input FILE.csv        
   Path to input CSV. Input from HC-006-MOD or HC-007-MOD (x, y, z, entropy, laplacian).
   If not provided, the script will scan for matching files in
   "../../data/interim/HC-006-MOD*" and "../../data/interim/HC-007-MOD*"
--lattice-size INT     
   Lattice dimension (cubic, default: 32). This is ignored if the filename contains a pattern "_L#".
--slice-z INT          
   Z-slice index to visualize (default: 16)

Outputs:
---------
- A matplotlib quiver plot showing ∇Laplacian on the given z-slice,
  saved as an SVG in "../../data/figures/HC-011-ANA_{timestamp}_curvature_gradient.svg"
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Visualize ∇Laplacian from PEPS data")
    parser.add_argument("--input", type=str, default=None,
                        help="CSV file with x,y,z,entropy,laplacian. If not provided, scan for matching files in '../../data/interim/HC-006-MOD*' and '../../data/interim/HC-007-MOD*'")
    parser.add_argument("--lattice-size", type=int, default=32, help="Assumed lattice size (cubic). Overridden if filename contains '_L#'")
    parser.add_argument("--slice-z", type=int, default=16, help="Z-slice index to visualize")
    args = parser.parse_args()

    # If no input file is provided, scan for matching files and prompt the user
    if not args.input:
        print("No --input file provided. Scanning for matching files in '../../data/interim/'...")
        pattern1 = "../../data/interim/HC-006-MOD*"
        pattern2 = "../../data/interim/HC-007-MOD*"
        files1 = glob.glob(pattern1)
        files2 = glob.glob(pattern2)
        matching_files = sorted(files1 + files2)
        if not matching_files:
            print("No matching files found. Exiting.")
            return
        print("Found the following files:")
        for idx, fname in enumerate(matching_files):
            print(f"[{idx}] {fname}")
        try:
            choice = int(input("Enter the number corresponding to the desired input file: "))
            args.input = matching_files[choice]
        except Exception as e:
            print("Invalid selection. Exiting.")
            return

    print("Loading data from:", args.input)
    df = pd.read_csv(args.input)
    
    # Automatically detect lattice size from the filename if possible
    lattice_size = args.lattice_size
    match = re.search(r'_L(\d+)', args.input)
    if match:
        lattice_size = int(match.group(1))
        print(f"Detected lattice size from filename: {lattice_size} (i.e., {lattice_size}x{lattice_size}x{lattice_size})")
    else:
        print(f"No lattice size detected in filename. Using provided lattice size: {lattice_size}")

    # Use the detected lattice size to reshape the laplacian column into a 3D field
    try:
        laplacian_values = df['laplacian'].values
    except KeyError:
        print("Input CSV does not contain 'laplacian' column. Exiting.")
        return

    expected_points = lattice_size ** 3
    if len(laplacian_values) != expected_points:
        print(f"Warning: Expected {expected_points} points for a cubic lattice of size {lattice_size}, but got {len(laplacian_values)}. Adjusting lattice size is required.")
        # Optionally, you can add error handling here.
    
    try:
        lap_field = laplacian_values.reshape((lattice_size, lattice_size, lattice_size))
    except ValueError as ve:
        print("Error reshaping laplacian data. Check that the lattice size is correct.")
        return

    print("Computing gradients of the curvature field...")
    grad_x, grad_y, grad_z = np.gradient(lap_field)

    print(f"Visualizing gradient (∇Laplacian) on z-slice {args.slice_z}...")
    X, Y = np.meshgrid(np.arange(lattice_size), np.arange(lattice_size), indexing='ij')
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

    # Save figure to the figures directory with timestamp in the filename
    figures_dir = "../../data/figures"
    os.makedirs(figures_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_fig_file = os.path.join(figures_dir, f"HC-011-ANA_{timestamp}_curvature_gradient.svg")
    plt.savefig(output_fig_file, format="svg")
    print(f"Saved SVG figure to {output_fig_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
