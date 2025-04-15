# HC-012-ANA: Visualize Laplacian 3D
# Created on 2025-04-15T19:28:13.493979
"""
Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
This script visualizes the Laplacian curvature field from a PEPS simulation in 3D.
It loads a CSV file containing x, y, z, laplacian values.

Inputs (via CLI):
------------------
--input FILE.csv     
   Input from HC-008-ANA. CSV with x, y, z, laplacian.
   If not provided, the script will scan for files matching
   "../../data/processed/HC-008-ANA_*_cluster_analysis.csv" and prompt the user.
--sample N           
   Randomly sample N points (default: 10000)
--threshold T        
   Only visualize points with |laplacian| >= T (default: 0.0)
--output-dir DIR     
   Output directory for figure (default: ../../data/figures/)
--show               
   Display figure interactively

Output:
--------
- A colored scatter plot saved as an SVG with the filename
  "HC-012-ANA_{timestamp}_laplacian_3d.svg"
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="3D visualization of Laplacian curvature field")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV with x, y, z, laplacian. If not provided, scan for matching files in '../../data/processed/HC-008-ANA_*_cluster_analysis.csv'")
    parser.add_argument("--sample", type=int, default=10000, help="Number of points to sample")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min |laplacian| value to include")
    parser.add_argument("--output-dir", type=str, default="../../data/figures/", help="Directory to save output figure")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # If no input file is provided, scan and prompt the user.
    if not args.input:
        print("No --input file provided. Scanning for matching files in '../../data/processed/'...")
        search_pattern = "../../data/processed/HC-008-ANA_*_cluster_analysis.csv"
        matching_files = sorted(glob.glob(search_pattern))
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
    # Filter by threshold and sample if necessary
    df_filtered = df[np.abs(df['laplacian']) >= args.threshold]
    print(f"Filtered down to {len(df_filtered)} points with |laplacian| >= {args.threshold}")
    
    if len(df_filtered) > args.sample:
        df_sampled = df_filtered.sample(n=args.sample, random_state=42)
    else:
        df_sampled = df_filtered

    print(f"Visualizing {len(df_sampled)} points...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        df_sampled['x'], df_sampled['y'], df_sampled['z'],
        c=df_sampled['laplacian'], cmap='plasma', s=4, alpha=0.8
    )
    plt.colorbar(sc, label="Laplacian")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("3D Visualization of Entanglement Curvature")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_path = os.path.join(args.output_dir, f"HC-012-ANA_{timestamp}_laplacian_3d.svg")
    plt.savefig(output_path, format="svg")
    print(f"Figure saved to {output_path}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
