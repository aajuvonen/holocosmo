#!/usr/bin/env python3
"""
radial_profile.py
-----------------
This script computes the radial profile of an emergent entanglement curvature field.
It reads a CSV file containing a 3D lattice of data points with columns:
    x, y, z, entropy, laplacian
and computes the Euclidean distance of each point from a specified center.
Then it bins the data in radial shells and calculates statistics (mean, standard deviation,
and count) for the Laplacian (curvature proxy) in each bin.
The results are saved to an output CSV file.

Usage:
    python radial_profile.py --input-file path/to/input.csv \
                             --output-file radial_profile.csv \
                             [--bins 50] \
                             [--center 16 16 16]

If the --center option is omitted, the script computes the geometric center
of the input coordinates.

Author: The HoloCosmo Project
Date: April 2025
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from datetime import datetime
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute the radial profile of the curvature field from a 3D PEPS simulation CSV file.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input CSV file (should contain columns x, y, z, entropy, laplacian).")
    parser.add_argument("--output-file", type=str, default="radial_profile.csv",
                        help="Base name for the output CSV file (prefix will be added).")
    parser.add_argument("--output-dir", type=str, default="../../data/processed/",
                        help="Output directory for the radial profile CSV.")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of radial bins to use.")
    parser.add_argument("--center", type=float, nargs=3, metavar=('CX', 'CY', 'CZ'),
                        help="Coordinates (x y z) of the center point to compute radial distances from. "
                             "If omitted, the center of the data's bounding box is used.")
    return parser.parse_args()

def compute_geometric_center(df):
    cx = (df['x'].min() + df['x'].max()) / 2.0
    cy = (df['y'].min() + df['y'].max()) / 2.0
    cz = (df['z'].min() + df['z'].max()) / 2.0
    return np.array([cx, cy, cz])

def compute_radial_profile(df, center, nbins):
    coords = df[['x', 'y', 'z']].values
    laplacian = df['laplacian'].values
    distances = np.linalg.norm(coords - center, axis=1)
    max_dist = distances.max()
    bins = np.linspace(0, max_dist, nbins + 1)

    mean_vals, bin_edges, _ = binned_statistic(distances, laplacian, statistic='mean', bins=bins)
    std_vals, _, _ = binned_statistic(distances, laplacian, statistic='std', bins=bins)
    counts, _, _ = binned_statistic(distances, laplacian, statistic='count', bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, mean_vals, std_vals, counts

def save_radial_profile(output_dir, output_file, bin_centers, mean_vals, std_vals, counts):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    final_filename = f"{timestamp}_{output_file}"
    full_path = os.path.join(output_dir, final_filename)
    os.makedirs(output_dir, exist_ok=True)

    df_out = pd.DataFrame({
        "radial_bin_center": bin_centers,
        "mean_laplacian": mean_vals,
        "std_laplacian": std_vals,
        "count": counts
    })
    df_out.to_csv(full_path, index=False)
    print(f"Radial profile saved to {full_path}")
    return final_filename

def main():
    args = parse_arguments()
    print("Loading data from", args.input_file)
    df = pd.read_csv(args.input_file)

    if args.center is not None:
        center = np.array(args.center)
        print("Using provided center:", center)
    else:
        center = compute_geometric_center(df)
        print("No center provided. Using geometric center:", center)

    print("Computing radial profile...")
    bin_centers, mean_vals, std_vals, counts = compute_radial_profile(df, center, args.bins)

    output_filename = save_radial_profile(args.output_dir, args.output_file, bin_centers, mean_vals, std_vals, counts)

    # Optional plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(bin_centers, mean_vals, yerr=std_vals, fmt='o-', capsize=4, label="Mean Laplacian")
    plt.title("Radial Profile of Entanglement Curvature")
    plt.xlabel("Radial Distance from Center")
    plt.ylabel("Mean Laplacian")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
