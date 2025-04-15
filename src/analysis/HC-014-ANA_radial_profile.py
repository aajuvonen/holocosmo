# HC-014-ANA: Radial profile
# Created on 2025-04-15T19:58:03.818034
"""
Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
This script computes the radial profile of an emergent entanglement curvature field.
It reads a CSV file containing a 3D lattice of data points with columns:
    x, y, z, entropy, laplacian
and computes the Euclidean distance of each point from a specified center.
Then it bins the data in radial shells and calculates statistics (mean, standard deviation,
and count) for the Laplacian (curvature proxy) in each bin.

Inputs (via CLI):
------------------
--input-file FILE.csv       
   Input from HC-008-ANA. CSV with x, y, z, entropy, laplacian.
   If not provided, the script will scan for files matching
   "../../data/processed/HC-008-ANA_*_cluster_analysis.csv" and prompt the user.
--output-file FILENAME     
   Base name for the output CSV file (default: HC-014-ANA_radial_profile.csv).
--output-dir DIR     
   Output directory for the radial profile CSV (default: ../../data/processed/).
--bins INT                 
   Number of radial bins to use (default: 50).
--center CX CY CZ          
   Coordinates of the center point. If omitted, the geometric center of the input coordinates is used.

Outputs:
---------
- The radial profile CSV is saved to:
    "../../data/processed/HC-014-ANA_{timestamp}_radial_profile.csv"
- A figure is saved to:
    "../../data/figures/HC-014-ANA_{timestamp}_radial_profile.svg"
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute the radial profile of the curvature field from a 3D PEPS simulation CSV file."
    )
    parser.add_argument("--input-file", type=str, default=None,
                        help="Path to the input CSV file (should contain columns x, y, z, entropy, laplacian). If not provided, the script will scan for matching files in '../../data/processed/HC-008-ANA_*_cluster_analysis.csv'.")
    parser.add_argument("--output-file", type=str, default="HC-014-ANA_radial_profile.csv",
                        help="Base name for the output CSV file. Timestamp and prefix will be added.")
    parser.add_argument("--output-dir", type=str, default="../../data/processed/",
                        help="Output directory for the radial profile CSV file.")
    parser.add_argument("--bins", type=int, default=50, help="Number of radial bins to use.")
    parser.add_argument("--center", type=float, nargs=3, metavar=('CX', 'CY', 'CZ'),
                        help="Coordinates (x y z) of the center point to compute radial distances from. If omitted, the geometric center is used.")
    return parser.parse_args()

def prompt_for_input_file():
    search_pattern = "../../data/processed/HC-008-ANA_*_cluster_analysis.csv"
    matching_files = sorted(glob.glob(search_pattern))
    if not matching_files:
        print("No matching files found in '../../data/processed/'. Exiting.")
        exit(0)
    print("Found the following files:")
    for idx, fname in enumerate(matching_files):
        print(f"[{idx}] {fname}")
    try:
        choice = int(input("Enter the index of the desired input file: "))
        return matching_files[choice]
    except Exception as e:
        print("Invalid selection. Exiting.")
        exit(0)

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

def save_radial_profile(output_dir, base_output_filename, bin_centers, mean_vals, std_vals, counts):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_filename = f"HC-014-ANA_{timestamp}_radial_profile.csv"
    full_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    df_out = pd.DataFrame({
        "radial_bin_center": bin_centers,
        "mean_laplacian": mean_vals,
        "std_laplacian": std_vals,
        "count": counts
    })
    df_out.to_csv(full_path, index=False)
    print(f"Radial profile CSV saved to: {full_path}")
    return output_filename  # returns filename with timestamp

def main():
    args = parse_arguments()
    
    if args.input_file is None:
        print("No --input-file provided.")
        args.input_file = prompt_for_input_file()

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
    
    # Save CSV in the output directory
    out_csv_filename = save_radial_profile(args.output_dir, args.output_file, bin_centers, mean_vals, std_vals, counts)
    
    # Plot and save figure into "../../data/figures/"
    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(bin_centers, mean_vals, yerr=std_vals, fmt='o-', capsize=4, label="Mean Laplacian")
    plt.title("Radial Profile of Entanglement Curvature")
    plt.xlabel("Radial Distance from Center")
    plt.ylabel("Mean Laplacian")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figures_dir = "../../data/figures/"
    os.makedirs(figures_dir, exist_ok=True)
    fig_filename = f"HC-014-ANA_{datetime.now().strftime('%Y%m%d-%H%M')}_radial_profile.svg"
    fig_path = os.path.join(figures_dir, fig_filename)
    plt.savefig(fig_path, format="svg")
    print(f"Figure saved to: {os.path.abspath(fig_path)}")
    
    plt.show()

if __name__ == "__main__":
    main()
