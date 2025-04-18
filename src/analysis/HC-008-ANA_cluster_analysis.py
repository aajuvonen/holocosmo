# HC-008-ANA: Cluster analysis
# Created on 2025-04-15T18:39:35.885562
"""
Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
This script analyzes entanglement curvature from a PEPS simulation outputs HC-006-MOD or HC-007-MOD.
It performs:
- DBSCAN clustering of (x, y, z, laplacian) values
- A spatial correlation function based on Laplacian difference over random pairs

Inputs (via CLI):
------------------
--input FILE.csv        Path to input CSV. Input from HC-006-MOD or HC-007-MOD (x, y, z, entropy, laplacian).
                        If not provided, the script will scan for matching files in
                        "../../data/interim/HC-006-MOD*" and "../../data/interim/HC-007-MOD*"
--output-dir PATH       Directory for output files (default: ../../data/processed/)
--pairs N               Number of random point pairs (default: 1,000,000)
--eps FLOAT             DBSCAN epsilon (default: 1.5)
--min-samples INT       DBSCAN min_samples (default: 5)
--bins INT              Number of distance bins (default: 200)
--laplacian-threshold F Only consider points with |laplacian| ≥ threshold for clustering (optional)

Outputs:
---------
- CSV with all points + cluster labels
- CSV with binned spatial correlation values (NaNs dropped)

Paper References:
------------------
Draws on clustering behavior and correlation structures in the Laplacian (curvature proxy),
referencing:
- HC-004-DOC: Entanglement Curvature: A Tensorial Approach to Emergent Geometry from Quantum Information
- HC-005-DOC: An Effective Field Equation for Emergent Gravity from Quantum Entanglement
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import binned_statistic
import csv
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Cluster & correlate entanglement curvature field")
    parser.add_argument("--input", type=str, default=None,
                        help="CSV file with x, y, z, laplacian. If not provided, the script scans for HC-006-MOD* and HC-007-MOD* files in ../../data/interim/")
    parser.add_argument("--output-dir", type=str, default="../../data/processed", help="Directory for output CSVs")
    parser.add_argument("--pairs", type=int, default=1000000, help="Number of random point pairs")
    parser.add_argument("--eps", type=float, default=1.5, help="DBSCAN epsilon")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--bins", type=int, default=200, help="Number of distance bins for correlation")
    parser.add_argument("--laplacian-threshold", type=float, default=None,
                        help="Only include points with |laplacian| >= threshold for DBSCAN (optional)")
    args = parser.parse_args()

    # If no input file is provided, search the default directories
    if not args.input:
        print("No --input file provided. Scanning for matching files in '../../data/interim/'...")
        files_hc006 = glob.glob("../../data/interim/HC-006-MOD*")
        files_hc007 = glob.glob("../../data/interim/HC-007-MOD*")
        all_files = sorted(files_hc006 + files_hc007)
        if not all_files:
            print("No matching files found. Exiting.")
            return
        print("Found the following files:")
        for idx, f in enumerate(all_files):
            print(f"[{idx}] {f}")
        try:
            choice = int(input("Enter the number corresponding to the desired input file: "))
            args.input = all_files[choice]
        except Exception as e:
            print("Invalid selection. Exiting.")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading input data from:", args.input)
    data = pd.read_csv(args.input)

    # Filter data for clustering if threshold is set
    if args.laplacian_threshold is not None:
        cluster_mask = np.abs(data['laplacian']) >= args.laplacian_threshold
        cluster_data = data[cluster_mask].copy()
        print(f"Filtered {len(data) - len(cluster_data)} points with |laplacian| < {args.laplacian_threshold}")
    else:
        cluster_data = data.copy()

    coords = cluster_data[['x', 'y', 'z']].values
    laplacian = cluster_data['laplacian'].values

    print("Running DBSCAN clustering...")
    if len(coords) == 0:
        print("No points passed the Laplacian threshold. Skipping clustering.")
        cluster_data['cluster_label'] = []
    else:
        dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=-1)
        clusters = dbscan.fit_predict(np.column_stack((coords, laplacian)))
        cluster_data['cluster_label'] = clusters

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    cluster_file = os.path.join(args.output_dir, f"HC-008-ANA_{timestamp}_cluster_analysis.csv")
    cluster_data.to_csv(cluster_file, index=False)
    print(f"Saved cluster results to {cluster_file}")

    # --- Spatial Correlation ---
    print("Computing spatial correlation function...")
    np.random.seed(42)
    all_coords = data[['x', 'y', 'z']].values
    all_laplacian = data['laplacian'].values
    total_points = len(all_coords)

    idx_a = np.random.choice(total_points, args.pairs, replace=True)
    idx_b = np.random.choice(total_points, args.pairs, replace=True)

    distances = np.linalg.norm(all_coords[idx_a] - all_coords[idx_b], axis=1)
    laplacian_diff = all_laplacian[idx_a] - all_laplacian[idx_b]

    corr_values, bin_edges, _ = binned_statistic(
        distances,
        laplacian_diff,
        statistic=lambda x: np.mean(x**2),
        bins=args.bins
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Debug: print bin occupancy
    hist, _ = np.histogram(distances, bins=bin_edges)
    populated_bins = (hist > 0).sum()
    print(f"Correlation: {populated_bins}/{len(hist)} bins populated with data.")
    print(f"Dropping {np.isnan(corr_values).sum()} bins with NaN correlation values.")

    correlation_file = os.path.join(args.output_dir, f"HC-008-ANA_{timestamp}_spatial_correlation.csv")
    with open(correlation_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['distance_bin_center', 'correlation_value'])
        for dist, corr in zip(bin_centers, corr_values):
            if not np.isnan(corr):
                writer.writerow([dist, corr])
    print(f"Saved spatial correlation to {correlation_file}")

if __name__ == "__main__":
    main()
