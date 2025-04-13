"""
entanglement_cluster_analysis.py

Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
This script analyzes entanglement curvature from a PEPS simulation output.
It performs:
- DBSCAN clustering of (x, y, z, laplacian) values
- A spatial correlation function based on Laplacian difference over random pairs

Inputs (via CLI):
------------------
--input FILE.csv        Path to input CSV (must contain x, y, z, laplacian)
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

Scientific Basis:
------------------
Draws on clustering behavior and correlation structures in the Laplacian (curvature proxy),
referencing:
- "Entanglement Curvature: A Tensorial Approach..."
- "An Effective Field Equation for Emergent Gravity..."
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import binned_statistic
import csv
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Cluster & correlate entanglement curvature field")
    parser.add_argument("--input", type=str, required=True, help="CSV file with x, y, z, laplacian")
    parser.add_argument("--output-dir", type=str, default="../../data/processed", help="Directory for output CSVs")
    parser.add_argument("--pairs", type=int, default=1000000, help="Number of random point pairs")
    parser.add_argument("--eps", type=float, default=1.5, help="DBSCAN epsilon")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--bins", type=int, default=200, help="Number of distance bins for correlation")
    parser.add_argument("--laplacian-threshold", type=float, default=None,
                        help="Only include points with |laplacian| >= threshold for DBSCAN (optional)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading input data...")
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
    cluster_file = os.path.join(args.output_dir, f"{timestamp}_cluster_analysis.csv")
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

    correlation_file = os.path.join(args.output_dir, f"{timestamp}_spatial_correlation.csv")
    with open(correlation_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['distance_bin_center', 'correlation_value'])
        for dist, corr in zip(bin_centers, corr_values):
            if not np.isnan(corr):
                writer.writerow([dist, corr])
    print(f"Saved spatial correlation to {correlation_file}")

if __name__ == "__main__":
    main()
