"""
Corrected Entanglement Curvature Analysis Script

Performs:
- DBSCAN clustering on the full dataset.
- Computes spatial correlation function using random pairs.

Input:
- 'peps_results.csv' (full dataset).

Outputs:
- 'cluster_analysis.csv' (all points labeled)
- 'spatial_correlation.csv' (spatial correlation)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import binned_statistic
import csv

# Adjustable parameters (optimized for local 16GB RAM environment)
INPUT_CSV = 'peps_results.csv'
CLUSTER_CSV = 'cluster_analysis.csv'
CORRELATION_CSV = 'spatial_correlation.csv'

NUM_RANDOM_PAIRS = 1000000
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 5
NUM_DISTANCE_BINS = 1000

# Load data
print("Loading full data...")
data = pd.read_csv(INPUT_CSV)
coords = data[['x', 'y', 'z']].values
laplacian = data['laplacian'].values

# --- FULL CLUSTER ANALYSIS ---
print("Performing full cluster analysis...")
dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
clusters = dbscan.fit_predict(np.column_stack((coords, laplacian)))

# Save cluster analysis results (all data points)
print(f"Saving full clustering results to {CLUSTER_CSV}...")
cluster_df = pd.DataFrame({
    'x': data['x'],
    'y': data['y'],
    'z': data['z'],
    'laplacian': laplacian,
    'cluster_label': clusters
})
cluster_df.to_csv(CLUSTER_CSV, index=False)

# --- SPATIAL CORRELATION ---
print("Computing spatial correlation function...")

# Random pair indices from the full dataset
np.random.seed(42)
total_points = len(coords)
idx_a = np.random.choice(total_points, NUM_RANDOM_PAIRS, replace=True)
idx_b = np.random.choice(total_points, NUM_RANDOM_PAIRS, replace=True)

# Compute distances and Laplacian differences
distances = np.linalg.norm(coords[idx_a] - coords[idx_b], axis=1)
laplacian_diff = laplacian[idx_a] - laplacian[idx_b]

# Calculate correlation (variance) in distance bins
corr_values, bin_edges, _ = binned_statistic(
    distances,
    laplacian_diff,
    statistic=lambda x: np.mean(x**2),
    bins=NUM_DISTANCE_BINS
)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Save correlation results
print(f"Saving spatial correlation results to {CORRELATION_CSV}...")
with open(CORRELATION_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['distance_bin_center', 'correlation_value'])
    for dist, corr in zip(bin_centers, corr_values):
        writer.writerow([dist, corr])

print("Analysis complete.")
