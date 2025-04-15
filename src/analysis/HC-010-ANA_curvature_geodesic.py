# HC-010-ANA: Curvature geodesic
# Created on 2025-04-15T19:07:16.081302
"""
Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
This approximates extremal paths in emergent curvature by treating |ΔS| as a cost surface.
It reflects concepts from tensor network geometry and entanglement-based gravity.

Finds and visualizes a geodesic path through the entanglement curvature field,
interpreted as a cost field (via |laplacian|), using Dijkstra's algorithm in 3D.

Inputs (via CLI):
------------------
--input FILE.csv       
   Input from HC-008-ANA. CSV with x, y, z, laplacian.
   If not provided, the script will scan for files matching
   "../../data/processed/HC-008-ANA_*_cluster_analysis.csv" and prompt the user.
--start x y z          
   Start coordinate in the lattice (default: 2 2 2)
--end x y z            
   End coordinate in the lattice (default: 29 29 29)
--smooth-sigma FLOAT   
   Sigma for Gaussian smoothing (default: 1.5)

Outputs:
---------
- A 3D plot showing the curvature field and the geodesic path, saved as an SVG in "../../data/figures/HC-010-ANA_{timestamp}_curvature_geodesic.svg"
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import heapq
from datetime import datetime

def dijkstra_3d(cost_grid, start, end):
    nx, ny, nz = cost_grid.shape
    visited = np.full((nx, ny, nz), False)
    dist = np.full((nx, ny, nz), np.inf)
    prev = np.empty((nx, ny, nz), dtype=object)
    heap = [(0, start)]
    dist[start] = 0

    neighbors = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]

    while heap:
        current_dist, (x, y, z) = heapq.heappop(heap)
        if visited[x, y, z]:
            continue
        visited[x, y, z] = True
        if (x, y, z) == end:
            break
        for dx, dy, dz in neighbors:
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
                new_dist = current_dist + cost_grid[nx_, ny_, nz_]
                if new_dist < dist[nx_, ny_, nz_]:
                    dist[nx_, ny_, nz_] = new_dist
                    prev[nx_, ny_, nz_] = (x, y, z)
                    heapq.heappush(heap, (new_dist, (nx_, ny_, nz_)))

    path = []
    p = end
    while p != start:
        path.append(p)
        p = prev[p]
        if p is None:
            return []  # No path found
    path.append(start)
    path.reverse()
    return path

def main():
    parser = argparse.ArgumentParser(description="Visualize geodesic path through curvature field")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV with x, y, z, laplacian. If not provided, the script will scan for files matching '../../data/processed/HC-008-ANA_*_cluster_analysis.csv'")
    parser.add_argument("--start", nargs=3, type=int, default=[2, 2, 2], help="Start coordinates (x y z)")
    parser.add_argument("--end", nargs=3, type=int, default=[29, 29, 29], help="End coordinates (x y z)")
    parser.add_argument("--smooth-sigma", type=float, default=1.5, help="Gaussian smoothing sigma")
    args = parser.parse_args()

    # If input file not provided, scan for matching files and prompt user
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

    print("Loading curvature data from:", args.input)
    df = pd.read_csv(args.input)

    Lx = int(df['x'].max()) + 1
    Ly = int(df['y'].max()) + 1
    Lz = int(df['z'].max()) + 1

    laplacian_grid = np.zeros((Lx, Ly, Lz))
    for _, row in df.iterrows():
        laplacian_grid[int(row['x']), int(row['y']), int(row['z'])] = abs(row['laplacian'])

    print(f"Applying Gaussian smoothing (sigma={args.smooth_sigma})...")
    laplacian_smooth = gaussian_filter(laplacian_grid, sigma=args.smooth_sigma)

    print(f"Finding geodesic path from {tuple(args.start)} to {tuple(args.end)}...")
    path = dijkstra_3d(laplacian_smooth, tuple(args.start), tuple(args.end))
    if not path:
        print("No valid path found.")
        return

    print(f"Geodesic path found with {len(path)} steps.")
    xs, ys, zs = zip(*path)

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Geodesic Path Through Entanglement Curvature")

    # Plot a subset of the curvature field points for visualization
    sample = (slice(None, None, 4), slice(None, None, 4), slice(None, None, 4))
    grid_shape = laplacian_smooth[sample].shape
    # Generate coordinates for the sampled points
    x_idx, y_idx, z_idx = np.indices(grid_shape)
    sc = ax.scatter(x_idx.flatten(), y_idx.flatten(), z_idx.flatten(),
                    c=laplacian_smooth[sample].flatten(), cmap='plasma', s=5)

    # Plot the geodesic path
    ax.plot(xs, ys, zs, color='cyan', linewidth=2, label='Geodesic Path')
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.colorbar(sc, label='|ΔS| (Curvature)')
    plt.tight_layout()

    # Create figures output directory and save SVG figure with required naming
    figures_dir = "../../data/figures"
    os.makedirs(figures_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_fig_file = os.path.join(figures_dir, f"HC-010-ANA_{timestamp}_curvature_geodesic.svg")
    plt.savefig(output_fig_file, format="svg")
    print(f"Saved SVG figure to {output_fig_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
