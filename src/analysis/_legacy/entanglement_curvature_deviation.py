"""
entanglement_curvature_deviation.py

Author: The HoloCosmo Project
Date: April 2025

Description:
-------------
Compares geodesic deviation to the magnitude of entanglement curvature.
This estimates how sensitive geodesic paths are to small perturbations in starting position.

Inputs (via CLI):
------------------
--input FILE.csv       CSV with x, y, z, laplacian
--start x y z          Central path start point (default: 2 2 2)
--end x y z            End point (default: 29 29 29)
--smooth-sigma FLOAT   Gaussian smoothing (default: 1.5)

Output:
--------
- A scatter plot: Laplacian vs average geodesic deviation

Scientific Context:
-------------------
Deviation measures path instability, revealing the local "stiffness" of entanglement geometry.
Related to geodesic congruence and curvature gradients in emergent gravity frameworks.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import heapq

def dijkstra_3d(cost_grid, start, end):
    shape = cost_grid.shape
    visited = np.full(shape, False)
    dist = np.full(shape, np.inf)
    prev = np.empty(shape, dtype=object)
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
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                new_dist = current_dist + cost_grid[nx, ny, nz]
                if new_dist < dist[nx, ny, nz]:
                    dist[nx, ny, nz] = new_dist
                    prev[nx, ny, nz] = (x, y, z)
                    heapq.heappush(heap, (new_dist, (nx, ny, nz)))

    # Reconstruct path
    path = []
    p = end
    while p != start:
        path.append(p)
        p = prev[p]
        if p is None:
            return []
    path.append(start)
    path.reverse()
    return path

def main():
    parser = argparse.ArgumentParser(description="Compare curvature vs. geodesic deviation")
    parser.add_argument("--input", type=str, required=True, help="Input CSV with x,y,z,laplacian")
    parser.add_argument("--start", nargs=3, type=int, default=[2, 2, 2], help="Start point for central path")
    parser.add_argument("--end", nargs=3, type=int, default=[29, 29, 29], help="End point")
    parser.add_argument("--smooth-sigma", type=float, default=1.5, help="Gaussian smoothing sigma")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    shape = (
        int(df['x'].max()) + 1,
        int(df['y'].max()) + 1,
        int(df['z'].max()) + 1
    )

    lap_grid = np.zeros(shape)
    for _, row in df.iterrows():
        lap_grid[int(row['x']), int(row['y']), int(row['z'])] = abs(row['laplacian'])

    print(f"Smoothing curvature field (σ={args.smooth_sigma})...")
    lap_smooth = gaussian_filter(lap_grid, sigma=args.smooth_sigma)

    print(f"Computing central path from {tuple(args.start)} to {tuple(args.end)}...")
    central_path = dijkstra_3d(lap_smooth, tuple(args.start), tuple(args.end))

    offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    neighbor_paths = []

    print("Computing neighbor geodesics...")
    for dx, dy, dz in offsets:
        offset_start = (args.start[0]+dx, args.start[1]+dy, args.start[2]+dz)
        if all(0 <= o < s for o, s in zip(offset_start, lap_smooth.shape)):
            neighbor_paths.append(dijkstra_3d(lap_smooth, offset_start, tuple(args.end)))

    deviations = []
    for i in range(len(central_path)):
        c = np.array(central_path[i])
        dists = []
        for path in neighbor_paths:
            if len(path) > i:
                p = np.array(path[i])
                dists.append(np.linalg.norm(p - c))
        deviations.append(np.mean(dists) if dists else np.nan)

    dev_field = np.full_like(lap_smooth, np.nan, dtype=float)
    for i, point in enumerate(central_path):
        if not np.isnan(deviations[i]):
            dev_field[point] = deviations[i]

    coords = np.argwhere(~np.isnan(dev_field))
    dev_vals = np.array([dev_field[tuple(c)] for c in coords])
    lap_vals = np.array([lap_smooth[tuple(c)] for c in coords])

    print("Plotting deviation vs curvature...")
    plt.figure(figsize=(8, 6))
    plt.scatter(lap_vals, dev_vals, alpha=0.7, c='darkblue')
    plt.xlabel("|ΔS(x)| (Laplacian)")
    plt.ylabel("Geodesic Deviation D(x)")
    plt.title("Deviation vs. Entanglement Curvature")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

