import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import heapq

# --- Load Laplacian Field ---
df = pd.read_csv("cluster_analysis.csv")

Lx = int(df['x'].max()) + 1
Ly = int(df['y'].max()) + 1
Lz = int(df['z'].max()) + 1
laplacian_grid = np.zeros((Lx, Ly, Lz))

for _, row in df.iterrows():
    laplacian_grid[int(row['x']), int(row['y']), int(row['z'])] = abs(row['laplacian'])

# Smooth the field for better path behavior
laplacian_smooth = gaussian_filter(laplacian_grid, sigma=1.5)

# --- Dijkstra Geodesic in 3D ---
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

# --- Compute Deviation from Neighbors ---
start = (2, 2, 2)
end = (29, 29, 29)
central_path = dijkstra_3d(laplacian_smooth, start, end)

# Neighbor offsets
offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
neighbor_paths = []

for dx, dy, dz in offsets:
    offset_start = (start[0]+dx, start[1]+dy, start[2]+dz)
    if all(0 <= o < s for o, s in zip(offset_start, laplacian_smooth.shape)):
        neighbor_paths.append(dijkstra_3d(laplacian_smooth, offset_start, end))

# Measure deviation
deviations = []
for i in range(len(central_path)):
    c = np.array(central_path[i])
    dists = []
    for path in neighbor_paths:
        if len(path) > i:
            p = np.array(path[i])
            dists.append(np.linalg.norm(p - c))
    if dists:
        deviations.append(np.mean(dists))
    else:
        deviations.append(np.nan)

# --- Build sparse deviation field ---
deviation_field = np.full_like(laplacian_smooth, np.nan, dtype=float)
for i, point in enumerate(central_path):
    if not np.isnan(deviations[i]):
        x, y, z = point
        deviation_field[x, y, z] = deviations[i]

# --- Compare Deviation to Curvature ---
coords = np.argwhere(~np.isnan(deviation_field))
dev_vals = np.array([deviation_field[tuple(c)] for c in coords])
lap_vals = np.array([laplacian_smooth[tuple(c)] for c in coords])

plt.figure(figsize=(8, 6))
plt.scatter(lap_vals, dev_vals, alpha=0.7, c='darkblue')
plt.xlabel("|ΔS(x)| (Laplacian)")
plt.ylabel("Geodesic Deviation D(x)")
plt.title("Deviation vs. Entanglement Curvature")
plt.grid(True)
plt.tight_layout()
plt.show()
