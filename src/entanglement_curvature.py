import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import heapq

# Load Laplacian data
df = pd.read_csv("cluster_analysis.csv")

# Create 3D grid from Laplacian field
Lx = int(df['x'].max()) + 1
Ly = int(df['y'].max()) + 1
Lz = int(df['z'].max()) + 1
laplacian_grid = np.zeros((Lx, Ly, Lz))

for _, row in df.iterrows():
    laplacian_grid[int(row['x']), int(row['y']), int(row['z'])] = abs(row['laplacian'])

# Optional: smooth field for cleaner pathfinding
laplacian_smooth = gaussian_filter(laplacian_grid, sigma=1.5)

# Dijkstra's algorithm in 3D
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

# Choose start and end points
start = (2, 2, 2)
end = (29, 29, 29)

# Compute path
geodesic_path = dijkstra_3d(laplacian_smooth, start, end)

# Extract coordinates
xs, ys, zs = zip(*geodesic_path)

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Geodesic Path Through Entanglement Curvature")

# Plot curvature field (downsampled for performance)
sample = (slice(None, None, 4), slice(None, None, 4), slice(None, None, 4))
sc = ax.scatter(*np.indices(laplacian_smooth[sample].shape).reshape(3, -1), 
                c=laplacian_smooth[sample].flatten(), cmap='plasma', s=5)

# Plot path
ax.plot(xs, ys, zs, color='cyan', linewidth=2, label='Geodesic Path')
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.colorbar(sc, label='|ΔS| (Curvature)')
plt.tight_layout()
plt.show()
