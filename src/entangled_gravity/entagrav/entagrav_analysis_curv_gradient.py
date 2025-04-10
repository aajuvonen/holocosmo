import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
CSV_FILE = "peps_results.csv"  # Input CSV file (with x,y,z,entropy,laplacian)
L = 32                         # Lattice size (assumes cubic)
SLICE_Z = 16                   # z-slice to visualize gradient on

# -----------------------------
# Load and reshape data
# -----------------------------
print("Loading data...")
df = pd.read_csv(CSV_FILE)
laplacian_values = df['laplacian'].values
lap_field = laplacian_values.reshape((L, L, L))  # shape: (x, y, z)

# -----------------------------
# Compute gradient (∇Laplacian)
# -----------------------------
print("Computing gradients...")
grad_x, grad_y, grad_z = np.gradient(lap_field)

# -----------------------------
# Visualize a 2D gradient slice
# -----------------------------
print(f"Visualizing curvature gradient on z-slice {SLICE_Z}...")

X, Y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
U = grad_x[:, :, SLICE_Z]
V = grad_y[:, :, SLICE_Z]

plt.figure(figsize=(8, 8))
plt.quiver(X, Y, -U, -V, color='darkred', scale=50, width=0.002)
plt.title(f"Curvature Gradient Field (z = {SLICE_Z})")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
