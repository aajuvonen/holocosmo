"""
3D PEPS Simulation for a 32x32x32 Transverse-Field Ising Model
Features:
 - Lattice size: 32x32x32
 - Simplified imaginary time evolution using a PEPS “simple update”
 - Computation of local entanglement entropy at each site
 - Discrete Laplacian calculation of the entropy field (as a curvature proxy)
 - 3D scatter visualization of the Laplacian values
 - CSV export of (x, y, z, entropy, laplacian) data

Requirements:
  numpy, scipy, tensornetwork, matplotlib, and csv (standard library).
Install TensorNetwork via: pip install tensornetwork
"""

import numpy as np
import scipy.linalg as la
import tensornetwork as tn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# -----------------------------
# Lattice and simulation parameters
# -----------------------------
Lx, Ly, Lz = 32, 32, 32         # Lattice dimensions
d = 2                           # Physical dimension: spin-1/2
D = 2                           # Bond dimension (kept small for demonstration)
tau = 0.01                      # Imaginary time step
num_steps = 5                   # Number of imaginary time evolution steps (for demo)
J = 1.0                         # Coupling strength
h = 1.0                         # Transverse field strength

# -----------------------------
# Initialize 3D PEPS state
# -----------------------------
def init_peps(Lx, Ly, Lz, d, D):
    """
    Initialize a 3D PEPS state on a cubic lattice.
    Each tensor at site (i,j,k) has shape: (d, Dl, Dr, Du, Dd, Df, Db)
    where bond dimensions are D in the bulk and 1 at the boundaries.
    """
    peps = {}
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                Dl = D if i > 0       else 1
                Dr = D if i < Lx - 1  else 1
                Du = D if j > 0       else 1
                Dd = D if j < Ly - 1  else 1
                Df = D if k > 0       else 1
                Db = D if k < Lz - 1  else 1
                shape = (d, Dl, Dr, Du, Dd, Df, Db)
                peps[(i, j, k)] = np.random.rand(*shape) * 0.1
    return peps

# -----------------------------
# Gate Definitions
# -----------------------------
def two_site_gate(J, tau):
    """
    Construct a two-site gate: exp(tau * J * σ^z⊗σ^z)
    Returns a gate tensor of shape (d, d, d, d)
    """
    sz = np.array([[1, 0], [0, -1]])
    H_int = -J * np.kron(sz, sz)
    U = la.expm(-tau * H_int)
    return U.reshape(2, 2, 2, 2)

def single_site_gate(h, tau):
    """
    Construct a single-site gate: exp(tau * h * σ^x)
    Returns a 2x2 matrix.
    """
    sx = np.array([[0, 1], [1, 0]])
    U = la.expm(-tau * h * sx)
    return U

# -----------------------------
# Simplified Bond Update (schematic)
# -----------------------------
def update_bond(peps, site_a, site_b, axis_a, axis_b, gate):
    """
    Apply a two-site gate to the bond connecting site_a and site_b.
    This function contracts the two tensors along the specified bond,
    applies the gate to the physical indices, and splits the tensor via SVD.
    
    NOTE: This is a greatly simplified placeholder.
    """
    A = peps[site_a]
    B = peps[site_b]
    
    # Create tensornetwork nodes for controlled contraction
    nodeA = tn.Node(A, name=str(site_a))
    nodeB = tn.Node(B, name=str(site_b))
    # Connect the chosen bond; note: physical index is axis 0, so bonds start at index 1.
    # To connect site_a's right bond (index 2) and site_b's left bond (index 1):
    tn.connect(nodeA[axis_a+1], nodeB[axis_b+1])
    
    combined_node = tn.contract_between(nodeA, nodeB, name=f"combined_{site_a}_{site_b}")
    combined = combined_node.get_tensor()
    
    # Group the physical indices (assumed to be the first indices from each tensor)
    combined_shape = combined.shape
    new_shape = (d * d, -1)
    combined_reshaped = combined.reshape(new_shape)
    
    # Apply the two-site gate: reshape gate as a matrix acting on the two physical indices
    gate_matrix = gate.reshape(d*d, d*d)
    updated = gate_matrix @ combined_reshaped
    updated = updated.reshape(combined_shape)
    
    # Dummy SVD to "split" updated tensor (schematic; not a true update)
    U, S, Vh = la.svd(updated.reshape(d*d, -1), full_matrices=False)
    chi = min(D, len(S))
    U = U[:, :chi]
    S = S[:chi]
    Vh = Vh[:chi, :]
    
    # In a full update, one would reshape U and Vh into new tensors.
    # Here we simply overwrite the tensors with new random placeholders.
    peps[site_a] = np.random.rand(*A.shape)
    peps[site_b] = np.random.rand(*B.shape)
    
    return peps

# -----------------------------
# Simple Update Routine
# -----------------------------
def simple_update(peps, J, h, tau, num_steps):
    """
    Perform a simplified imaginary time evolution using the simple update:
      1. Apply single-site transverse field gate to all sites.
      2. Apply two-site gate updates along one bond direction (e.g., x-direction).
    """
    U_two = two_site_gate(J, tau)
    U_single = single_site_gate(h, tau)
    
    for step in range(num_steps):
        # Single-site update: contract U_single with the physical index (axis 0)
        for site in peps:
            A = peps[site]
            updated_tensor = np.tensordot(U_single, A, axes=([1], [0]))
            peps[site] = updated_tensor
        # Two-site updates along x-direction (bonds between (i,j,k) and (i+1,j,k))
        for i in range(Lx - 1):
            for j in range(Ly):
                for k in range(Lz):
                    site_a = (i, j, k)
                    site_b = (i+1, j, k)
                    # For correct connection:
                    #   site_a: use right bond (index 2) --> axis_a = 1 (since 1+1=2)
                    #   site_b: use left bond (index 1)  --> axis_b = 0 (since 0+1=1)
                    peps = update_bond(peps, site_a, site_b, axis_a=1, axis_b=0, gate=U_two)
        print(f"Completed evolution step {step+1}/{num_steps}")
    return peps

# -----------------------------
# Compute Local Entanglement Entropy (schematic)
# -----------------------------
def compute_local_entropy(peps, site):
    """
    Compute an approximate local von Neumann entropy for a given site tensor.
    Here we reshape the tensor to group the physical index and perform an SVD.
    (A true evaluation requires an environment contraction.)
    """
    A = peps[site]
    tensor_mat = A.reshape(d, -1)
    U, S, Vh = la.svd(tensor_mat, full_matrices=False)
    norm = np.linalg.norm(S)
    S_norm = S / (norm + 1e-12)
    eps_val = 1e-12
    entropy = -np.sum((S_norm**2) * np.log(S_norm**2 + eps_val))
    return entropy

# -----------------------------
# Compute Discrete Laplacian of a 3D Field
# -----------------------------
def discrete_laplacian(field):
    """
    Compute the discrete Laplacian of a 3D scalar field.
    At each point: Laplacian = sum(neighbor values) - (number of neighbors * value)
    """
    lap = np.zeros_like(field)
    nx, ny, nz = field.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                s_center = field[i, j, k]
                neighbor_sum = 0.0
                count = 0
                for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        neighbor_sum += field[ni, nj, nk]
                        count += 1
                lap[i, j, k] = neighbor_sum - count * s_center
    return lap

# -----------------------------
# Save results to CSV
# -----------------------------
def save_csv_data(filename, entropy_field, laplacian_field):
    """
    Save the 3D data for each lattice site:
      x, y, z, entropy, laplacian
    """
    nx, ny, nz = entropy_field.shape
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "z", "entropy", "laplacian"])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    writer.writerow([i, j, k, entropy_field[i,j,k], laplacian_field[i,j,k]])
    print(f"CSV data saved to {filename}")

# -----------------------------
# 3D Visualization of Laplacian (scatter plot)
# -----------------------------
def visualize_3d(laplacian_field):
    """
    Produce a 3D scatter plot of Laplacian values over the lattice.
    """
    nx, ny, nz = laplacian_field.shape
    xs, ys, zs, vals = [], [], [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                xs.append(i)
                ys.append(j)
                zs.append(k)
                vals.append(laplacian_field[i,j,k])
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=vals, cmap='plasma', marker='o', s=2)
    plt.colorbar(sc, label='Laplacian value')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("3D Scatter Plot of Entanglement Curvature (Laplacian)")
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    # Initialize PEPS state on 32x32x32 lattice
    print("Initializing PEPS state...")
    peps = init_peps(Lx, Ly, Lz, d, D)
    
    # Perform simplified imaginary time evolution
    print("Starting imaginary time evolution...")
    peps = simple_update(peps, J, h, tau, num_steps)
    
    # Compute local entanglement entropy for each site
    print("Computing local entropy field...")
    entropy_field = np.zeros((Lx, Ly, Lz))
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                entropy_field[i,j,k] = compute_local_entropy(peps, (i,j,k))
    
    # Compute the discrete Laplacian of the entropy field
    print("Computing discrete Laplacian...")
    laplacian_field = discrete_laplacian(entropy_field)
    
    # Save the data to CSV for further analysis
    save_csv_data("peps_results.csv", entropy_field, laplacian_field)
    
    # Visualize the Laplacian field in 3D using a scatter plot
    visualize_3d(laplacian_field)

