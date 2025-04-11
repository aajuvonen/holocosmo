"""
3D PEPS Simulation with Local Impurity (3x3x3 cube)
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
Lx, Ly, Lz = 32, 32, 32
d = 2
D = 3
tau = 0.01
num_steps = 15
J = 1.0
h = 1.0
h_impurity = 3.0  # transverse field in impurity region

# -----------------------------
# Initialize PEPS
# -----------------------------
def init_peps(Lx, Ly, Lz, d, D):
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
    sz = np.array([[1, 0], [0, -1]])
    H_int = -J * np.kron(sz, sz)
    U = la.expm(-tau * H_int)
    return U.reshape(2, 2, 2, 2)

def single_site_gate(h_val, tau):
    sx = np.array([[0, 1], [1, 0]])
    U = la.expm(-tau * h_val * sx)
    return U

# -----------------------------
# Simplified Bond Update
# -----------------------------
def update_bond(peps, site_a, site_b, axis_a, axis_b, gate):
    A = peps[site_a]
    B = peps[site_b]
    nodeA = tn.Node(A)
    nodeB = tn.Node(B)
    tn.connect(nodeA[axis_a+1], nodeB[axis_b+1])
    combined_node = tn.contract_between(nodeA, nodeB)
    combined = combined_node.get_tensor()
    combined_reshaped = combined.reshape(d * d, -1)
    gate_matrix = gate.reshape(d * d, d * d)
    updated = gate_matrix @ combined_reshaped
    updated = updated.reshape(combined.shape)
    U, S, Vh = la.svd(updated.reshape(d * d, -1), full_matrices=False)
    chi = min(D, len(S))
    peps[site_a] = np.random.rand(*A.shape)
    peps[site_b] = np.random.rand(*B.shape)
    return peps

# -----------------------------
# Simple Update with Impurity Region
# -----------------------------
def simple_update(peps, J, h, tau, num_steps, impurity_cube, h_impurity):
    U_two = two_site_gate(J, tau)
    U_single_default = single_site_gate(h, tau)
    U_single_imp = single_site_gate(h_impurity, tau)

    for step in range(num_steps):
        for site in peps:
            A = peps[site]
            if site in impurity_cube:
                updated_tensor = np.tensordot(U_single_imp, A, axes=([1], [0]))
            else:
                updated_tensor = np.tensordot(U_single_default, A, axes=([1], [0]))
            peps[site] = updated_tensor

        for i in range(Lx - 1):
            for j in range(Ly):
                for k in range(Lz):
                    site_a = (i, j, k)
                    site_b = (i+1, j, k)
                    peps = update_bond(peps, site_a, site_b, axis_a=1, axis_b=0, gate=U_two)

        print(f"Completed evolution step {step+1}/{num_steps}")
    return peps

# -----------------------------
# Entanglement Entropy Approximation
# -----------------------------
def compute_local_entropy(peps, site):
    A = peps[site]
    tensor_mat = A.reshape(d, -1)
    U, S, Vh = la.svd(tensor_mat, full_matrices=False)
    norm = np.linalg.norm(S)
    S_norm = S / (norm + 1e-12)
    eps_val = 1e-12
    entropy = -np.sum((S_norm**2) * np.log(S_norm**2 + eps_val))
    return entropy

# -----------------------------
# Discrete Laplacian
# -----------------------------
def discrete_laplacian(field):
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
# Save CSV
# -----------------------------
def save_csv_data(filename, entropy_field, laplacian_field):
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
# Visualize 3D Laplacian
# -----------------------------
def visualize_3d(laplacian_field):
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
    plt.title("3D Scatter Plot of Entanglement Curvature with Impurity")
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    print("Initializing PEPS state...")
    peps = init_peps(Lx, Ly, Lz, d, D)

    # Define impurity region (3x3x3 cube around center)
    cx, cy, cz = Lx//2, Ly//2, Lz//2
    impurity_cube = {
        (i, j, k)
        for i in range(cx-1, cx+2)
        for j in range(cy-1, cy+2)
        for k in range(cz-1, cz+2)
    }

    print("Starting imaginary time evolution with impurity cube...")
    peps = simple_update(peps, J, h, tau, num_steps, impurity_cube, h_impurity)

    print("Computing entropy field...")
    entropy_field = np.zeros((Lx, Ly, Lz))
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                entropy_field[i,j,k] = compute_local_entropy(peps, (i,j,k))

    print("Computing discrete Laplacian...")
    laplacian_field = discrete_laplacian(entropy_field)

    save_csv_data("peps_impurity_results.csv", entropy_field, laplacian_field)

    visualize_3d(laplacian_field)
