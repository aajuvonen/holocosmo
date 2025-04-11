import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# --- Helper functions for building operators ---

# Pauli matrices
sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
sz = np.array([[1, 0],
               [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def kron_N(ops):
    """Compute the Kronecker product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def pauli_operator(N, site, op):
    """
    Construct an operator acting as 'op' on the given site,
    and as the identity on all other sites.
    'site' is an integer index (0 to N-1).
    """
    ops = []
    for i in range(N):
        ops.append(op if i == site else I2)
    return kron_N(ops)

# --- Setting up the 3D lattice: a 2x2x2 cube --- 
# Total number of spins:
Lx, Ly, Lz = 2, 2, 2
N = Lx * Ly * Lz  # Here, N = 8

def lattice_index(i, j, k):
    """Map 3D coordinate (i,j,k) to a linear index."""
    return i + Lx * (j + Ly * k)

def neighbors(i, j, k):
    """List the nearest neighbors (with open boundaries) for site (i,j,k)."""
    neigh = []
    for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < Lx and 0 <= nj < Ly and 0 <= nk < Lz:
            neigh.append((ni, nj, nk))
    return neigh

def build_3D_hamiltonian(J=1.0, h0=1.0, impurity_site=(0,0,0), h_imp=1.2):
    """
    Construct the Hamiltonian for a 3D transverse-field Ising model
    on a 2x2x2 lattice.
    
    H = -J * sum_<ij> σ^z_i σ^z_j - sum_i h_i σ^x_i,
    
    with h_i = h_imp for the impurity_site (to break full symmetry)
         and h_i = h0 otherwise.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    
    # Interaction: over nearest neighbors
    for k in range(Lz):
        for j in range(Ly):
            for i in range(Lx):
                idx = lattice_index(i, j, k)
                for (ni, nj, nk) in neighbors(i,j,k):
                    # To avoid double counting, only add if (i,j,k) < (ni,nj,nk) in lex order
                    if (i, j, k) < (ni, nj, nk):
                        idx_neighbor = lattice_index(ni, nj, nk)
                        op_i = pauli_operator(N, idx, sz)
                        op_j = pauli_operator(N, idx_neighbor, sz)
                        H -= J * (op_i @ op_j)
    
    # Transverse field term
    for k in range(Lz):
        for j in range(Ly):
            for i in range(Lx):
                idx = lattice_index(i, j, k)
                # Assign impurity transverse field on the chosen site.
                h_here = h_imp if (i,j,k) == impurity_site else h0
                H -= h_here * pauli_operator(N, idx, sx)
    
    return H

# --- Build and diagonalize the Hamiltonian ---
J = 1.0
h0 = 1.0
h_imp = 1.2  # Impurity: slightly different transverse field on one site.
H = build_3D_hamiltonian(J, h0, impurity_site=(0,0,0), h_imp=h_imp)

# Diagonalize to get the ground state.
eigs, vecs = la.eigh(H)
ground_state = vecs[:, 0]  # ground state vector

# --- Compute local (single–site) entanglement entropy ---
def single_site_entropy(psi, site, dims):
    """
    Compute the von Neumann entropy for the reduced density matrix
    of a single site 'site' (given by index) by tracing over all other subsystems.
    
    psi: ground state vector (1D array)
    dims: list of local Hilbert space dimensions (here, [2]*N)
    """
    N_system = len(dims)
    psi_tensor = psi.reshape(dims)
    # We want to trace out all sites except 'site'
    # Use tensordot: contract over all axes except the one at index 'site'.
    axes_to_trace = tuple(i for i in range(N_system) if i != site)
    rho = np.tensordot(psi_tensor, psi_tensor.conj(), axes=(axes_to_trace, axes_to_trace))
    # The resulting density matrix has shape (dims[site], dims[site]) (i.e. 2x2)
    # Ensure it is Hermitian:
    rho = 0.5 * (rho + rho.conj().T)
    # Compute eigenvalues and von Neumann entropy
    evals = np.linalg.eigvalsh(rho)
    # Avoid log(0): add a small epsilon (entropy is unchanged for zero eigenvalue)
    eps = 1e-12
    S = -np.sum(evals * np.log(evals + eps))
    return S.real

# dims: each spin is 2-dimensional.
dims = [2] * N
# Create an array to store the entropy for each lattice site.
S_field = np.zeros((Lx, Ly, Lz))
for k in range(Lz):
    for j in range(Ly):
        for i in range(Lx):
            idx = lattice_index(i, j, k)
            S_field[i,j,k] = single_site_entropy(ground_state, idx, dims)

print("Local Entanglement Entropy Field (S) on the 2x2x2 lattice:")
print(S_field)

# --- Compute a discrete Laplacian of the entropy field ---
# We use a simple nearest-neighbor difference approximation (with open boundaries)

def discrete_laplacian(field, spacing=1.0):
    """
    Compute the discrete Laplacian of a 3D field on a grid.
    For each site r, Laplacian = sum_{neighbors}(S(neighbor) - S(r)).
    """
    lap = np.zeros_like(field)
    nx, ny, nz = field.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                s_center = field[i,j,k]
                neighbor_sum = 0.0
                count = 0
                # Check the 6 nearest neighbors (if within bounds)
                for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        neighbor_sum += field[ni,nj,nk]
                        count += 1
                # Laplacian: (sum_neighbor - (count)*s_center)
                lap[i,j,k] = neighbor_sum - count * s_center
    return lap

lap_S = discrete_laplacian(S_field)
print("\nDiscrete Laplacian (entanglement curvature proxy) on the lattice:")
print(lap_S)

# --- Visualization ---
# To visualize the 3D fields, we can show slices along one direction.
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Show a slice (say, at k=0) of the entropy field.
im0 = axes[0].imshow(S_field[:,:,0], cmap='viridis', origin='lower',
                      extent=(0, Ly-1, 0, Lx-1))
axes[0].set_title("Local Entropy Field S(x,y) at k=0")
axes[0].set_xlabel("y")
axes[0].set_ylabel("x")
fig.colorbar(im0, ax=axes[0])

# Show the corresponding Laplacian slice (k=0)
im1 = axes[1].imshow(lap_S[:,:,0], cmap='plasma', origin='lower',
                      extent=(0, Ly-1, 0, Lx-1))
axes[1].set_title("Discrete Laplacian of S at k=0")
axes[1].set_xlabel("y")
axes[1].set_ylabel("x")
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()

