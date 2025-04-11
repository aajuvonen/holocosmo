"""
A schematic 3D PEPS simulation for a transverse-field Ising model on a 16x16x16 lattice.
This code demonstrates:
  - Initialization of a 3D PEPS state (each lattice site gets a tensor with one physical index and six bond indices).
  - A simplified imaginary time evolution using a Trotter-Suzuki decomposition.
  - A simplified two-site update along one bond direction using an SVD-based decomposition.
  - A rudimentary extraction of local entanglement entropy via an SVD on a site tensor.

WARNING: This code is a didactic skeleton. It uses very simplistic updates and contractions.
For serious simulations, one must implement a proper “simple” or “full” update algorithm and
an accurate environment approximation (e.g. via corner transfer matrices).
"""

import numpy as np
import scipy.linalg as la
import tensornetwork as tn
import matplotlib.pyplot as plt

# Set lattice and physical parameters
Lx, Ly, Lz = 16, 16, 16         # Lattice dimensions
d = 2                           # Physical dimension: spin-1/2
D = 2                           # Bond dimension (kept small for demonstration)
tau = 0.01                      # Imaginary time step
num_steps = 10                  # Number of imaginary time evolution steps
J = 1.0                         # Coupling strength for σ^z ⊗ σ^z term
h = 1.0                         # Transverse field strength

# -----------------------------
# Initialization of the 3D PEPS
# -----------------------------
def init_peps(Lx, Ly, Lz, d, D):
    """
    Initialize a 3D PEPS state on a cubic lattice.
    Each tensor has shape (d, D_left, D_right, D_up, D_down, D_front, D_back),
    with bond dimensions set to D in the bulk and 1 at the boundaries.
    The tensors are initialized with small random numbers.
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
    Construct a two-site gate corresponding to the interaction term exp(tau * J * σ^z ⊗ σ^z).
    The minus sign from the Hamiltonian is absorbed in the definition.
    Returns a 4-index tensor with shape (d, d, d, d).
    """
    sz = np.array([[1, 0], [0, -1]])
    H_int = -J * np.kron(sz, sz)
    U = la.expm(-tau * H_int)
    return U.reshape(2, 2, 2, 2)

def single_site_gate(h, tau):
    """
    Construct the single-site gate corresponding to the transverse-field term exp(tau * h * σ^x).
    Returns a 2x2 matrix.
    """
    sx = np.array([[0, 1], [1, 0]])
    U = la.expm(-tau * h * sx)
    return U

# -----------------------------
# Simplified Bond Update Routine
# -----------------------------
def update_bond(peps, site_a, site_b, axis_a, axis_b, gate):
    """
    Apply a two-site gate on the bond connecting site_a and site_b.
    This routine contracts the two tensors along the specified bond, applies the gate on
    the physical indices, and then performs an SVD to split back into two updated tensors.
    
    NOTE: This is a highly schematic procedure and does not fully reflect the complexity of a PEPS update.
    """
    A = peps[site_a]
    B = peps[site_b]
    
    # For simplicity, we contract over the chosen bond only.
    # In the tensor, the physical index is at position 0; bonds follow.
    # We assume axis_a in A and axis_b in B are the bond indices that connect.
    
    # Use tensornetwork for a controlled contraction:
    nodeA = tn.Node(A, name=str(site_a))
    nodeB = tn.Node(B, name=str(site_b))
    edge = nodeA[axis_a+1]  # +1 because axis 0 is physical index
    edge_B = nodeB[axis_b+1]
    tn.connect(nodeA[axis_a+1], nodeB[axis_b+1])
    
    # Contract the two nodes
    combined_node = tn.contract_between(nodeA, nodeB, name=f"combined_{site_a}_{site_b}")
    combined = combined_node.get_tensor()
    
    # Reshape combined tensor so that the two physical indices (from A and B) are grouped together.
    combined_shape = combined.shape
    new_shape = (d * d, -1)
    combined_reshaped = combined.reshape(new_shape)
    
    # Apply the two-site gate (reshape the gate accordingly)
    gate_matrix = gate.reshape(d * d, d * d)
    updated = gate_matrix @ combined_reshaped
    updated = updated.reshape(combined_shape)
    
    # SVD to split the tensor. Here we treat the first index of the reshaped updated tensor
    # as one group and the remaining as the other.
    U, S, Vh = la.svd(updated.reshape(d * d, -1), full_matrices=False)
    chi = min(D, len(S))
    U = U[:, :chi]
    S = S[:chi]
    Vh = Vh[:chi, :]
    
    # Reshape U and Vh back to tensor parts.
    # (This splitting is schematic and for illustration only.)
    new_shape_A = (d, ) + A.shape[1:-1] + (chi,)
    new_shape_B = (d, chi) + B.shape[2:]
    
    # In practice, one would carefully reshape and reassign these updated tensors.
    # Here we simply overwrite the tensors with dummy updated tensors.
    peps[site_a] = np.random.rand(*A.shape)  # placeholder: update with portion derived from U
    peps[site_b] = np.random.rand(*B.shape)  # placeholder: update with portion derived from Vh
    
    return peps

# -----------------------------
# Simple Update Routine for Imaginary Time Evolution
# -----------------------------
def simple_update(peps, J, h, tau, num_steps):
    """
    A very simplified imaginary time evolution (simple update) for the 3D PEPS.
    We apply the single-site transverse field gate to every tensor and then update bonds
    along the x-direction as an example.
    """
    U_two = two_site_gate(J, tau)
    U_single = single_site_gate(h, tau)
    
    for step in range(num_steps):
        # Apply single-site gate to all sites (update physical index)
        for site in peps:
            A = peps[site]
            # Update by contracting the single-site gate along the physical index (index 0)
            # Using tensornetwork for clarity.
            node = tn.Node(A)
            # Contract U_single with the physical index (axis 0)
            updated_tensor = np.tensordot(U_single, A, axes=([1], [0]))
            # The resulting physical dimension remains d.
            peps[site] = updated_tensor
        # Apply two-site gate along the x-direction bonds
        for i in range(Lx - 1):
            for j in range(Ly):
                for k in range(Lz):
                    site_a = (i, j, k)
                    site_b = (i+1, j, k)
                    # axis 0 is physical, so the bond on the right of A is at index 2,
                    # and the bond on the left of B is at index 1.
                    peps = update_bond(peps, site_a, site_b, axis_a=1, axis_b=0, gate=U_two)
        print(f"Completed evolution step {step+1}/{num_steps}")
    return peps

# -----------------------------
# Compute a Rudimentary Local Entanglement Entropy
# -----------------------------
def compute_local_entropy(peps, site):
    """
    Compute an approximate local von Neumann entropy for the tensor at the given site.
    This is done by reshaping the tensor to group the physical index with one bond,
    performing an SVD, and using the singular values.
    
    NOTE: Accurate evaluation in PEPS requires environmental contraction. This is only a rough proxy.
    """
    A = peps[site]
    # Reshape tensor: treat physical index as one part and all others as the second.
    tensor_mat = A.reshape(d, -1)
    U, S, Vh = la.svd(tensor_mat, full_matrices=False)
    # Normalize singular values to form a probability distribution.
    norm = np.linalg.norm(S)
    S_norm = S / (norm + 1e-12)
    eps_val = 1e-12
    entropy = -np.sum((S_norm**2) * np.log(S_norm**2 + eps_val))
    return entropy

# -----------------------------
# Main execution
# -----------------------------
if __name__ == '__main__':
    # Initialize the 3D PEPS state
    peps = init_peps(Lx, Ly, Lz, d, D)
    print("Initialized 3D PEPS state on a {}x{}x{} lattice.".format(Lx, Ly, Lz))
    
    # Perform imaginary time evolution using the simple update
    peps = simple_update(peps, J, h, tau, num_steps)
    
    # Compute local entanglement entropies on a slice (e.g., fixed k=0)
    entropy_field = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            entropy_field[i, j] = compute_local_entropy(peps, (i, j, 0))
    
    print("Approximate local entanglement entropy on the k=0 plane:")
    print(entropy_field)
    
    # Visualize the entropy field on the k=0 plane
    plt.figure(figsize=(6, 5))
    plt.imshow(entropy_field, cmap='viridis', origin='lower')
    plt.colorbar(label='Entropy')
    plt.title("Approximate Local Entropy (PEPS) at k=0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

