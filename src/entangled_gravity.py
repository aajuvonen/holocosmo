import numpy as np
import matplotlib.pyplot as plt

# Set up a spatial grid
N = 100  # grid size in each dimension
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Define a synthetic "entropy" field S(x, y)
# For our toy model we use a Gaussian which is peaked at the origin.
S = np.exp(- (X**2 + Y**2))

# Compute first derivatives with respect to x and y using finite differences
S_x = np.gradient(S, dx, axis=1)
S_y = np.gradient(S, dy, axis=0)

# Compute second derivatives (the Hessian components)
S_xx = np.gradient(S_x, dx, axis=1)
S_yy = np.gradient(S_y, dy, axis=0)
S_xy = np.gradient(S_x, dy, axis=0)
S_yx = np.gradient(S_y, dx, axis=1)  # should be similar to S_xy

# Define components of our "entanglement curvature tensor" E_ij
E_xx = S_xx
E_yy = S_yy
# Symmetrize the off-diagonal component
E_xy = 0.5 * (S_xy + S_yx)

# For each point, compute the eigenvalues of the 2x2 Hessian matrix.
# These eigenvalues provide information analogous to principal curvatures.
eig1 = np.zeros_like(S)
eig2 = np.zeros_like(S)

for i in range(N):
    for j in range(N):
        H = np.array([[E_xx[i, j], E_xy[i, j]],
                      [E_xy[i, j], E_yy[i, j]]])
        vals, _ = np.linalg.eig(H)
        eig1[i, j] = np.max(vals)  # principal (largest) eigenvalue
        eig2[i, j] = np.min(vals)  # secondary (smallest) eigenvalue

# --- Visualization ---

# Plot the synthetic entropy field S(x,y)
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, S, levels=50, cmap='viridis')
plt.colorbar(label='S(x, y)')
plt.title('Synthetic Entropy Field S(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Plot the Hessian (entanglement curvature) components:
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, E_xx, levels=50, cmap='viridis')
plt.colorbar(label='E_xx')
plt.title('Entanglement Curvature Component E_xx')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, E_xy, levels=50, cmap='viridis')
plt.colorbar(label='E_xy')
plt.title('Entanglement Curvature Component E_xy')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, E_yy, levels=50, cmap='viridis')
plt.colorbar(label='E_yy')
plt.title('Entanglement Curvature Component E_yy')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Plot the eigenvalues of the entanglement curvature tensor.
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, eig1, levels=50, cmap='viridis')
plt.colorbar(label='Principal Eigenvalue')
plt.title('Principal Eigenvalue of the Curvature Tensor')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, eig2, levels=50, cmap='viridis')
plt.colorbar(label='Secondary Eigenvalue')
plt.title('Secondary Eigenvalue of the Curvature Tensor')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
