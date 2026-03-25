import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
from functorch.dim import Tensor
from numba import njit
import torch

import scipy as sc
import time
# -------------------------------
# Q8 Shape functions
# -------------------------------
time_beginning = time.time()
eta = 0  
k = 2*np.pi/(343) * 3000       # wave number or similar
N = 160     # elements in x
M = 160      # elements in y
H = 1
L = 1
def N_func(xi, eta):
    N = np.zeros(9)
    N[0] = 0.25 * xi*(xi-1) * eta*(eta-1)      # bottom-left
    N[1] = 0.5  * (1-xi**2) * eta*(eta-1)      # bottom-middle
    N[2] = 0.25 * xi*(xi+1) * eta*(eta-1)      # bottom-right
    N[3] = 0.5  * xi*(xi-1) * (1-eta**2)       # middle-left
    N[4] = (1-xi**2)*(1-eta**2)                # center
    N[5] = 0.5  * xi*(xi+1) * (1-eta**2)       # middle-right
    N[6] = 0.25 * xi*(xi-1) * eta*(eta+1)      # top-left
    N[7] = 0.5  * (1-xi**2) * eta*(eta+1)      # top-middle
    N[8] = 0.25 * xi*(xi+1) * eta*(eta+1)      # top-right
    return N

# -------------------------------
# Natural derivatives
# -------------------------------
def dN_nat(xi, eta):
    dN_dxi = np.zeros(9)
    dN_deta = np.zeros(9)
    
    # Node ordering:
    # 0 1 2
    # 3 4 5
    # 6 7 8

    # derivatives w.r.t xi
    dN_dxi[0] = 0.25*(2*xi - 1)*(eta - 1)*eta
    dN_dxi[1] = -xi*(eta - 1)*eta
    dN_dxi[2] = 0.25*(2*xi + 1)*(eta - 1)*eta
    dN_dxi[3] = 0.5*(2*xi - 1)*(1 - eta**2)
    dN_dxi[4] = -2*xi*(1 - eta**2)
    dN_dxi[5] = 0.5*(2*xi + 1)*(1 - eta**2)
    dN_dxi[6] = 0.25*(2*xi - 1)*(eta + 1)*eta
    dN_dxi[7] = -xi*(eta + 1)*eta
    dN_dxi[8] = 0.25*(2*xi + 1)*(eta + 1)*eta

    # derivatives w.r.t eta
    dN_deta[0] = 0.25*(xi - 1)*xi*(2*eta - 1)
    dN_deta[1] = 0.5*(1 - xi**2)*(2*eta - 1)
    dN_deta[2] = 0.25*(xi + 1)*xi*(2*eta - 1)
    dN_deta[3] = -0.5*(xi - 1)*xi*(2*eta)
    dN_deta[4] = -2*(1 - xi**2)*eta
    dN_deta[5] = -0.5*(xi + 1)*xi*(2*eta)
    dN_deta[6] = 0.25*(xi - 1)*xi*(2*eta + 1)
    dN_deta[7] = 0.5*(1 - xi**2)*(2*eta + 1)
    dN_deta[8] = 0.25*(xi + 1)*xi*(2*eta + 1)

    return dN_dxi, dN_deta
# -------------------------------
# Jacobian
# -------------------------------
def J(x_vector, y_vector, xi, eta):
    dN_dxi, dN_deta = dN_nat(xi, eta)
    Jmat = np.zeros((2,2))
    Jmat[0,0] = np.dot(dN_dxi, x_vector)   # dx/dxi
    Jmat[0,1] = np.dot(dN_dxi, y_vector)   # dy/dxi
    Jmat[1,0] = np.dot(dN_deta, x_vector)  # dx/deta
    Jmat[1,1] = np.dot(dN_deta, y_vector)  # dy/deta
    return Jmat

# -------------------------------
# Mass matrix (3x3 Gauss)
# -------------------------------


def B(x_vector, y_vector, xi, eta):
    # 1. natural derivatives
    dN_dxi, dN_deta = dN_nat(xi, eta)
    
    # 2. Jacobian
    Jmat = J(x_vector, y_vector, xi, eta)
    Jinv = np.linalg.inv(Jmat)
    
    # 3. derivatives in physical coordinates
    dN_dx_dy = Jinv @ np.vstack((dN_dxi, dN_deta))  # 2x8
    
    return dN_dx_dy  # shape (2, 8)


def M_element(x_vector, y_vector):
    # 3-point Gauss quadrature
    gp = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    gw = np.array([5/9, 8/9, 5/9])

    M = np.zeros((9, 9))  # Q9: 9 nodes

    for i in range(3):
        for j in range(3):
            xi  = gp[i]
            eta = gp[j]
            weight = gw[i] * gw[j]

            Jmat = J(x_vector, y_vector, xi, eta)
            detJ_val = np.linalg.det(Jmat)

            Nvals = N_func(xi, eta).flatten()  # length 9

            M += weight * detJ_val * np.outer(Nvals, Nvals)

    return M


def K_element(x_vector, y_vector):
    # 3-point Gauss quadrature
    gp = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    gw = np.array([5/9, 8/9, 5/9])

    K = np.zeros((9, 9))  # Q9: 9 nodes

    for i in range(3):
        for j in range(3):
            xi  = gp[i]
            eta = gp[j]
            weight = gw[i] * gw[j]

            Jmat = J(x_vector, y_vector, xi, eta)
            detJ_val = np.linalg.det(Jmat)

            Bvals = B(x_vector, y_vector, xi, eta)  # 2x9

            K += weight * detJ_val * np.matmul(Bvals.T, Bvals)

    return K

# -------------------------------
# Q9 element nodal coordinates (reference square)
x = np.array([-1,  0,  1,   -1,  0,  1,   -1,  0,  1])*(H/N)/2
y = np.array([-1, -1, -1,    0,  0,  0,    1,  1,  1])*(H/M)/2

# Compute element matrices
M_matrix_element = M_element(x, y)
K_matrix_element = K_element(x, y)

# total nodes in x and y for Q8
nx_nodes = 2*N + 1
ny_nodes = 2*M + 1

coordinates = []
node_index_map = {}
node_count = 0
for j in range(ny_nodes):
    for i in range(nx_nodes):
        coordinates.append([i, j])
        node_index_map[(i,j)] = node_count
        node_count += 1

coordinates = np.array(coordinates)
total_nodes = len(coordinates)


element_connectivity = []
for j in range(M):
    for i in range(N):
        n1 = node_index_map[(2*i  , 2*j  )]  # bottom-left
        n2 = node_index_map[(2*i+1, 2*j  )]  # bottom-middle
        n3 = node_index_map[(2*i+2, 2*j  )]  # bottom-right
        n4 = node_index_map[(2*i  , 2*j+1)]  # middle-left
        n5 = node_index_map[(2*i+1, 2*j+1)]  # center
        n6 = node_index_map[(2*i+2, 2*j+1)]  # middle-right
        n7 = node_index_map[(2*i  , 2*j+2)]  # top-left
        n8 = node_index_map[(2*i+1, 2*j+2)]  # top-middle
        n9 = node_index_map[(2*i+2, 2*j+2)]  # top-right
        element_connectivity.append([n1,n2,n3,n4,n5,n6,n7,n8,n9])

element_connectivity = np.array(element_connectivity, dtype=int)


# Initialize global matrices (DENSE)
"""
K_global = np.zeros((total_nodes, total_nodes))
M_global = np.zeros((total_nodes, total_nodes))
"""

# -------------------------------
# Sparse assembly
# -------------------------------
rows = []
cols = []
vals_K = []
vals_M = []

for e in range(N*M):
    nodes = element_connectivity[e]

    for i_local, I in enumerate(nodes):
        for j_local, J in enumerate(nodes):

            rows.append(I)
            cols.append(J)
            vals_K.append(K_matrix_element[i_local, j_local])
            vals_M.append(M_matrix_element[i_local, j_local])

# Convert to sparse matrices
K_global = sp.coo_matrix((vals_K, (rows, cols)), shape=(total_nodes, total_nodes)).tocsr()
M_global = sp.coo_matrix((vals_M, (rows, cols)), shape=(total_nodes, total_nodes)).tocsr()

# -------------------------------
# System matrix


C_gamma_vals = []
rows_C = []
cols_C = []
print(coordinates)
# Identify outer boundary nodes
left_boundary   = np.where(coordinates[:,0] == 0)[0]
right_boundary  = np.where(coordinates[:,0] == 2*N)[0]
bottom_boundary = np.where(coordinates[:,1] == 0)[0]
top_boundary    = np.where(coordinates[:,1] == 2*M)[0]
print(left_boundary)
boundary_nodes = np.unique(
    np.concatenate([left_boundary, right_boundary, bottom_boundary, top_boundary]))

# Simple 1D edge contribution (lumped)
for node in boundary_nodes:
    C_gamma_vals.append(1.0)  # weight for lumped boundary
    rows_C.append(node)
    cols_C.append(node)

C_gamma = sp.coo_matrix((C_gamma_vals, (rows_C, cols_C)), shape=(total_nodes, total_nodes)).tocsr()






A = K_global - k**2 * M_global - 1j*k*(eta)*C_gamma
A = A.tocsr()

# -------------------------------
# Boundary conditions
# 1 = free, 0 = fixed
bc_flags = np.ones(total_nodes)
# Suppose you want a vertical wall in the middle
wall_x = N+1  # middle column
"""
for i, (x_node, y_node) in enumerate(coordinates):
    if x_node == 0 or x_node == 2*N or y_node == 0 or y_node == 2*M:# or (x_node == wall_x and y_node<M):
        bc_flags[i] = 0  # Fixed
"""


# -------------------------------
# Force vector
# Apply at the center node
center_node = np.argmin(np.sum((coordinates - np.array([N, M]))**2, axis=1))
F = np.zeros(total_nodes)
F[((2*N+1)**2)//2] = 1 

# -------------------------------
# Free and fixed nodes
free_nodes = np.where(bc_flags==1)[0]
fixed_nodes = np.where(bc_flags==0)[0]

# -------------------------------
# Reduced system
A_reduced = A[np.ix_(free_nodes, free_nodes)]
F_reduced = F[free_nodes]

# Solve
import scipy.sparse.linalg as spla

#solution_free,info = spla.bicgstab(A_reduced, F_reduced, rtol = 1e-5, maxiter=100)

### Pytorch implementation
# -------------------------------
"""
A_reduced = torch.tensor(A_reduced)
F_reduced = torch.tensor(F_reduced)
solution_free = torch.linalg.solve(A_reduced,F_reduced)
"""

# Scipy Sparse implementation
# -------------------------------
print('framme til solver')
time_solver = time.time()
tid_til_solver = time_solver - time_beginning
print('time to solver is ',tid_til_solver)

A_reduced = sp.csr_matrix(A_reduced)
solution_free = sp.linalg.spsolve(A_reduced, F_reduced)

# Full solution including zeros at fixed nodes
solution_full = np.zeros(total_nodes)
solution_full[free_nodes] = solution_free
solution_full[fixed_nodes] = 0

# -------------------------------
# Build solution matrix for plotting
solution_matrix = np.zeros((ny_nodes, nx_nodes))
for node, (i, j) in enumerate(coordinates):
    solution_matrix[j, i] = solution_full[node]
time_end = time.time()
print('Time total is',time_end-time_beginning)
print('Time solving is',time_end-time_solver)
# Plot
"""
plt.imshow(np.abs(solution_matrix), origin='lower', cmap='viridis', extent=[0-1, nx_nodes, 0, ny_nodes-1])
plt.colorbar(label='Solution')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('FEM Solution (Q9)')
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt

def helmholtz_solution(X, Y, k, N=100):
    """
    Helmholtz solution with Neumann (reflective) BCs
    and point source at center (0.5, 0.5).
    """

    u = np.zeros_like(X, dtype=float)

    for m in range(0, N, 2):   # even modes only (center symmetry)
        for n in range(0, N, 2):
            if m == 0 and n == 0:
                continue

            lam = np.pi**2 * (m*m + n*n) - k**2

            # avoid resonance blow-up
            if abs(lam) < 1e-8:
                continue

            alpha_m = 1 if m == 0 else 2
            alpha_n = 1 if n == 0 else 2

            sign_m = (-1)**(m//2)
            sign_n = (-1)**(n//2)

            coeff = (alpha_m * alpha_n * sign_m * sign_n) / lam

            u += coeff * np.cos(m*np.pi*X) * np.cos(n*np.pi*Y)

    return u

def solve_analytical(x,y,k,x0,y0,N=100):
    # Green's function
    G = np.zeros_like(X, dtype=float)

    # Number of modes
    M = 100

    # Normalization factor for Neumann cosine basis
    def c(m):
        return 1.0 if m == 0 else np.sqrt(2.0)

    for m in range(M):
        for n in range(M):

            # Skip constant mode (Neumann nullspace / compatibility fix)
            if m == 0 and n == 0:
                continue

            lam = np.pi**2 * (m**2 + n**2)

            # Avoid resonance (division by zero)
            if abs(k**2 - lam) < 1e-10:
                continue

            cm = c(m)
            cn = c(n)

            # Properly normalized eigenfunctions
            phi_xy = cm * cn * np.cos(m * np.pi * X) * np.cos(n * np.pi * Y)
            phi_x0 = cm * cn * np.cos(m * np.pi * x0) * np.cos(n * np.pi * y0)

            G += (phi_xy * phi_x0) / (k**2 - lam)

    # Optional: remove any residual mean (numerical drift)
    G -= np.mean(G)
    return G


n = 321
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

k = 2*np.pi/(343) * 3000  # 👈 adjust this
G = solve_analytical(X, Y, k, 0.5, 0.5, N = 200)

U = helmholtz_solution(X, Y, k, N=200)



fig, axes = plt.subplots(1, 3, figsize=(20, 5))

absmax = max(
    #np.max(np.abs(solution_matrix)),
    np.max(np.abs(U)),
    np.max(np.abs(G))
)

vmin, vmax = -absmax, absmax
error = solution_matrix-U

im0 = axes[0].imshow(solution_matrix, origin='lower',
                     extent=[0, H, 0, L],
                     cmap='seismic', vmin=vmin/10, vmax=vmax/10)

im1 = axes[1].imshow(U, origin='lower',
                     extent=[0, 1, 0, 1],
                     cmap='seismic', vmin=vmin/10, vmax=vmax/10)

im2 = axes[2].imshow(error, origin='lower',
                     extent=[0, 1, 0, 1],
                     cmap='seismic', vmin=vmin/10, vmax=vmax/10)

fig.colorbar(im2, ax=axes, fraction=0.046, pad=0.04)


plt.tight_layout()
plt.show()
