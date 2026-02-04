import numpy as np
import matplotlib.pyplot as plt

# Laplace's equation d^2u/dx^2 + d^2u/dy^2 = 0

# Laplace's equation assuming dx and dy constant and using central difference
# d^2u/dx^2 = (u[i+1,j] - 2u[i,j] + u[i-1,j]) / delta_x^2
# d^2u/dy^2 = (u[i,j+1] - 2u[i,j] + u[i,j-1]) / delta_y^2

# Letting delta_x = delta_y:
# => delta_u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) / 4 - u[i,j]

# Update u
# u[i,j] = u[i,j] + delta_u[i,j]

# Boundary conditions
# 0 <= x <= L
# 0 <= y <= L
# u(0,y) = 0
# u(x,0) = 0
# u(x,L) = 0
# u(L,y) = u_max * np.sin(np.pi * y / L)
# u[i,j] = 0 # n = 0

Lx = 10.0
Ly = 20.0

n = 100
m = 100

dx = Lx / n
dy = Ly / m

u_max = 20.0

u = np.zeros((n + 1, m + 1))  # initial guess

# Boundary conditions
u[:, 0] = 0.0  # u(x,0) = 0
u[:, m] = 0.0  # u(x,Ly) = 0
u[0, :] = 0.0  # u(0,y) = 0
u[n, :] = u_max * np.sin(np.pi * np.linspace(0, Ly, m + 1) / Ly)  # u(Lx,y)

# def FDM_laplace(u, n, m):
#     delta_u = np.zeros((n+1,m+1))
#     for i in range(1,n):
#         for j in range(1,m):
#             delta_u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) / 4 - u[i,j]
#             u[i,j] = u[i,j] + delta_u[i,j]
#     return u

def FDM_laplace(u, n, m, dx, dy, tol=1e-3, max_iter=20000):
    c = 1.0 / (2.0 * (dx**2 + dy**2))
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(1, n):
            for j in range(1, m):
                u_old = u[i, j]
                u[i,j] = ((u[i+1,j] + u[i-1,j]) * dy**2 + (u[i,j+1] + u[i,j-1]) * dx**2) * c
                diff = abs(u[i, j] - u_old)
                if diff > max_diff:
                    max_diff = diff
        if max_diff < tol:
            print(f"Converged after {it + 1} iterations with max update {max_diff}")
            return u, it + 1, max_diff
        if it % 1000 == 0:
            print(f"Iteration {it}: max update {max_diff}")
        
    print(f"Did not converge after maximum iterations ({max_iter}) with max update {max_diff}")
    return u, max_iter, max_diff

u, iterations, last_update = FDM_laplace(u, n, m, dx, dy)

# Plotting
x = np.linspace(0, Lx, n+1)
y = np.linspace(0, Ly, m+1)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()

# 2D heatmap
plt.imshow(u.T, extent=(0, Lx, 0, Ly), origin='lower', cmap='viridis')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('y')
plt.show()