import numpy as np
import matplotlib.pyplot as plt
import pygame
from config import *

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

# def FDM_laplace(u, n, m):
#     delta_u = np.zeros((n+1,m+1))
#     for i in range(1,n):
#         for j in range(1,m):
#             delta_u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) / 4 - u[i,j]
#             u[i,j] = u[i,j] + delta_u[i,j]
#     return u

class Wall:
    def __init__(self, x_start, x_end, y_start, y_end):
        n = 100
        m = 100
        Lx = 10.0
        Ly = 20.0
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.slice_x = slice(x_start, x_end)
        self.slice_y = slice(y_start, y_end)

    def contains(self, x, y):
        return self.x_start <= x <= self.x_end and self.y_start <= y <= self.y_end

def FDM_laplace(tol=5e-4, max_iter=20000):

    # Constants
    Lx = 10.0 # Length in x-direction
    Ly = 20.0 # Length in y-direction

    n = 100
    m = 100

    dx = Lx / n
    dy = Ly / m

    u_max = 200.0

    u = np.zeros((n+1, m+1))  # initial guess
    walls = np.zeros((n+1, m+1), dtype=bool)  # wall mask for interior obstacles

    # Boundary conditions
    u[:, 0] = 0.0  # u(x,0) = 0
    u[:, m] = 0.0  # u(x,Ly) = 0
    u[0, :] = 0.0  # u(0,y) = 0
    u[n, :] = 0.0  # u(Lx,y) = 0
    u[n//2, m//2] = u_max

    # Walls
    walls = [Wall(n//4, 3*n//4, m//3, m//3 + 2),
             Wall(n//3, n//3 + 2, m//2, 3*m//4),
             Wall(n//2, 3*n//4, m//4, m//4 + 2)]
    wall_pixels = np.zeros((n+1, m+1), dtype=bool)
    for wall in walls:
        u[wall.slice_x, wall.slice_y] = 0.0
        wall_pixels[wall.slice_x, wall.slice_y] = True
        print(f"Wall slice x: {wall.slice_x}, slice y: {wall.slice_y}")
    # wall_pixels[40:60, 5:8] = True
    # wall_pixels[30:35, 40:70] = True
    # wall_pixels[30:70, 35:40] = True

    # Propagation/Calculation
    frame = 0
    c = 1.0 / (2.0 * (dx**2 + dy**2))
    u_animation = np.zeros((u.shape[0], u.shape[1], max_iter // 10))
    u_animation[:,:,frame] = u.copy()
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(1, n):
            for j in range(1, m):
                # Skip points inside walls
                if wall_pixels[i,j]:
                    u[i,j] = 0.0
                    continue
                    
                # Check if neighbors are walls and apply reflection
                left = u[i-1,j] if not wall_pixels[i-1,j] else u[i+1,j]
                right = u[i+1,j] if not wall_pixels[i+1,j] else u[i-1,j]
                down = u[i,j-1] if not wall_pixels[i,j-1] else u[i,j+1]
                up = u[i,j+1] if not wall_pixels[i,j+1] else u[i,j-1]
                
                u_old = u[i,j]
                u[i,j] = ((right + left) * dy**2 + (up + down) * dx**2) * c
                diff = abs(u[i,j] - u_old)
                if diff > max_diff:
                    max_diff = diff
                    
        if max_diff < tol:
            print(f"Converged after {it + 1} iterations with max update {max_diff}")
            u_animation = u_animation[:,:,:frame+1]
            return u_animation, walls, frame + 1, it + 1, max_diff
        if it % 10 == 0:
            frame += 1
            u_animation[:,:,frame] = u.copy()
            print(f"Iteration {it}: max update {max_diff}")

        
    print(f"Did not converge after maximum iterations ({max_iter}) with max update {max_diff}")
    u_animation = u_animation[:,:,:frame+1]
    return u_animation, walls, frame + 1, max_iter, max_diff

def draw_frame(screen, u, walls, frame, normalisation=False):
    if u is None:
        return

    if u.ndim == 2:
        u_frame = u
    else:
        frame_index = frame % u.shape[2]
        u_frame = u[:, :, frame_index]

    if np.all(u_frame == 0):
        return

    vmin = float(np.min(u_frame))
    vmax = float(np.max(u_frame))

    u_frame = np.flipud(np.where(u_frame < 0.01, 0, u_frame).T)
    rgb = np.zeros((u_frame.shape[0], u_frame.shape[1], 3), dtype=np.uint8)
    for i in range(u_frame.shape[0]):
        for j in range(u_frame.shape[1]):
            value = u_frame[i,j]
            cmap_value = int(255 * value / vmax) if vmax != 0 else 0
            rgb[i,j] = [cmap_value, cmap_value, cmap_value]
    

    surface = pygame.surfarray.make_surface(rgb)

    target_rect = screen.get_rect().copy()
    target_rect.height = max(1, target_rect.height - 50)
    surface = pygame.transform.smoothscale(surface, (target_rect.width, target_rect.height))
    # Walls
    # for wall in walls:
        # pygame.draw.rect(surface, RED, wall.rect)
    screen.blit(surface, target_rect)

# Plotting
# x = np.linspace(0, Lx, n+1)
# y = np.linspace(0, Ly, m+1)
# X, Y = np.meshgrid(x, y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, u[:,:,-1], cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('u')
# plt.show()

# # 2D heatmap
# plt.imshow(u[:,:,-1].T, extent=(0, Lx, 0, Ly), origin='lower', cmap='viridis')
# plt.colorbar(label='u')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

