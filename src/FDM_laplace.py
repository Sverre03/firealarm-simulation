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
    u_max = 800.0

    u = np.zeros((n+1, m+1))


    # Boundary conditions
    walls = np.zeros((n+1, m+1), dtype=bool)  # wall mask for interior obstacles
    u[:, 0] = 0.0  # u(x,0) = 0
    u[:, m] = 0.0  # u(x,Ly) = 0
    u[0, :] = 0.0  # u(0,y) = 0
    u[n, :] = 0.0  # u(Lx,y) = 0
    u[n//2, m//2] = u_max

    # Walls
    wall_pixels = np.zeros((n+1, m+1), dtype=bool)

    walls = [Wall(int(0.25*n), int(0.75*n), int(0.33*m), int(0.33*m + 4)),
             Wall(int(0.75*n), int(0.75*n + 2), int(0.5*m), int(0.75*m)),
             Wall(int(0.5*n), int(0.75*n), int(0.25*m), int(0.25*m + 2))]
    
    for wall in walls:
        wall_pixels[wall.slice_x, wall.slice_y] = True
 
    c = 1.0 / (2.0 * (dx**2 + dy**2)) # constant factor in update formula

    # Propagation/Calculation
    frame = 0
    u_animation = np.zeros((u.shape[0], u.shape[1], max_iter // 10))
    u_animation[:,:,frame] = u.copy()
    for it in range(max_iter):
        u_old = u.copy()
        max_diff = 0.0

        left = u_old[0:n-1, 1:m]
        right = u_old[2:n+1, 1:m]
        down = u_old[1:n, 0:m-1]
        up = u_old[1:n, 2:m+1]

        wall_left = wall_pixels[0:n-1, 1:m]
        wall_right = wall_pixels[2:n+1, 1:m]
        wall_down = wall_pixels[1:n, 0:m-1]
        wall_up = wall_pixels[1:n, 2:m+1]

        left_eff = np.where(wall_left, right, left)
        right_eff = np.where(wall_right, left, right)
        down_eff = np.where(wall_down, up, down)
        up_eff = np.where(wall_up, down, up)

        u_new = c * ((right_eff + left_eff) * dy**2 + (up_eff + down_eff) * dx**2)
        interior_walls = wall_pixels[1:n, 1:m]
        u[1:n, 1:m] = np.where(interior_walls, 0.0, u_new)

        diff = np.abs(u - u_old)
        max_diff = np.max(diff[1:n, 1:m])

        if max_diff < tol:
            print(f"Converged after {it + 1} iterations with max update {max_diff}")
            return u_animation, walls, frame + 1, it + 1, max_diff
        if it % 10 == 0:
            frame += 1
            u_animation[:,:,frame] = u.copy()
            print(f"Iteration {it}: max update {max_diff}")

        
    print(f"Did not converge after maximum iterations ({max_iter}) with max update {max_diff}")
    u_animation = u_animation[:,:,:frame+1]
    return u_animation, walls, frame + 1, max_iter, max_diff

def draw_frame(screen, u, walls, frame, SCREEN_WIDTH, SCREEN_HEIGHT, normalisation=False, paused=False):
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

    u_frame = np.flipud(np.where(u_frame < 0.01, 0, u_frame))
    
    # Vectorized RGB conversion - much faster than nested loops!
    if vmax != 0:
        cmap_values = (255 * u_frame / vmax).astype(np.uint8)
    else:
        cmap_values = np.zeros_like(u_frame, dtype=np.uint8)
    
    # Create RGB array by stacking grayscale values
    rgb = np.stack([cmap_values, cmap_values, cmap_values], axis=2)
    
    surface = pygame.surfarray.make_surface(rgb)

    target_rect = screen.get_rect().copy()
    target_rect.height = max(1, target_rect.height - MENU_HEIGHT_MULTI * SCREEN_HEIGHT)
    surface = pygame.transform.scale(surface, (target_rect.width, target_rect.height))
    # Walls
    # for wall in walls:
        # pygame.draw.rect(surface, RED, wall.rect)
    screen.blit(surface, target_rect)

    if paused:
        font = pygame.font.SysFont(None, 48)
        text_surf = font.render("Paused", True, RED)
        text_rect = text_surf.get_rect(center=screen.get_rect().center)
        screen.blit(text_surf, text_rect)

    # # Draw grid lines
    # n, m = u_frame.shape
    # for i in range(1, n):
    #     x = i * target_rect.width // n
    #     pygame.draw.line(screen, DARK_GREY, (x, target_rect.top), (x, target_rect.bottom))
    # for j in range(1, m):
    #     y = j * target_rect.height // m
    #     pygame.draw.line(screen, DARK_GREY, (target_rect.left, y), (target_rect.right, y))

    # cell_width = target_rect.width / n
    # cell_height = target_rect.height / m
    # for wall in walls:
    #     wall_rect = pygame.Rect(
    #         target_rect.left + wall.x_start * cell_width,
    #         target_rect.top + wall.y_start * cell_height,
    #         (wall.x_end - wall.x_start) * cell_width,
    #         (wall.y_end - wall.y_start) * cell_height
    #     )
    #     pygame.draw.rect(screen, RED, wall_rect)

