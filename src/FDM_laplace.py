import numpy as np
import matplotlib.pyplot as plt
import pygame
from config import *

# Laplace's equation d^2u/dx^2 + d^2u/dy^2 = 0

# Laplace's equation assuming dx and dy constant and using central difference
# d^2u/dx^2 = (potential[i+1,j] - 2u[i,j] + potential[i-1,j]) / delta_x^2
# d^2u/dy^2 = (potential[i,j+1] - 2u[i,j] + potential[i,j-1]) / delta_y^2

# Letting delta_x = delta_y:
# => delta_u[i,j] = (potential[i+1,j] + potential[i-1,j] + potential[i,j+1] + potential[i,j-1]) / 4 - potential[i,j]

# Update potential
# potential[i,j] = potential[i,j] + delta_u[i,j]

class Source:
    def __init__(self, x, y, strength):
        self.x = int(x)
        self.y = int(y)
        self.strength = strength

class Obstacle:
    def __init__(self, x_start, x_end, y_start, y_end):
        grid_x = 100
        grid_y = 100
        Lx = 10.0
        Ly = 20.0
        self.x_start = int(x_start)
        self.x_end = int(x_end)
        self.y_start = int(y_start)
        self.y_end = int(y_end)
        self.slice_x = slice(self.x_start, self.x_end)
        self.slice_y = slice(self.y_start, self.y_end)

    def contains(self, x, y):
        return self.x_start <= x <= self.x_end and self.y_start <= y <= self.y_end

def FDM_laplace(tol=5e-4, max_iter=20000):

    # Constants
    Lx = 10.0 # Length in x-direction
    Ly = 20.0 # Length in y-direction
    grid_x = 100
    grid_y = 100
    dx = Lx / grid_x
    dy = Ly / grid_y
    source_strength = 800.0

    potential = np.zeros((grid_x+1, grid_y+1))


    # Boundary conditions
    obstacles = np.zeros((grid_x+1, grid_y+1), dtype=bool)  # obstacle mask for interior obstacles
    potential[:, 0] = 0.0  # potential(x,0) = 0
    potential[:, grid_y] = 0.0  # potential(x,Ly) = 0
    potential[0, :] = 0.0  # potential(0,y) = 0
    potential[grid_x, :] = 0.0  # potential(Lx,y) = 0

    # Obstacles
    obstacle_mask = np.zeros((grid_x+1, grid_y+1), dtype=bool)

    obstacles = [Obstacle(0.25*grid_x, 0.75*grid_x, 0.33*grid_y, 0.33*grid_y + 4),
             Obstacle(0.75*grid_x, 0.75*grid_x + 2, 0.5*grid_y, 0.75*grid_y),
             Obstacle(0.5*grid_x, 0.75*grid_x, 0.25*grid_y, 0.25*grid_y + 2)]
    
    for obstacle in obstacles:
        obstacle_mask[obstacle.slice_x, obstacle.slice_y] = True

    # Sources
    sources = [Source(0.5*grid_x, 0.5*grid_y, source_strength), 
               Source(0.15*grid_x, 0.25*grid_y, 0.5*source_strength), 
               Source(0.8*grid_x, 0.25*grid_y, 0.5*source_strength)]
 
    denom = 2.0 / dx**2 + 2.0 / dy**2  # factor for dx != dy

    # Propagation/Calculation
    frame_index = 0
    potential_frames = np.zeros((potential.shape[0], potential.shape[1], max_iter // 10))
    potential_frames[:,:,frame_index] = potential.copy()
    for iteration in range(max_iter):
        potential_old = potential.copy()
        update = 0.0

        neighbor_left = potential_old[0:grid_x-1, 1:grid_y]
        neighbor_right = potential_old[2:grid_x+1, 1:grid_y]
        neighbor_down = potential_old[1:grid_x, 0:grid_y-1]
        neighbor_up = potential_old[1:grid_x, 2:grid_y+1]

        obstacle_left_mask = obstacle_mask[0:grid_x-1, 1:grid_y]
        obstacle_right_mask = obstacle_mask[2:grid_x+1, 1:grid_y]
        obstacle_down_mask = obstacle_mask[1:grid_x, 0:grid_y-1]
        obstacle_up_mask = obstacle_mask[1:grid_x, 2:grid_y+1]

        left_effective = np.where(obstacle_left_mask, neighbor_right, neighbor_left) # Alle steder hvor vi har en vegg til venstre, bruker vi verdien til høyre for veggen, eller bruker vi verdien til venstre
        right_effective = np.where(obstacle_right_mask, neighbor_left, neighbor_right)
        down_effective = np.where(obstacle_down_mask, neighbor_up, neighbor_down)
        up_effective = np.where(obstacle_up_mask, neighbor_down, neighbor_up)

        potential_new = (1.0 / denom) * ((right_effective + left_effective) / dx**2 + (up_effective + down_effective) / dy**2)
        interior_obstacles = obstacle_mask[1:grid_x, 1:grid_y]
        potential[1:grid_x, 1:grid_y] = np.where(interior_obstacles, 0.0, potential_new)
        for source in sources:
            potential[source.x, source.y] = source.strength

        delta = np.abs(potential - potential_old)
        update = np.max(delta[1:grid_x, 1:grid_y])

        if update < tol:
            print(f"Converged after {iteration + 1} iterations with max update {update}")
            return potential_frames, obstacles, frame_index + 1, iteration + 1, update
        if iteration % 10 == 0:
            frame_index += 1
            potential_frames[:,:,frame_index] = potential.copy()
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: max update {update}")

        
    print(f"Did not converge after maximum iterations ({max_iter}) with max update {update}")
    potential_frames = potential_frames[:,:,:frame_index+1]
    return potential_frames, obstacles, frame_index + 1, max_iter, update

def draw_frame(screen, potential, obstacles, frame, SCREEN_WIDTH, SCREEN_HEIGHT, paused=False):
    if potential is None:
        return 0.0

    if potential.ndim == 2:
        potential_frame = potential
    else:
        frame_index = frame % potential.shape[2]
        potential_frame = potential[:, :, frame_index]

    if np.all(potential_frame == 0):
        return 0.0


    potential_frame = np.flipud(np.where(potential_frame < 0.1, 0, potential_frame))

    covered = potential_frame > 0
    total_cells = covered.size
    covered_cells = np.sum(covered)

    percentage = 100 * covered_cells / total_cells
    
    potential_max = float(np.max(potential_frame))
    color_map_values = (255 * potential_frame / potential_max).astype(np.uint8)
    
    rgb = convert_to_rgb(color_map_values, color_map="rainbow")
    
    surface = pygame.surfarray.make_surface(rgb)

    target_rect = screen.get_rect().copy()
    target_rect.height = max(1, target_rect.height - MENU_HEIGHT_MULTI * SCREEN_HEIGHT)
    surface = pygame.transform.scale(surface, (target_rect.width, target_rect.height))
  
    screen.blit(surface, target_rect)

    if paused:
        font = pygame.font.SysFont(None, 48)
        text_surf = font.render("Paused", True, RED)
        text_rect = text_surf.get_rect(center=screen.get_rect().center)
        screen.blit(text_surf, text_rect)

    return percentage

def convert_to_rgb(values, color_map="rainbow"):
    if color_map == "rainbow":
        cmap = plt.get_cmap("rainbow")
    elif color_map == "hot":
        cmap = plt.get_cmap("hot")
    else:
        cmap = plt.get_cmap("viridis")

    normalised_values = values / 255.0
    rgba_colors = cmap(normalised_values)
    rgb_colors = (rgba_colors[:, :, :3] * 255).astype(np.uint8)
    zero_mask = values == 0
    rgb_colors = np.where(zero_mask[..., None], 0, rgb_colors)
    return rgb_colors
