import numpy as np
import matplotlib.pyplot as plt
import pygame
from config import *

# Laplace's equation d^2u/self.dx^2 + d^2u/self.dy^2 = 0

# Laplace's equation assuming self.dx and self.dy constant and using central difference
# d^2u/self.dx^2 = (self.potential[i+1,j] - 2u[i,j] + self.potential[i-1,j]) / delta_x^2
# d^2u/self.dy^2 = (self.potential[i,j+1] - 2u[i,j] + self.potential[i,j-1]) / delta_y^2

# Letting delta_x = delta_y:
# => delta_u[i,j] = (self.potential[i+1,j] + self.potential[i-1,j] + self.potential[i,j+1] + self.potential[i,j-1]) / 4 - self.potential[i,j]

# Update self.potential
# self.potential[i,j] = self.potential[i,j] + delta_u[i,j]

class Source:
    def __init__(self, x, y, strength):
        self.x = int(x)
        self.y = int(y)
        self.strength = strength

class Obstacle:
    def __init__(self, x_start, x_end, y_start, y_end):
        self.grid_x = 100
        self.grid_y = 100
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
    
class Room:
    def __init__(self):
        # Constants
        Lx = 10.0 # Length in x-direction
        Ly = 20.0 # Length in y-direction
        self.grid_x = 100
        self.grid_y = 100
        self.dx = Lx / self.grid_x
        self.dy = Ly / self.grid_y
        self.source_strength = 800.0

        self.potential = np.zeros((self.grid_x+1, self.grid_y+1))

        self.obstacle_mask = np.zeros((self.grid_x+1, self.grid_y+1), dtype=bool)  # obstacle mask for interior obstacles
        self.obstacles = [Obstacle(0.25*self.grid_x, 0.75*self.grid_x, 0.33*self.grid_y, 0.33*self.grid_y + 4),
                Obstacle(0.75*self.grid_x, 0.75*self.grid_x + 2, 0.5*self.grid_y, 0.75*self.grid_y),
                Obstacle(0.5*self.grid_x, 0.75*self.grid_x, 0.25*self.grid_y, 0.25*self.grid_y + 2)]
        
        self.sources = [Source(0.5*self.grid_x, 0.5*self.grid_y, self.source_strength), 
                Source(0.15*self.grid_x, 0.25*self.grid_y, 0.5*self.source_strength), 
                Source(0.8*self.grid_x, 0.25*self.grid_y, 0.5*self.source_strength)]

    def calculate_potential(self, tol=5e-4, max_iter=20000):
        # Boundary conditions
        self.potential[:, 0] = 0.0  # self.potential(x,0) = 0
        self.potential[:, self.grid_y] = 0.0  # self.potential(x,Ly) = 0
        self.potential[0, :] = 0.0  # self.potential(0,y) = 0
        self.potential[self.grid_x, :] = 0.0  # self.potential(Lx,y) = 0
        
        for obstacle in self.obstacles:
            self.obstacle_mask[obstacle.slice_x, obstacle.slice_y] = True
    
        denom = 2.0 / self.dx**2 + 2.0 / self.dy**2  # factor for self.dx != self.dy

        # Propagation/Calculation
        frame_index = 0
        self.potential_frames = np.zeros((self.potential.shape[0], self.potential.shape[1], max_iter // 10))
        self.potential_frames[:,:,frame_index] = self.potential.copy()
        for iteration in range(max_iter):
            self.potential_old = self.potential.copy()
            update = 0.0

            neighbor_left = self.potential_old[0:self.grid_x-1, 1:self.grid_y]
            neighbor_right = self.potential_old[2:self.grid_x+1, 1:self.grid_y]
            neighbor_down = self.potential_old[1:self.grid_x, 0:self.grid_y-1]
            neighbor_up = self.potential_old[1:self.grid_x, 2:self.grid_y+1]

            obstacle_left_mask = self.obstacle_mask[0:self.grid_x-1, 1:self.grid_y]
            obstacle_right_mask = self.obstacle_mask[2:self.grid_x+1, 1:self.grid_y]
            obstacle_down_mask = self.obstacle_mask[1:self.grid_x, 0:self.grid_y-1]
            obstacle_up_mask = self.obstacle_mask[1:self.grid_x, 2:self.grid_y+1]

            left_effective = np.where(obstacle_left_mask, neighbor_right, neighbor_left) # Alle steder hvor vi har en vegg til venstre, bruker vi verdien til høyre for veggen, eller bruker vi verdien til venstre
            right_effective = np.where(obstacle_right_mask, neighbor_left, neighbor_right)
            down_effective = np.where(obstacle_down_mask, neighbor_up, neighbor_down)
            up_effective = np.where(obstacle_up_mask, neighbor_down, neighbor_up)

            self.potential_new = (1.0 / denom) * ((right_effective + left_effective) / self.dx**2 + (up_effective + down_effective) / self.dy**2)
            interior_obstacles = self.obstacle_mask[1:self.grid_x, 1:self.grid_y]
            self.potential[1:self.grid_x, 1:self.grid_y] = np.where(interior_obstacles, 0.0, self.potential_new)
            for source in self.sources:
                self.potential[source.x, source.y] = source.strength

            delta = np.abs(self.potential - self.potential_old)
            update = np.max(delta[1:self.grid_x, 1:self.grid_y])

            if update < tol:
                print(f"Converged after {iteration + 1} iterations with max update {update}")
                self.potential = self.potential_frames
                return self.potential_frames, frame_index + 1, iteration + 1, update
            if iteration % 10 == 0:
                frame_index += 1
                self.potential_frames[:,:,frame_index] = self.potential.copy()
                if iteration % 1000 == 0:
                    print(f"Iteration {iteration}: max update {update}")

            
        print(f"Did not converge after maximum iterations ({max_iter}) with max update {update}")
        self.potential_frames = self.potential_frames[:,:,:frame_index+1]
        self.potential = self.potential_frames
        return self.potential_frames, frame_index + 1, max_iter, update

    def draw_frame(self, screen, frame, SCREEN_WIDTH, SCREEN_HEIGHT, paused=False):
        if self.potential is None:
            return 0.0

        if self.potential.ndim == 2:
            self.potential_frame = self.potential
        elif self.potential.ndim == 3:
            frame_index = frame % self.potential.shape[2]
            self.potential_frame = self.potential[:, :, frame_index]
        else:
            self.potential_frame = self.potential

        if np.all(self.potential_frame == 0):
            return 0.0


        self.potential_frame = np.flipud(np.where(self.potential_frame < 0.1, 0, self.potential_frame))

        covered = self.potential_frame > 0
        total_cells = covered.size
        covered_cells = np.sum(covered)

        percentage = 100 * covered_cells / total_cells
        
        self.potential_max = float(np.max(self.potential_frame))
        color_map_values = (255 * self.potential_frame / self.potential_max).astype(np.uint8)
        
        rgb = convert_to_rgb(color_map_values, color_map="rainbow")
        
        surface = pygame.surfarray.make_surface(rgb)

        target_rect = screen.get_rect().copy()
        target_rect.height = max(1, target_rect.height - MENU_HEIGHT_MULTI * SCREEN_HEIGHT)
        surface = pygame.transform.scale(surface, (target_rect.width, target_rect.height))
    
        screen.blit(surface, target_rect)

        for source in self.sources:
            source_x = int((self.grid_x - source.x + 0.5) * target_rect.width / (self.grid_x + 1)) + target_rect.x
            source_y = int((source.y + 0.5) * target_rect.height / (self.grid_y + 1)) + target_rect.y
            pygame.draw.circle(screen, GREEN, (source_x, source_y), 5)
            # Draw coordinates for sources
            font = pygame.font.SysFont(None, 24)
            text_surf = font.render(f"({source.x}, {source.y})", True, DARK_GREEN)
            screen.blit(text_surf, (source_x + 10, source_y + 10))

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
