import numpy as np
import pygame
from FDM_laplace import Obstacle

# Outputs a 2D obstacle mask
def create_obstacle_mask(obstacles, x_dim=100, y_dim=100):
    obstacle_mask = np.zeros((x_dim, y_dim), dtype=bool)
    for obstacle in obstacles:
        obstacle_mask[obstacle.slice_x, obstacle.slice_y] = True
    return obstacle_mask

def create_obstacle_mask_from_list(obstacle_list, x_dim=100, y_dim=100):
    obstacles = [create_obstacle(*params) for params in obstacle_list]
    left_wall = Obstacle(0, 1, 0, y_dim)
    top_wall = Obstacle(0, x_dim, 0, 1)
    right_wall = Obstacle(x_dim-1, x_dim, 0, y_dim)
    bottom_wall = Obstacle(0, x_dim, y_dim-1, y_dim)
    obstacles.extend([left_wall, top_wall, right_wall, bottom_wall])
    return create_obstacle_mask(obstacles, x_dim, y_dim)

def create_obstacle(x, y, width, height):
    return Obstacle(x, x + width, y, y + height)

rooms = {
    1 : create_obstacle_mask_from_list([
        (25, 25, 40, 1), # horizontal top wall of the room part 1
        (70, 25, 5, 1), # horizontal top wall of the room part 2
        (75, 25, 1, 50), # rightmost vertical wall of the room
        (25, 26, 1, 50), # leftmost vertical wall of the room
        (26, 75, 50, 1), # horizontal bottom wall of the room part 1
    ]),
    2 : create_obstacle_mask_from_list([
        (0, 25, 40, 1), # horizontal top wall of the room part 1
        (45, 25, 30, 1), # horizontal top wall of the room part 2
    ]),
}

def draw_room(screen, obstacle_mask, color=(200, 0, 0)):
    for x in range(obstacle_mask.shape[0]):
        for y in range(obstacle_mask.shape[1]):
            if obstacle_mask[x, y]:
                pygame.draw.rect(screen, color, (x*5, y*5, 5, 5))

screen = pygame.display.set_mode((500, 500))
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    draw_room(screen, rooms[2])
    pygame.display.flip()