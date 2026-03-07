import numpy as np
import matplotlib.pyplot as plt
import pygame
from config import BLACK, GREEN, RED, DARK_GREY_BLUE, MENU_HEIGHT_MULTI

class Obstacle:
    def __init__(self, x_start, x_end, y_start, y_end):
        self.x_start = int(x_start)
        self.x_end = int(x_end)
        self.y_start = int(y_start)
        self.y_end = int(y_end)
        self.slice_x = slice(self.x_start, self.x_end)
        self.slice_y = slice(self.y_start, self.y_end)

    def contains(self, x, y):
        return self.x_start <= x <= self.x_end and self.y_start <= y <= self.y_end

def create_obstacle(x, y, width, height):
    return Obstacle(x, x + width, y, y + height)

def border_obstacles(x_dim, y_dim):
    return [
        Obstacle(0, 1, 0, y_dim),
        Obstacle(0, x_dim, 0, 1),
        Obstacle(x_dim - 1, x_dim, 0, y_dim),
        Obstacle(0, x_dim, y_dim - 1, y_dim),
    ]

def create_obstacle_mask(obstacles, x_dim=200, y_dim=150):
    obstacle_mask = np.zeros((x_dim, y_dim), dtype=bool)
    for obstacle in obstacles:
        obstacle_mask[obstacle.slice_x, obstacle.slice_y] = True
    return obstacle_mask

def create_obstacle_mask_from_list(obstacle_list, x_dim=200, y_dim=150):
    obstacles = [create_obstacle(*params) for params in obstacle_list]
    obstacles.extend(border_obstacles(x_dim, y_dim))
    return create_obstacle_mask(obstacles, x_dim, y_dim)

def get_room_mask(room_id=1, x_dim=200, y_dim=150):
    room_layouts = {
        1: [
            (25, 25, 40, 1),
            (70, 25, 5, 1),
            (75, 25, 1, 50),
            (25, 26, 1, 50),
            (26, 75, 50, 1),
        ],
        # Room divided into 3x4 grid of smaller rooms
        2: [
            # Vertical walls
            (0, 50, 200, 1),
            (0, 100, 200, 1),
            # Horizontal walls
            (50, 0, 1, 150),
            (100, 0, 1, 150),
            (150, 0, 1, 150),
        ],

        3: [
            (20, 20, 60, 1),
            (20, 20, 1, 60),
            (80, 20, 1, 60),
            (20, 80, 61, 1),
        ],
    }
    if room_id not in room_layouts:
        raise ValueError(f"Invalid room ID {room_id}")
    return create_obstacle_mask_from_list(room_layouts[room_id], x_dim=x_dim, y_dim=y_dim)

rooms = {
    1: get_room_mask(1, 200, 150), # Square room with a door in the middle of the larger room
    2: get_room_mask(2, 200, 150), # Room divided into 3x4 grid of smaller rooms
    3: get_room_mask(3, 200, 150), # Square room without a door in the middle of the larger room
}

def create_room_heatmap(obstacle_mask, potential=None, threshold=100.0):
    x_dim, y_dim = obstacle_mask.shape
    heatmap = np.full((x_dim, y_dim, 3), 20, dtype=np.uint8)

    if potential is not None:
        free_mask = ~obstacle_mask
        max_potential = float(np.max(potential[free_mask])) if np.any(free_mask) else 0.0
        if max_potential > 0:
            normalized_potential = np.clip(potential / max_potential, 0.0, 1.0)

            # Create heatmap
            heat = np.zeros_like(heatmap)
            heat = plt.cm.viridis(normalized_potential)[:, :, :3] * 255

            covered = (potential >= threshold) & free_mask # Above threshold
            below_threshold = (~covered) & free_mask

            heatmap[covered] = heat[covered]
            heatmap[below_threshold] = np.array(DARK_GREY_BLUE, dtype=np.uint8)

    heatmap[obstacle_mask] = np.array(RED, dtype=np.uint8)
    return heatmap

def draw_room(screen, room_number, color=RED, scale=5, potential=None, alarms=None, threshold=100.0):

    menu_offset = int(MENU_HEIGHT_MULTI * screen.get_height())
    target_rect = pygame.Rect(0, menu_offset, screen.get_width(), screen.get_height() - menu_offset)
    obstacle_mask = rooms[room_number]

    if potential is None:
        for x in range(obstacle_mask.shape[0]):
            for y in range(obstacle_mask.shape[1]):
                if obstacle_mask[x, y]:
                    pygame.draw.rect(screen, color, (x * scale, y * scale, scale, scale))
        if alarms:
            for ax, ay in alarms:
                pygame.draw.circle(screen, RED, (int(ax * scale), int(ay * scale)), max(3, scale // 2))
        return

    heatmap = create_room_heatmap(obstacle_mask, potential=potential, threshold=threshold)
    surface = pygame.surfarray.make_surface(heatmap)
    scaled = pygame.transform.scale(surface, (target_rect.width, target_rect.height))
    screen.blit(scaled, target_rect)

    if alarms:
        x_dim, y_dim = obstacle_mask.shape
        for ax, ay in alarms:
            px = int((ax + 0.5) * target_rect.width / x_dim) + target_rect.x
            py = int((ay + 0.5) * target_rect.height / y_dim) + target_rect.y
            pygame.draw.circle(screen, RED, (px, py), 6)
            pygame.draw.circle(screen, BLACK, (px, py), 6, width=1)