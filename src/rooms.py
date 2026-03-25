import numpy as np
import matplotlib.pyplot as plt
import pygame
from config import BLACK, BROWN, DARK_GREY, GREEN, RED, DARK_GREY_BLUE, MENU_HEIGHT_MULTI, SOUND_THRESHOLD

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
            (0, 30, 12, 1),
            (27, 30, 13, 1),
            (40, 0, 1, 35),

            (30, 55, 1, 20),
            (30, 55, 10, 1),
            (57, 55, 1, 20),
            (120, 0, 1, 45),

            (0, 75, 40, 1),
            (55, 75, 40, 1),
            (110, 75, 20, 1),
            (145, 75, 55, 1),

            (60, 75, 1, 55),
            (60, 103, 20, 1),
            (80, 90, 1, 14),
            (80, 90, 15, 1),
            (120, 75, 1, 35),
            (110, 100, 10, 1),
            (120, 140, 1, 10), 
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
            (20, 20, 40, 1),
            (20, 20, 1, 40),
            (60, 20, 1, 10),
            (60, 50, 1, 10),
            (20, 60, 41, 1),

            (140, 20, 40, 1),
            (140, 20, 1, 10),
            (140, 50, 1, 10),
            (180, 20, 1, 40),
            (140, 60, 41, 1),

            (20, 90, 40, 1),
            (20, 90, 1, 40),
            (60, 90, 1, 10),
            (60, 120, 1, 10),
            (20, 130, 41, 1),

            (140, 90, 40, 1),
            (140, 90, 1, 10),
            (140, 120, 1, 10),
            (180, 90, 1, 40),
            (140, 130, 41, 1),
        ],
        4: [
            (50, 34, 1, 34),
            (50, 34, 15, 1),
            (65, 0, 1, 35),
            (113, 0, 1, 68),
            (142, 0, 1, 68),

            (0, 68, 32, 1),
            (47, 68, 48, 1),
            (110, 68, 16, 1),
            (141, 68, 3, 1),
            (159, 68, 12, 1),

            (171, 68, 1, 50),
            (171, 100, 29, 1),
            (171, 133, 1, 27),
        ], 
        5: [
            (0, 60, 17, 1),
            (33, 60, 34, 1),
            (83, 60, 34, 1),
            (133, 60, 34, 1),
            (183, 60, 17, 1),

            (50, 0, 1, 60),
            (100, 0, 1, 60),
            (150, 0, 1, 60),

            (0, 90, 17, 1),
            (33, 90, 34, 1),
            (83, 90, 34, 1),
            (133, 90, 34, 1),
            (183, 90, 17, 1),

            (50, 90, 1, 60),
            (100, 90, 1, 60),
            (150, 90, 1, 60),
        ], 
        6: [
            (0, 50, 40, 1),
            (55, 50, 5, 1),
            (0, 80, 40, 1),
            (55, 80, 5, 1),

            (60, 0, 1, 51),
            (60, 80, 1, 70),
        ],
        7: [
            (0, 100, 5, 1),
            (20, 100, 20, 1),
            (30, 100, 1, 50),
            (40, 80, 1, 21),
            (40, 80, 20, 1),
            (75, 80, 40, 1),
            (115, 80, 1, 70),

            (55, 78, 1, 2),
            (55, 0, 1, 62),
            (55, 59, 20, 1),
            (90, 59, 25, 1),
            (115, 0, 1, 60),
        ],
        8: [
            (0, 75, 45, 1),
            (90, 0, 1, 55),
            (90, 55, 45, 1),
            (135, 15, 1, 41),

            (135, 70, 65, 1),
            (135, 86, 1, 64),
            (90, 117, 1, 33),
            (90, 117, 30, 1),
        ],
        9: [
        ],
    }
    if room_id not in room_layouts:
        raise ValueError(f"Invalid room ID {room_id}")
    return create_obstacle_mask_from_list(room_layouts[room_id], x_dim=x_dim, y_dim=y_dim)

rooms = {
    1: get_room_mask(1, 200, 150), # Complex room with various obstacles, walls and corridors
    2: get_room_mask(2, 200, 150), # Room divided into 3x4 grid of smaller rooms
    3: get_room_mask(3, 200, 150), # 4 square rooma with a door in the corners of the larger room
    4: get_room_mask(4, 200, 150), # Livingroom with 3 bedrooms, 1 bathroom and entrance 
    5: get_room_mask(5, 200, 150), # Corridor with 4 rooms on each side, with doors
    6: get_room_mask(6, 200, 150), 
    7: get_room_mask(7, 200, 150),
    8: get_room_mask(8, 200, 150), 
    9: get_room_mask(9, 200, 150), # Empty room, no obstacles, just borders
}

def create_room_heatmap(obstacle_mask, sound_pressure=None):
    x_dim, y_dim = obstacle_mask.shape
    heatmap = np.full((x_dim, y_dim, 3), 243, dtype=np.uint8)

    if sound_pressure is not None:
        free_mask = ~obstacle_mask
        if np.any(free_mask):
            free_values = sound_pressure[free_mask]
            normalized_sound_pressure = np.clip((sound_pressure - free_values.min()) / (free_values.max() - free_values.min()), 0.0, 1.0)

            heatmap = (plt.cm.viridis(normalized_sound_pressure)[:, :, :3] * 255)

            covered = (sound_pressure >= SOUND_THRESHOLD) & free_mask
            below_threshold = (~covered) & free_mask
            bg = np.array(DARK_GREY_BLUE, dtype=np.float32)
            heatmap[below_threshold] = (
                0.45 * heatmap[below_threshold].astype(np.float32) + 0.55 * bg
            ).astype(np.uint8)

    heatmap[obstacle_mask] = np.array(DARK_GREY, dtype=np.uint8)
    return heatmap

def draw_room(screen, room_number, color=RED, scale=5, sound_pressure=None, alarms=None):
    top_margin = int(MENU_HEIGHT_MULTI * screen.get_height())
    menu_offset = int(MENU_HEIGHT_MULTI * screen.get_height())
    target_rect = pygame.Rect(screen.get_width()*0.05, top_margin*1.1, screen.get_width() - screen.get_width()*0.1, screen.get_height() - menu_offset - top_margin*1.2)
    obstacle_mask = rooms[room_number]

    heatmap = create_room_heatmap(obstacle_mask, sound_pressure=sound_pressure)
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
            font = pygame.font.SysFont(None, 20)
            text_surf = font.render('(' + str(ax) + ', ' + str(150-ay) + ')', True, RED)
            text_rect = pygame.Rect(px - text_surf.get_width() // 2, py + 12, text_surf.get_width(), text_surf.get_height())
            screen.blit(text_surf, text_rect)

def room_showcase(screen, room_number, left, top):
    menu_offset = int(MENU_HEIGHT_MULTI * screen.get_height())
    target_rect = pygame.Rect(left, top+1, screen.get_width()//3, screen.get_height()//3 - menu_offset)
    obstacle_mask = rooms[room_number]

    heatmap = create_room_heatmap(obstacle_mask)
    surface = pygame.surfarray.make_surface(heatmap)
    scaled = pygame.transform.scale(surface, (target_rect.width, target_rect.height))
    font = pygame.font.SysFont(None, 40)
    text_surf = font.render(f'Room No.{room_number}', True, BROWN)
    text_rect = pygame.Rect(target_rect.centerx - text_surf.get_width() // 2, target_rect.centery - text_surf.get_height() // 2, text_surf.get_width(), text_surf.get_height())
    screen.blit(scaled, target_rect)
    screen.blit(text_surf, text_rect) # the center of the room rect
