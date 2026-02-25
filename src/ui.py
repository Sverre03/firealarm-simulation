
import pygame
from config import *
from buttons import *

class Menu:
    def __init__(self, x, y, SCREEN_WIDTH, SCREEN_HEIGHT, color, text = 'Menu'):
        self.menu_height = MENU_HEIGHT_MULTI * SCREEN_HEIGHT
        self.rect = pygame.Rect(x, y, SCREEN_WIDTH, self.menu_height)
        self.color = color
        self.text = text
        self.font = pygame.font.SysFont(None, 24)

        self.quit_button = Switch(SCREEN_WIDTH - 100, SCREEN_HEIGHT-self.menu_height, 100, self.menu_height, 'Quit', 'Quit', GREEN, RED, False)
        self.room_toggle = Toggle(0, SCREEN_HEIGHT - self.menu_height, 150, self.menu_height, '2D room', '2D room', True, 0)
        self.floor_toggle = Toggle(150, SCREEN_HEIGHT - self.menu_height, 150, self.menu_height, '2D floor', '2D floor', False, 1)
        self.pause_button = Switch(450, SCREEN_HEIGHT - self.menu_height, 150, self.menu_height, 'Pause', 'Pause', GREEN, RED, False)
        self.wave_sim = Toggle(300, SCREEN_HEIGHT - self.menu_height, 150, self.menu_height, 'Wave sim', 'Wave sim', False, 1)
        self.alarm_amount_floor = InputBox(SCREEN_WIDTH//2, SCREEN_HEIGHT - self.menu_height, 150, self.menu_height//2, 'Alarm amount:')
        self.animation_speed = InputBox(0, 0, 140, 30, 'Speed:')
        self.animation_speed.number_value = 1

        self.alarm_amount_room = InputBox(SCREEN_WIDTH//2, SCREEN_HEIGHT - self.menu_height, 150, self.menu_height//2, 'Alarm amount:')
        self.coverage_percentage_room = NumberDisplay(0, 0, 140, 30, 'Coverage:')
        self.max_coverage_percentage_room = NumberDisplay(0, 30, 140, 30, 'Max coverage:')
        self.potential_max = NumberDisplay(0, 60, 140, 30, 'Potential max:')
        self.add_obstacle = InputMultipleBox(SCREEN_WIDTH//2, SCREEN_HEIGHT - self.menu_height // 2, 200, self.menu_height//2, 'Add obstacle (x,y,w,h):')

        self.update_layout(SCREEN_WIDTH, SCREEN_HEIGHT)

    def update_layout(self, screen_width, screen_height):
        top_margin = 10
        gap = 8
        speed_width = 140
        speed_height = 30
        total_speed_width = speed_width * 1.5

        speed_x = screen_width - top_margin - total_speed_width
        speed_y = top_margin

        self.animation_speed.rect_label.update(speed_x, speed_y, speed_width, speed_height)
        self.animation_speed.rect.update(speed_x + speed_width, speed_y, speed_width * 0.5, speed_height)

        pause_width = 120
        pause_height = speed_height
        pause_x = speed_x - gap - pause_width
        self.pause_button.rect.update(pause_x, speed_y, pause_width, pause_height)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        if self.text:
            text_surf = self.font.render(self.text, True, WHITE)
            text_rect = text_surf.get_rect(center=self.rect.center)
            screen.blit(text_surf, text_rect)
        
        self.quit_button.draw(screen)
        self.room_toggle.draw(screen)
        self.floor_toggle.draw(screen)
        self.wave_sim.draw(screen)
        self.pause_button.draw(screen)
        self.animation_speed.draw(screen)
        
        # Draw input boxes if toggles are active
        if self.room_toggle.state and self.room_toggle.value == 0:
            self.alarm_amount_room.draw(screen)
            self.coverage_percentage_room.draw(screen, "%")
            self.max_coverage_percentage_room.draw(screen, "%")
            self.potential_max.draw(screen)
            self.add_obstacle.draw(screen)
        if self.floor_toggle.state and self.floor_toggle.value == 0:
            self.alarm_amount_floor.draw(screen)
    def update(self, dt, value):
        self.quit_button.update(dt)
        self.pause_button.update(dt)
        self.room_toggle.update(dt)
        if self.room_toggle.state and self.room_toggle.value ==0:
            self.floor_toggle.value = 1
            self.wave_sim.value = 1
            self.coverage_percentage_room.update(value[0])
            self.max_coverage_percentage_room.update(value[1])
            self.potential_max.update(value[2])
        self.floor_toggle.update(dt)
        if self.floor_toggle.state and self.floor_toggle.value ==0:
            self.room_toggle.value = 1
            self.wave_sim.value = 1
        self.wave_sim.update(dt)
        if self.wave_sim.state and self.wave_sim.value ==0:
            self.room_toggle.value = 1
            self.floor_toggle.value = 1

    # Handle events for input boxes
    def handle_event(self, event):
        self.animation_speed.handle_event(event)
        if self.room_toggle.state and self.room_toggle.value ==0:
            self.alarm_amount_room.handle_event(event)
            self.add_obstacle.handle_event(event)

        if self.floor_toggle.state and self.floor_toggle.value ==0:
            self.alarm_amount_floor.handle_event(event)