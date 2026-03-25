
import pygame
from config import *
from buttons import *    

class Menu:
    def __init__(self, x, y, screen_width, screen_height, color, text='Menu'):
        self.color = color
        self.text = text
        self.font = pygame.font.SysFont(None, 24)

        self.menu_height = MENU_HEIGHT_MULTI * screen_height
        self.rect = pygame.Rect(0, screen_height - self.menu_height, screen_width, self.menu_height)

        # Create widgets without fixed coordinates
        self.quit_button = Switch(0, 0, 0, 0, 'Quit', 'Quit', TOGGLE_OFF, RED, False)
        self.room_toggle = Toggle(0, 0, 0, 0, 'Room heatmap', 'Room heatmap', True, 0)
        self.floor_toggle = Toggle(0, 0, 0, 0, 'Available rooms', 'Available rooms', False, 1)
        self.pause_button = Switch(0, 0, 0, 0, 'Unpause', 'Pause', GREEN, RED, False)
        self.animation_speed = InputBox(0, 0, 0, 0, 'Speed:')
        self.animation_speed.number_value = 1

        self.alarm_amount_room = InputBox(0, 0, 0, 0, 'Alarm amount:', 1, 1, ALARM_MAX)
        self.room_choice = InputBox(0, 0, 0, 0, 'Room number:', 1, 1, ROOM_NR_MAX)
        self.calculate_button = Switch(0, 0, 0, 0, 'Calculating...', 'Calculate', TOGGLE_OFF, GREEN, False)
        self.coverage_percentage_room = NumberDisplay(0, 0, 140, 30, 'Coverage:')
        self.potential_max = NumberDisplay(0, 30, 140, 30, 'Potential max:')

        # Set positions based on initial screen size
        self.update_layout(screen_width, screen_height)

    def update_layout(self, screen_width, screen_height):
        self.menu_height = MENU_HEIGHT_MULTI * screen_height
        self.rect.update(0, screen_height - self.menu_height, screen_width, self.menu_height)
        menu_y = screen_height - self.menu_height

        gap = 8
        top_margin = 10

        # Quit button (right-aligned)
        self.quit_button.rect.update(screen_width - 100, menu_y, 100, self.menu_height)

        # Page toggles (bottom left)
        self.room_toggle.rect.update(0, menu_y, 150, self.menu_height)
        self.floor_toggle.rect.update(150, menu_y, 150, self.menu_height)

        # Alarm amount / room inputs (middle)
        self.alarm_amount_room.rect_label.update(screen_width // 2, menu_y, 150, self.menu_height // 2)
        self.alarm_amount_room.rect.update(screen_width // 2 + 150, menu_y, 150*0.5, self.menu_height // 2)
        self.room_choice.rect_label.update(screen_width // 2, menu_y + self.menu_height // 2, 150, self.menu_height // 2)
        self.room_choice.rect.update(screen_width // 2 + 150, menu_y + self.menu_height // 2, 150*0.5, self.menu_height // 2)
        self.calculate_button.rect.update(screen_width // 2 + 240, menu_y, 150, self.menu_height)

        # Coverage and potential displays (top left)
        self.coverage_percentage_room.rect_label.update(0, 0, 140, self.menu_height // 2)
        self.coverage_percentage_room.rect.update(0 + 140, 0, 140*0.5, self.menu_height // 2)

        # Animation speed (top right)
        speed_width = 140
        speed_height = 30
        total_speed_width = speed_width * 1.5
        speed_x = screen_width - top_margin - total_speed_width
        speed_y = top_margin
        self.animation_speed.rect_label.update(speed_x, speed_y, speed_width, speed_height)
        self.animation_speed.rect.update(speed_x + speed_width, speed_y, speed_width * 0.5, speed_height)

        # Pause button near top right
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
        
        # Draw input boxes if toggles are active
        if self.room_toggle.state and self.room_toggle.value == 0:
            self.alarm_amount_room.draw(screen)
            self.room_choice.draw(screen)
            self.calculate_button.draw(screen)
            self.coverage_percentage_room.draw(screen, "%")

    def update(self, dt, value):
        self.quit_button.update(dt)
        self.pause_button.update(dt)
        self.room_toggle.update(dt)
        if not self.calculate_button.state:
            self.calculate_button.update(dt)
        if self.room_toggle.state and self.room_toggle.value ==0:
            self.floor_toggle.value = 1
            self.coverage_percentage_room.update(value[0])

        self.floor_toggle.update(dt)
        if self.floor_toggle.state and self.floor_toggle.value ==0:
            self.room_toggle.value = 1

    # Handle events for input boxes
    def handle_event(self, event):
        self.animation_speed.handle_event(event)
        if self.room_toggle.state and self.room_toggle.value ==0 and not self.calculate_button.state:
            self.alarm_amount_room.handle_event(event)
            self.room_choice.handle_event(event)

