
import pygame
from config import *
from buttons import *

class Menu:
    def __init__(self, x, y, SCREEN_WIDTH, SCREEN_HEIGHT, color, text = 'Menu'):
        self.rect = pygame.Rect(x, y, SCREEN_WIDTH, 50)
        self.color = color
        self.text = text
        self.font = pygame.font.SysFont(None, 24)
        self.quit_button = Switch(SCREEN_WIDTH - 100, SCREEN_HEIGHT-50, 100, 50, 'Quit', 'Quit', False)
        self.room_toggle = Toggle(0, SCREEN_HEIGHT - 50, 150, 50, '2D room', '2D room', True, 0)
        self.floor_toggle = Toggle(150, SCREEN_HEIGHT - 50, 150, 50, '2D floor', '2D floor', False, 1)
        self.wave_sim = Toggle(300, SCREEN_HEIGHT - 50, 150, 50, 'Wave sim', 'Wave sim', False, 1)

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

    def update(self, dt):
        self.quit_button.update(dt)
        self.room_toggle.update(dt)
        if self.room_toggle.state and self.room_toggle.value ==0:
            self.floor_toggle.value = 1
            self.wave_sim.value = 1
        self.floor_toggle.update(dt)
        if self.floor_toggle.state and self.floor_toggle.value ==0:
            self.room_toggle.value = 1
            self.wave_sim.value = 1
        self.wave_sim.update(dt)
        if self.wave_sim.state and self.wave_sim.value ==0:
            self.room_toggle.value = 1
            self.floor_toggle.value = 1