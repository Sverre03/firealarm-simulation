
import pygame
from config import *

class Switch:
    def __init__(self, x, y, width, height, color, text=''):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.SysFont(None, 24)
        self.state = True  # True for 'Active', False for 'Inactive'
        self.switch_delay = 0.5 # seconds
        self.last_switch_time = 0

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        if self.text:
            text_surf = self.font.render(self.text, True, BLACK)
            text_rect = text_surf.get_rect(center=self.rect.center)
            screen.blit(text_surf, text_rect)

    def update(self, dt):
        if pygame.mouse.get_pressed()[0] and self.last_switch_time > self.switch_delay:
            self.last_switch_time = 0
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                print(f'Switch "{self.text}" clicked!')
                self.state = not self.state
                self.color = GREEN if self.state else RED
                self.text = 'Active' if self.state else 'Inactive'
        self.last_switch_time += dt