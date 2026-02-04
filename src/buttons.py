
import pygame
from config import *

class Switch:
    def __init__(self, x, y, width, height, text_active='Active', text_inactive='Inactive', initial_state=True):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = GREEN if initial_state else RED
        self.text_active = text_active
        self.text_inactive = text_inactive
        self.text = text_active if initial_state else text_inactive
        self.font = pygame.font.SysFont(None, 24)
        self.state = initial_state  # True for 'Active', False for 'Inactive'
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
                self.text = self.text_active if self.state else self.text_inactive
        self.last_switch_time += dt

class Toggle:
    def __init__(self, x, y, width, height, text_active='Active', text_inactive='Inactive', initial_state=False, value=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = TOGGLE_ON if initial_state else TOGGLE_OFF
        self.text_active = text_active
        self.text_inactive = text_inactive
        self.text = text_active if initial_state else text_inactive
        self.font = pygame.font.SysFont(None, 24)
        self.state = initial_state  # True for 'Active', False for 'Inactive'
        self.switch_delay = 0.5 # seconds
        self.last_switch_time = 0
        self.value = value

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        if self.text:
            text_surf = self.font.render(self.text, True, BLACK)
            text_rect = text_surf.get_rect(center=self.rect.center)
            screen.blit(text_surf, text_rect)

    def update(self, dt):
        if pygame.mouse.get_pressed()[0] and self.last_switch_time > self.switch_delay and self.value ==1:
            self.last_switch_time = 0
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                print(f'Switch "{self.text}" clicked!')
                self.value = 0
                self.state = True
                self.color = TOGGLE_ON if self.state else TOGGLE_OFF
                self.text = self.text_active if self.state else self.text_inactive
                
        elif self.value == 1:
            self.state = False
            self.color = TOGGLE_ON if self.state else TOGGLE_OFF
            self.text = self.text_active if self.state else self.text_inactive

        self.last_switch_time += dt