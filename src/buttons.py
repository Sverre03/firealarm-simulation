
import pygame
from config import *

# Simple switch button
class Switch:
    def __init__(self, x, y, width, height, text_active='Active', text_inactive='Inactive', active_color=GREEN, inactive_color=RED, initial_state=True):
        self.rect = pygame.Rect(x, y, width, height)
        self.active_color = active_color
        self.inactive_color = inactive_color
        self.color = self.active_color if initial_state else self.inactive_color
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
                self.color = self.active_color if self.state else self.inactive_color
                self.text = self.text_active if self.state else self.text_inactive
        self.last_switch_time += dt

# Toggle button for mutually exclusive options
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

# Input box for numerical values
class InputBox:
    def __init__(self, x, y, width, height, label=''):
        self.rect = pygame.Rect(x + width, y, width*0.5, height)
        self.rect_label = pygame.Rect(x, y, width, height)
        self.color = WHITE
        self.text = ""
        self.label = label
        self.font = pygame.font.SysFont(None, 24)
        self.active = False
        self.number_value = 2

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, TOGGLE_ON, self.rect_label)

        display_text = self.text+'_' if self.active else str(self.number_value)
        text_surf = self.font.render(display_text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

        text_surf_label = self.font.render(self.label, True, BLACK)
        text_rect_label = text_surf_label.get_rect(center=self.rect_label.center)
        screen.blit(text_surf_label, text_rect_label)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                if self.text.isdigit():
                    self.number_value = int(self.text)
                    print("Number set to:", self.number_value)
                self.text = ""
                self.active = False 
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if event.unicode.isdigit():
                    new_text = self.text + event.unicode
                
                    if int(new_text) <= ALARM_MAX: 
                        self.text = new_text

class NumberDisplay:
    def __init__(self, x, y, width, height, label=''):
        self.rect = pygame.Rect(x + width, y, width*0.5, height)
        self.rect_label = pygame.Rect(x, y, width, height)
        self.color = TOGGLE_ON
        self.value = 0
        self.label = label
        self.font = pygame.font.SysFont(None, 24)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, self.color, self.rect_label)

        display_text = f"{self.value:.1f}%"
        text_surf = self.font.render(display_text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

        text_surf_label = self.font.render(self.label, True, BLACK)
        text_rect_label = text_surf_label.get_rect(center=self.rect_label.center)
        screen.blit(text_surf_label, text_rect_label)

    def update(self, new_value):
        self.value = new_value  