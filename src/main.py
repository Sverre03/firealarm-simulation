import pygame
from buttons import Switch
from config import *

def main():
    # Initialize Pygame
    pygame.init()
    SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.get_desktop_sizes()[0]

    # Set up display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Fire Alarm Simulation")

    switch = Switch(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 100, 100, 50, GREEN, 'Active')
    quit_button = Switch(SCREEN_WIDTH - 110, 10, 100, 50, RED, 'Quit')

    # Main loop
    running = True
    clock = pygame.time.Clock()
    while running:
        dt = clock.tick(FPS) / 1000
        SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update
        switch.update(dt)
        quit_button.update(dt)
        if not quit_button.state:
            running = False

        # Draw
        screen.fill(WHITE)

        switch.draw(screen)
        quit_button.draw(screen)
        
        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()