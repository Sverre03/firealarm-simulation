import pygame
from buttons import Switch

WHITE = (255, 255, 255)
FPS = 60

def main():
    # Initialize Pygame
    pygame.init()

    # Set up display
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Fire Alarm Simulation")

    switch = Switch(350, 500, 100, 50, (0, 255, 0), 'Active')

    # Main loop
    running = True
    clock = pygame.time.Clock()
    while running:
        dt = clock.tick(FPS) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update
        switch.update(dt)

        # Draw
        screen.fill(WHITE)
        switch.draw(screen)
        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()