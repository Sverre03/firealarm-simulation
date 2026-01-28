import pygame
from buttons import Button

WHITE = (255, 255, 255)
FPS = 1000

def main():
    # Initialize Pygame
    pygame.init()

    # Set up display
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Fire Alarm Simulation")

    button = Button(350, 500, 100, 50, (0, 255, 0), 'Activate')

    # Main loop
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(FPS)
        # Update

        # Draw
        screen.fill(WHITE)
        button.draw(screen)
        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()