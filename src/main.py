import pygame
import numpy as np
from buttons import *
from config import *
from ui import *
from FEM import FEM_draw, FEM_setup

def main():
    # Initialize Pygame
    pygame.init()
    SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.get_desktop_sizes()[0]

    # Set up display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Fire Alarm Simulation")

    ui = Menu(0, SCREEN_HEIGHT - 50, SCREEN_WIDTH, SCREEN_HEIGHT, DARK_GREY, '')

    # Main loop
    running = True
    frame = 0

    u, number_of_nodes = FEM_setup(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # Det som er under her + FEM_draw() er laget av KI
    nt=u.shape[1]

    # x-coordinates in pixels
    x_pixels = np.linspace(50, SCREEN_WIDTH - 50, number_of_nodes)

    y_center = SCREEN_HEIGHT // 2

    # auto-scale displacement
    u_max = np.max(np.abs(u))
    y_scale = 0.45 * SCREEN_HEIGHT / u_max
    # Det som kommer under er laget på egenhånd

    clock = pygame.time.Clock()
    while running:
        dt = clock.tick(FPS) / 1000
        SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update
        ui.update(dt)
        if ui.quit_button.state:
            running = False
        
        # Draw
        screen.fill(GREY)

        ui.draw(screen)
        

        FEM_draw(screen, frame, u, number_of_nodes, x_pixels, y_center, y_scale, nt)
        frame = (frame + 1) % nt

        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()