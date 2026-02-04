import pygame
import numpy as np
from buttons import *
from config import *
from ui import *
from FEM import FEM_draw, FEM_setup
from FDM_laplace import FDM_laplace, draw_frame

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
    u_laplace, walls, number_of_frames, iterations, last_update = FDM_laplace()
    room_frame = 0
    
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
        
        if ui.wave_sim.state and ui.wave_sim.value == 0:
            FEM_draw(screen, frame, u, number_of_nodes, x_pixels, y_center, y_scale, nt)
        if ui.room_toggle.state and ui.room_toggle.value == 0:
            room_frame = room_frame + (dt * FPS) % number_of_frames
            draw_frame(screen, u_laplace, walls, int(room_frame))

        frame = (frame + 1) % nt

        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()