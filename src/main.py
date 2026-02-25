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

    ui = Menu(0, SCREEN_HEIGHT - SCREEN_HEIGHT*MENU_HEIGHT_MULTI, SCREEN_WIDTH, SCREEN_HEIGHT, DARK_GREY, '')

    # Main loop
    running = True
    wave_frame = 0.0
    coverage_percentage = 0.0
    max_coverage_percentage = 0.0
    number_values = [0.0, 0.0, 0.0]  # List to hold values for updating the UI

    potential_1D_wave, number_of_nodes = FEM_setup(SCREEN_WIDTH, SCREEN_HEIGHT)
    potential_2D_laplace, obstacles, number_of_frames, iterations, last_update = FDM_laplace()
    room_frame = 0.0
    
    number_of_timesteps=potential_1D_wave.shape[1]

    # x-coordinates in pixels
    x_pixels = np.linspace(50, SCREEN_WIDTH - 50, number_of_nodes)

    # auto-scale displacement
    potential_1D_wave_max = np.max(np.abs(potential_1D_wave))
    y_scale = 0.4 * (SCREEN_HEIGHT - MENU_HEIGHT_MULTI*SCREEN_HEIGHT) / potential_1D_wave_max

    clock = pygame.time.Clock()
    while running:
        dt = clock.tick(FPS) / 1000
        SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
        ui.update_layout(SCREEN_WIDTH, SCREEN_HEIGHT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            ui.handle_event(event)
        
        # Update
        number_values[0] = coverage_percentage
        number_values[1] = max_coverage_percentage
        number_values[2] = np.max(np.abs(potential_2D_laplace))
        ui.update(dt, number_values[0:])
        
        if ui.quit_button.state:
            running = False            
        
        # Draw
        screen.fill(GREY)

        speed = max(0.0, float(ui.animation_speed.number_value))
        if ui.wave_sim.state and ui.wave_sim.value == 0:
            if not ui.pause_button.state:
                wave_frame = (wave_frame + speed) % number_of_timesteps
            FEM_draw(screen, int(wave_frame), potential_1D_wave, number_of_nodes, x_pixels, SCREEN_WIDTH, SCREEN_HEIGHT, y_scale, number_of_timesteps, paused=ui.pause_button.state)
        if ui.room_toggle.state and ui.room_toggle.value == 0:
            if not ui.pause_button.state:
                room_frame = (room_frame + dt * FPS * speed) % number_of_frames
            coverage_percentage = draw_frame(screen, potential_2D_laplace, obstacles, int(room_frame), SCREEN_WIDTH, SCREEN_HEIGHT, paused=ui.pause_button.state)  
            if ui.alarm_amount_room.number_value != ui.alarm_amount_room.number_value_past: 
                max_coverage_percentage = coverage_percentage
                ui.alarm_amount_room.number_value_past = ui.alarm_amount_room.number_value
            else:
                # Continue tracking max
                max_coverage_percentage = max(max_coverage_percentage, coverage_percentage)


            
        ui.draw(screen)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()