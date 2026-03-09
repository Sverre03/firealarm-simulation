import pygame
import numpy as np
from buttons import *
from config import *
from ui import *
from FEM import FEM_draw, FEM_setup
from rooms import draw_room, rooms
from room_optimization import optimize_alarms

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
    potential_max = 0.0
    number_values = [0.0, 0.0]  # List to hold values for updating the UI
    calculating = False
    previous_calculate_state = ui.calculate_button.state
    room_result = None # Returned by the optimization function,
    # Should contain the potential, alarm positions, coverage and maybe other stuff that has been calculated

    potential_1D_wave, number_of_nodes = FEM_setup(SCREEN_WIDTH, SCREEN_HEIGHT)    
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
        number_values[1] = potential_max
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
            calculation_requested = ui.calculate_button.state and not previous_calculate_state
            previous_calculate_state = ui.calculate_button.state

            if ui.room_choice.number_value != ui.room_choice.number_value_past:
                # Reset everything
                room_result = None
                coverage_percentage = 0.0
                potential_max = 0.0
                ui.room_choice.number_value_past = ui.room_choice.number_value

            if not calculating and calculation_requested:
                calculating = True
                alarm_count = ui.alarm_amount_room.number_value
                room_choice = ui.room_choice.number_value if ui.room_choice.number_value in rooms else 1
                obstacle_mask = rooms[room_choice]

                # Optimization sketch:
                room_result = optimize_alarms(obstacle_mask, alarm_count)
                coverage_percentage = room_result.coverage_percentage
                potential_max = room_result.potential_max

                calculating = False
                ui.calculate_button.set_state(False)

            if ui.alarm_amount_room.number_value != ui.alarm_amount_room.number_value_past: 
                room_result = None
                coverage_percentage = 0.0
                ui.alarm_amount_room.number_value_past = ui.alarm_amount_room.number_value

            room_choice = ui.room_choice.number_value if ui.room_choice.number_value in rooms else 1

            if room_result is not None: # Draw heatmap
                draw_room(screen, room_choice, potential=room_result.potential, alarms=room_result.alarm_positions)
            else: # Draw room without potentials
                draw_room(screen, room_choice)

        else:
            previous_calculate_state = ui.calculate_button.state
            
        ui.draw(screen)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()