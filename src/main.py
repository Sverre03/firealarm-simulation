import argparse
import pygame
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from buttons import *
from config import *
from ui import *
from FEM import FEM_draw, FEM_setup
from rooms import draw_room, room_showcase, rooms
from room_optimization import optimize_alarms

def main(debug_optimization=False):
    # Initialize Pygame
    pygame.init()
    SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.get_desktop_sizes()[0]

    # Set up display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Fire Alarm Simulation")

    ui = Menu(0, SCREEN_HEIGHT - SCREEN_HEIGHT*MENU_HEIGHT_MULTI, SCREEN_WIDTH, SCREEN_HEIGHT, DARK_GREY, '')

    # Main loop
    running = True
    coverage_percentage = 0.0
    sound_pressure_max = 0.0
    number_values = [0.0, 0.0]  # List to hold values for updating the UI
    calculating = False
    previous_calculate_state = ui.calculate_button.state
    room_result = None # Returned by the optimization function,
    # Should contain the sound_pressure, alarm positions, coverage and maybe other stuff that has been calculated
    optimization_pool = ThreadPoolExecutor(max_workers=1)
    optimization_task = None

    clock = pygame.time.Clock()

    while running:
        dt = clock.tick(FPS) / 1000
        SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
        ui.update_layout(SCREEN_WIDTH, SCREEN_HEIGHT+1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            ui.handle_event(event)
        
        # Update
        number_values[0] = coverage_percentage
        number_values[1] = sound_pressure_max
        ui.update(dt, number_values[0:])
        
        if ui.quit_button.state:
            running = False   
                
        # Draw
        screen.fill(GREY)

        speed = max(0.0, float(ui.animation_speed.number_value))
        if ui.room_toggle.state and ui.room_toggle.value == 0:
            calculation_requested = ui.calculate_button.state and not previous_calculate_state
            previous_calculate_state = ui.calculate_button.state

            if ui.room_choice.number_value != ui.room_choice.number_value_past:
                # Reset results when room changes.
                room_result = None
                coverage_percentage = 0.0
                sound_pressure_max = 0.0
                ui.room_choice.number_value_past = ui.room_choice.number_value

            if calculating and optimization_task is not None and optimization_task.done():
                room_result = optimization_task.result()
                coverage_percentage = room_result.coverage_percentage
                sound_pressure_max = room_result.sound_pressure_max

                calculating = False
                ui.calculate_button.set_state(False)
                optimization_task = None

            if not calculating and calculation_requested:
                calculating = True
                alarm_count = ui.alarm_amount_room.number_value
                room_choice = ui.room_choice.number_value if ui.room_choice.number_value in rooms else 1
                obstacle_mask = rooms[room_choice]

                # Optimization sketch:
                optimization_task = optimization_pool.submit(
                    optimize_alarms,
                    obstacle_mask,
                    alarm_count,
                    debug=debug_optimization,
                )

            if ui.alarm_amount_room.number_value != ui.alarm_amount_room.number_value_past: 
                # Reset results when alarm count changes.
                room_result = None
                coverage_percentage = 0.0
                sound_pressure_max = 0.0
                ui.alarm_amount_room.number_value_past = ui.alarm_amount_room.number_value

            room_choice = ui.room_choice.number_value if ui.room_choice.number_value in rooms else 1

            if room_result is not None: # Draw heatmap
                draw_room(screen, room_choice, sound_pressure=room_result.sound_pressure, alarms=room_result.alarm_positions)
            else: # Draw room without sound_pressures
                draw_room(screen, room_choice)

        if ui.floor_toggle.state and ui.floor_toggle.value == 0:
            room_showcase(screen, 1, left=0, top=screen.get_height()//24)
            room_showcase(screen, 2, left=1+screen.get_width()//3, top=screen.get_height()//24)
            room_showcase(screen, 3, left=1+2*screen.get_width()//3, top=screen.get_height()//24)
            room_showcase(screen, 4, left=0, top=screen.get_height()//3)   
            room_showcase(screen, 5, left=1+screen.get_width()//3, top=screen.get_height()//3)
            room_showcase(screen, 6, left=1+2*screen.get_width()//3, top=screen.get_height()//3)  
            room_showcase(screen, 7, left=0, top=2*screen.get_height()//3-screen.get_height()//24)
            room_showcase(screen, 8, left=1+screen.get_width()//3, top=2*screen.get_height()//3-screen.get_height()//24)
            room_showcase(screen, 9, left=1+2*screen.get_width()//3, top=2*screen.get_height()//3-screen.get_height()//24)

        else:
            previous_calculate_state = ui.calculate_button.state
            
        ui.draw(screen)

        pygame.display.flip()

    optimization_pool.shutdown(wait=False)

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Alarm Simulation")
    parser.add_argument(
        "--debug-optimization",
        action="store_true",
        help="Print optimization loop debug output.",
    )
    args = parser.parse_args()
    main(debug_optimization=args.debug_optimization)