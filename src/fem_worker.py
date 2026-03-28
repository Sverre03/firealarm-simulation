"""Isolated FEM worker process.

This script is executed as a separate Python process to protect the UI process
from PETSc/MPI aborts during FEniCSx solves.
"""

from __future__ import annotations

import sys
import numpy as np

from fem_fire_alarm_coverage import PointSource, solve_fire_alarm_intensity_map


def _obstacle_mask_to_wall_rectangles(obstacle_mask: np.ndarray):
    x_dim, y_dim = obstacle_mask.shape
    walls = []

    for x in range(x_dim):
        ys = np.flatnonzero(obstacle_mask[x, :])
        if ys.size == 0:
            continue

        start = int(ys[0])
        prev = int(ys[0])
        for y in ys[1:]:
            y = int(y)
            if y != prev + 1:
                walls.append((float(x), float(start), 1.0, float(prev - start + 1)))
                start = y
            prev = y

        walls.append((float(x), float(start), 1.0, float(prev - start + 1)))

    return walls


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: fem_worker.py <input.npz> <output.npz>", file=sys.stderr)
        return 2

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with np.load(input_path) as data:
        obstacle_mask = np.asarray(data["obstacle_mask"], dtype=bool)
        alarm_positions = np.asarray(data["alarm_positions"], dtype=np.int32)
        fem_nx = int(np.asarray(data["fem_nx"]).reshape(-1)[0])
        fem_ny = int(np.asarray(data["fem_ny"]).reshape(-1)[0])
        threshold = float(np.asarray(data["threshold"]).reshape(-1)[0])
        room_width_m = float(np.asarray(data["room_width_m"]).reshape(-1)[0])
        room_height_m = float(np.asarray(data["room_height_m"]).reshape(-1)[0])
        alarm_frequency_hz = float(np.asarray(data["alarm_frequency_hz"]).reshape(-1)[0])
        alarm_strength = float(np.asarray(data["alarm_strength"]).reshape(-1)[0])
        alarm_spread_m = float(np.asarray(data["alarm_spread_m"]).reshape(-1)[0])

    x_dim, y_dim = obstacle_mask.shape
    cell_size_x = room_width_m / float(max(1, x_dim))
    cell_size_y = room_height_m / float(max(1, y_dim))
    walls = _obstacle_mask_to_wall_rectangles(obstacle_mask)
    walls = [
        (
            float(wx) * cell_size_x,
            float(wy) * cell_size_y,
            float(ww) * cell_size_x,
            float(wh) * cell_size_y,
        )
        for wx, wy, ww, wh in walls
    ]
    alarms = [
        PointSource(
            x=(float(x) + 0.5) * cell_size_x,
            y=(float(y) + 0.5) * cell_size_y,
            strength=alarm_strength,
            spread=alarm_spread_m,
        )
        for x, y in alarm_positions
    ]

    _, _, _, pressure_grid, _ = solve_fire_alarm_intensity_map(
        room_width=float(room_width_m),
        room_height=float(room_height_m),
        nx=fem_nx,
        ny=fem_ny,
        alarms=alarms,
        walls=walls,
        frequency_hz=alarm_frequency_hz,
        threshold=threshold,
    )

    np.savez_compressed(output_path, pressure_grid=np.asarray(pressure_grid, dtype=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
