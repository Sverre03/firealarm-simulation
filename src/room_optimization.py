from dataclasses import dataclass
import numpy as np

from optimalisering2d import expected_improvement, loop
from config import WAVE_SOURCE_STRENGTH, WAVE_THRESHOLD, WAVE_STEPS, WAVE_WARMUP_STEPS, WAVE_SPEED, WAVE_DT, WAVE_DX, WAVE_DAMPING, WAVE_FREQUENCY

@dataclass
class OptimizationResult:
    alarm_positions: list[tuple[int, int]]
    potential: np.ndarray
    coverage_percentage: float
    potential_max: float

def FDM_solve(obstacle_mask, alarm_positions):
    source_strength = WAVE_SOURCE_STRENGTH
    threshold = WAVE_THRESHOLD
    n_steps = WAVE_STEPS
    warmup_steps = WAVE_WARMUP_STEPS
    wave_speed = WAVE_SPEED
    dt = WAVE_DT
    dx = WAVE_DX
    gamma = WAVE_DAMPING
    frequency = WAVE_FREQUENCY

    # Finite difference method:
    # u_tt + gamma * u_t = c^2 * Laplacian(u) + f(x, y, t)

    x_dim, y_dim = obstacle_mask.shape
    free_mask = ~obstacle_mask

    u_prev = np.zeros((x_dim, y_dim))
    u_current = np.zeros((x_dim, y_dim))

    valid_sources = []
    for x, y in alarm_positions:
        if 1 <= x < x_dim - 1 and 1 <= y < y_dim - 1 and not obstacle_mask[x, y]:
            valid_sources.append((x, y))

    Cx = wave_speed * dt / dx # Courant number
    Cx2 = Cx**2 # Stability condition for explicit scheme, must be <= 0.5
    if Cx2 > 0.5:
        print(f"Ustabilt valg av parametere, Cx^2 = {Cx2:.3f} > 0.5.")

    rms_accum = np.zeros_like(u_current)
    rms_count = 0
    omega = 2.0 * np.pi * frequency

    D2 = 1.0 - gamma * dt 
    D1 = 2.0 - gamma * dt

    for step in range(n_steps):
        left = u_current[:-2, 1:-1]
        right = u_current[2:, 1:-1]
        down = u_current[1:-1, :-2]
        up = u_current[1:-1, 2:]

        left_obs = obstacle_mask[:-2, 1:-1]
        right_obs = obstacle_mask[2:, 1:-1]
        down_obs = obstacle_mask[1:-1, :-2]
        up_obs = obstacle_mask[1:-1, 2:]

        # Approksimasjon av Neumann grensebetingelser ved å reflektere verdier ved hindringer
        left_effective = np.where(left_obs, right, left)
        right_effective = np.where(right_obs, left, right)
        down_effective = np.where(down_obs, up, down)
        up_effective = np.where(up_obs, down, up)

        # Laplacian
        u_xx = left_effective + right_effective + down_effective + up_effective - 4.0 * u_current[1:-1, 1:-1]

        u_next = np.zeros_like(u_current)
        # Oppdater med FDM, delvis inspirert av hplgit.github.io/fdm-book/doc/pub/book/sphinx/._book008.html
        # u[i,j][n+1] = (2 - gamma * dt) * u[i,j][n] - (1 - gamma * dt) * u[i,j][n-1] + Cx^2 * Laplacian(u[i,j][n])
        u_next[1:-1, 1:-1] = (D1*u_current[1:-1, 1:-1] # + (2 - gamma * dt) * u[i,j][n]
                              - D2*u_prev[1:-1, 1:-1] # + (1 - gamma * dt) * u[i,j][n-1]
                              + Cx2 * u_xx) # Cx^2 * Laplacian(u[i,j][n])

        # Simulerer en sinusformet kilde ved hver alarmplassering
        strength = source_strength * np.sin(omega * step * dt)
        for x, y in valid_sources:
            u_next[x, y] += (dt * dt) * strength # dt^2 * f(x, y, t)

        u_next[obstacle_mask] = 0.0

        if step >= warmup_steps:
            rms_accum += u_next * u_next
            rms_count += 1

        u_prev, u_current = u_current, u_next

    rms_map = np.sqrt(rms_accum / float(rms_count))

    covered_mask = (rms_map >= threshold) & free_mask
    free_cells = int(np.count_nonzero(free_mask))
    covered_cells = int(np.count_nonzero(covered_mask))
    coverage = (100.0 * covered_cells / free_cells)
    potential_max = float(np.max(rms_map[free_mask]))
    return rms_map, coverage, potential_max


def build_candidate_points(obstacle_mask, spacing):
    x_dim, y_dim = obstacle_mask.shape
    free_cells = np.argwhere(~obstacle_mask)

    interior_free_cells = free_cells[
        (free_cells[:, 0] > 0)
        & (free_cells[:, 0] < x_dim - 1)
        & (free_cells[:, 1] > 0)
        & (free_cells[:, 1] < y_dim - 1)
    ]
    if spacing <= 1:
        return interior_free_cells

    sampled_cells = interior_free_cells[(interior_free_cells[:, 0] % spacing == 0) & (interior_free_cells[:, 1] % spacing == 0)]
    if sampled_cells.shape[0] == 0:
        return interior_free_cells
    return sampled_cells


def sorted_alarm_positions(alarm_coordinates_flat, n_alarms): # Konverter 1D-array til liste av (x,y)-tupler
    alarm_coordinates = np.asarray(alarm_coordinates_flat, dtype=float).reshape(n_alarms, 2)
    alarm_coordinates = np.round(alarm_coordinates).astype(int)
    sorted_points = sorted((int(x), int(y)) for x, y in alarm_coordinates)
    return sorted_points


def build_domain(candidates, n_alarms, domain_limit, rng):
    candidate_indices = list(range(len(candidates)))
    seen_combos = set()
    alarm_layouts = []

    while len(alarm_layouts) < domain_limit:
        combo = tuple(sorted(rng.choice(candidate_indices, size=n_alarms, replace=False).tolist()))
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        layout = candidates[list(combo)].reshape(-1)
        alarm_layouts.append(layout)

    return alarm_layouts


def optimize_alarms(obstacle_mask, n_alarms, candidate_spacing=8, domain_limit=350, init_samples=8):
    rng = np.random.default_rng(None)

    # Returnerer gyldige alarmplasseringer og minimumsavstand mellom alarmer (redusere området den må søke i)
    candidates = build_candidate_points(obstacle_mask, candidate_spacing)

    if len(candidates) < n_alarms:
        raise ValueError("Not enough valid candidate cells to place the requested number of alarms.")

    candidate_layouts = np.asarray(build_domain(candidates, n_alarms, domain_limit, rng), dtype=float)

    cache = {} # Lagrer coverage, potential og max_potential for hver layout

    def objective(alarm_coordinates):
        layout_key = tuple(sorted_alarm_positions(alarm_coordinates, n_alarms))
        if layout_key in cache:
            return cache[layout_key][0]

        potential, coverage, max_potential = FDM_solve(obstacle_mask, alarm_positions=list(layout_key))
        cache[layout_key] = (coverage, potential, max_potential)
        return coverage

    init_count = min(max(1, init_samples), len(candidate_layouts))
    x_init = candidate_layouts[rng.choice(len(candidate_layouts), size=init_count, replace=False)]

    layout_samples, coverage_samples, _ = loop(x_init=x_init, sim_func=objective, acq_func=expected_improvement, domain=candidate_layouts)

    best_idx = int(np.argmax(coverage_samples)) # Get the index of the best alarm layout found by the optimization loop.
    alarm_positions = sorted_alarm_positions(layout_samples[best_idx], n_alarms)
    best_key = tuple(alarm_positions)

    coverage, potential, max_potential = cache[best_key]

    return OptimizationResult(alarm_positions, potential, coverage, max_potential)
