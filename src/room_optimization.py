from dataclasses import dataclass
import importlib
import time
import numpy as np

from config import (
    WAVE_THRESHOLD,
    WAVE_STEPS,
    WAVE_WARMUP_STEPS,
    WAVE_DAMPING,
    ROOM_WIDTH_M,
    ROOM_HEIGHT_M,
    GRID_CELLS_X,
    GRID_CELLS_Y,
    CELL_SIZE_M,
    ALARM_FREQUENCY_HZ,
    ALARM_SOURCE_STRENGTH_FDM,
    ALARM_SOURCE_STRENGTH_FEM,
    ALARM_SOURCE_SPREAD_M,
    COVERAGE_THRESHOLD_PA,
)

def _check_fem_available() -> bool:
    try:
        importlib.import_module("fem_fire_alarm_coverage")
        return True
    except Exception:
        return False


FEM_AVAILABLE = _check_fem_available()

@dataclass
class OptimizationResult:
    alarm_positions: list[tuple[int, int]]
    sound_pressure: np.ndarray
    coverage_percentage: float
    sound_pressure_max: float

def FDM_solve(obstacle_mask, alarm_positions, debug=False):
    source_strength = ALARM_SOURCE_STRENGTH_FDM
    threshold = COVERAGE_THRESHOLD_PA
    n_steps = WAVE_STEPS
    warmup_steps = WAVE_WARMUP_STEPS
    wave_speed = 343.0
    dx = CELL_SIZE_M
    dt = 0.00007
    gamma = WAVE_DAMPING
    frequency = ALARM_FREQUENCY_HZ

    # Finite difference method:
    # u_tt + gamma * u_t = c^2 * Laplacian(u) + f(x, y, t)
    # https://www.scirp.org/journal/paperinformation?paperid=112134

    x_dim, y_dim = obstacle_mask.shape
    free_mask = ~obstacle_mask

    u_prev = np.zeros((x_dim, y_dim))
    u_current = np.zeros((x_dim, y_dim))

    if debug:
        print(f"[FDM_solve] Testing positions: {alarm_positions}")

    valid_sources = []
    for x, y in alarm_positions:
        if 1 <= x < x_dim - 1 and 1 <= y < y_dim - 1 and not obstacle_mask[x, y]:
            valid_sources.append((x, y))

    if debug:
        print(f"[FDM_solve] Valid sources after filtering: {valid_sources} (grid {x_dim}x{y_dim})")

    Cx = wave_speed * dt / dx # Courant number
    Cx2 = Cx**2 # Stability condition for explicit scheme, must be <= 0.5
    if Cx2 > 0.5:
        print(f"Ustabilt valg av parametere, Cx^2 = {Cx2:.3f} > 0.5.")

    rms_accum = np.zeros_like(u_current)
    rms_count = 0
    omega = 2.0 * np.pi * frequency

    # D2 = 1.0 - gamma * dt 
    # D1 = 2.0 - gamma * dt
    D2 = 1
    D1 = 2

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
        center = u_current[1:-1, 1:-1]

        left_effective = np.where(left_obs, 0.0 * right, left)
        right_effective = np.where(right_obs, 0.0 * left, right)
        down_effective = np.where(down_obs, 0.0 * up, down)
        up_effective = np.where(up_obs, 0.0 * down, up)

        # Laplacian
        u_xx = left_effective + right_effective + down_effective + up_effective - 4.0 * center

        u_next = np.zeros_like(u_current)
        # Oppdater med FDM, delvis inspirert av hplgit.github.io/fdm-book/doc/pub/book/sphinx/._book008.html
        # u[i,j][n+1] = (2 - gamma * dt) * u[i,j][n] - (1 - gamma * dt) * u[i,j][n-1] + Cx^2 * Laplacian(u[i,j][n])
        u_next[1:-1, 1:-1] = (D1*u_current[1:-1, 1:-1] # + (2 - gamma * dt) * u[i,j][n]
                              - D2*u_prev[1:-1, 1:-1] # + (1 - gamma * dt) * u[i,j][n-1]
                              + Cx2 * u_xx) # Cx^2 * Laplacian(u[i,j][n])
        
        # D1 og D2 er absorpsjonskoeffisienter. Hvis D1 = 2 og D2 = 1 har vi ingen absorpsjon. 

        # Simulerer en sinusformet kilde ved hver alarmplassering
        strength = source_strength * np.sin(omega * step * dt)
        for x, y in valid_sources:
            u_next[x, y] += (dt * dt) * strength # dt^2 * f(x, y, t)

        # u_next[obstacle_mask] = 0.0

        if step >= warmup_steps:
            rms_accum += u_next * u_next
            rms_count += 1

        u_prev, u_current = u_current, u_next

    rms_map = np.sqrt(rms_accum / float(rms_count))

    covered_mask = (rms_map >= threshold) & free_mask
    free_cells = int(np.count_nonzero(free_mask))
    covered_cells = int(np.count_nonzero(covered_mask))
    coverage = (100.0 * covered_cells / free_cells)
    sound_pressure_max = float(np.max(rms_map[free_mask]))
    return rms_map, coverage, sound_pressure_max


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


def FEM_solve(obstacle_mask, alarm_positions):
    if not FEM_AVAILABLE:
        raise RuntimeError(
            "FEM solver is unavailable. Install FEniCSx dependencies (dolfinx, ufl, mpi4py, petsc4py)."
        )

    try:
        from fem_fire_alarm_coverage import PointSource, solve_fire_alarm_intensity_map
    except Exception as exc:
        raise RuntimeError(
            "FEM solver is unavailable. Install FEniCSx dependencies (dolfinx, ufl, mpi4py, petsc4py)."
        ) from exc

    x_dim, y_dim = obstacle_mask.shape
    free_mask = ~obstacle_mask

    walls = _obstacle_mask_to_wall_rectangles(obstacle_mask)

    fem_nx = GRID_CELLS_X
    fem_ny = GRID_CELLS_Y

    cell_size_x = ROOM_WIDTH_M / float(max(1, x_dim))
    cell_size_y = ROOM_HEIGHT_M / float(max(1, y_dim))

    walls_m = [
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
            strength=float(ALARM_SOURCE_STRENGTH_FEM),
            spread=float(ALARM_SOURCE_SPREAD_M),
        )
        for x, y in list(alarm_positions)
    ]

    _, _, _, pressure_grid, _ = solve_fire_alarm_intensity_map(
        room_width=float(ROOM_WIDTH_M),
        room_height=float(ROOM_HEIGHT_M),
        nx=int(fem_nx),
        ny=int(fem_ny),
        alarms=alarms,
        walls=walls_m,
        frequency_hz=float(ALARM_FREQUENCY_HZ),
        threshold=float(COVERAGE_THRESHOLD_PA),
    )

    # FEM grid is (y, x). Convert to project convention (x, y).
    fem_map = np.abs(pressure_grid.T)

    if fem_map.shape != obstacle_mask.shape:
        x_idx = np.linspace(0, fem_map.shape[0] - 1, x_dim)
        y_idx = np.linspace(0, fem_map.shape[1] - 1, y_dim)
        x_nn = np.round(x_idx).astype(int)
        y_nn = np.round(y_idx).astype(int)
        fem_map = fem_map[np.ix_(x_nn, y_nn)]

    covered_mask = (fem_map >= COVERAGE_THRESHOLD_PA) & free_mask
    free_cells = int(np.count_nonzero(free_mask))
    covered_cells = int(np.count_nonzero(covered_mask))
    coverage = (100.0 * covered_cells / free_cells) if free_cells else 0.0

    if np.any(free_mask):
        sound_pressure_max = float(np.max(fem_map[free_mask]))
    else:
        sound_pressure_max = 0.0

    return fem_map, coverage, sound_pressure_max


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


def optimize_alarms(obstacle_mask, n_alarms, candidate_spacing=8, domain_limit=350, init_samples=8, solver="FDM", debug=False):
    rng = np.random.default_rng(None)

    solver_name = str(solver).upper()
    if solver_name not in ("FDM", "FEM"):
        raise ValueError(f"Unsupported solver '{solver}'. Choose 'FDM' or 'FEM'.")

    if solver_name == "FEM" and not FEM_AVAILABLE:
        if debug:
            print("[DEBUG] FEM dependencies unavailable; falling back to FDM.")
        solver_name = "FDM"

    if solver_name == "FEM":
        # Keep FEM optimization responsive enough for UI interaction.
        candidate_spacing = max(candidate_spacing, 18)
        domain_limit = min(domain_limit, 80)
        init_samples = min(init_samples, 6)

    # Returnerer gyldige alarmplasseringer og minimumsavstand mellom alarmer (redusere området den må søke i)
    candidates = build_candidate_points(obstacle_mask, candidate_spacing)

    if debug:
        print(
            "[DEBUG] Starting optimization: "
            f"solver={solver_name}, n_alarms={n_alarms}, candidate_spacing={candidate_spacing}, "
            f"domain_limit={domain_limit}, init_samples={init_samples}, candidate_count={len(candidates)}"
        )

    optimization_start = time.perf_counter()
    iterations = 0

    if len(candidates) < n_alarms:
        raise ValueError("Not enough valid candidate cells to place the requested number of alarms.")

    candidate_layouts = np.asarray(build_domain(candidates, n_alarms, domain_limit, rng), dtype=float)

    cache = {} # Lagrer coverage, sound_pressure og max_sound_pressure for hver layout

    def objective(alarm_coordinates):
        layout_key = tuple(sorted_alarm_positions(alarm_coordinates, n_alarms))
        if layout_key in cache:
            return cache[layout_key][0]

        if solver_name == "FEM":
            sound_pressure, coverage, max_sound_pressure = FEM_solve(
                obstacle_mask,
                alarm_positions=list(layout_key),
            )
        else:
            sound_pressure, coverage, max_sound_pressure = FDM_solve(
                obstacle_mask,
                alarm_positions=list(layout_key),
                debug=debug,
            )
        cache[layout_key] = (coverage, sound_pressure, max_sound_pressure)
        return coverage

    if solver_name == "FEM":
        # Evaluate each candidate directly with FEM to keep comparison faithful
        # while avoiding sklearn/OpenMP interaction in this runtime stack.
        best_key = None
        best_coverage = -np.inf
        for candidate in candidate_layouts:
            iterations += 1
            coverage = objective(candidate)
            key = tuple(sorted_alarm_positions(candidate, n_alarms))
            if coverage > best_coverage:
                best_coverage = coverage
                best_key = key

        if best_key is None:
            raise RuntimeError("No valid FEM candidates were evaluated")
    else:
        from optimalisering2d import expected_improvement, loop

        init_count = min(max(1, init_samples), len(candidate_layouts))
        x_init = candidate_layouts[rng.choice(len(candidate_layouts), size=init_count, replace=False)]

        layout_samples, coverage_samples, _ = loop(
            x_init=x_init,
            sim_func=objective,
            acq_func=expected_improvement,
            domain=candidate_layouts,
            debug=debug,
        )

        iterations = max(0, int(layout_samples.shape[0] - init_count))

        best_idx = int(np.argmax(coverage_samples)) # Get the index of the best alarm layout found by the optimization loop.
        alarm_positions = sorted_alarm_positions(layout_samples[best_idx], n_alarms)
        best_key = tuple(alarm_positions)

    coverage, sound_pressure, max_sound_pressure = cache[best_key]

    if debug:
        elapsed_s = time.perf_counter() - optimization_start
        print(f"[DEBUG] Optimization finished: iterations={iterations}, elapsed_s={elapsed_s:.3f}")

    alarm_positions = list(best_key)

    return OptimizationResult(alarm_positions, sound_pressure, coverage, max_sound_pressure)
