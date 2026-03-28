"""FEM solver for fire-alarm acoustic coverage in 2D rooms.

This module uses FEniCSx (dolfinx) to solve a frequency-domain acoustic model
where:
- alarms are modeled as smoothed point sources,
- internal walls are modeled as material regions with different properties,
- coverage is defined by a pressure threshold.

PDE (real-valued damped Helmholtz-like model):
    -div((1/rho) grad(p)) + (omega^2 / K + sigma) p = s

The wall material contrast (rho, K, sigma) creates partial reflection and
transmission at interfaces.
"""

from dataclasses import dataclass
import inspect
from typing import List, Sequence, Tuple, Union

import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc


@dataclass(frozen=True)
class PointSource:
    x: float
    y: float
    strength: float
    spread: float = 0.08


@dataclass(frozen=True)
class WallRegion:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    transmission: float
    reflectivity: float


WallInput = Union[WallRegion, Sequence[float]]


def plot_fire_alarm_mesh_result(
    p_h: fem.Function,
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    walls: Sequence[WallInput],
    alarms: Sequence[PointSource],
    threshold: float,
    title: str = "FEM Fire-Alarm Intensity",
    show_mesh: bool = True,
    output_path: str | None = None,
    show_plot: bool = True,
    color_scale: str = "log",
    clip_percentiles: Tuple[float, float] = (2.0, 99.0),
) -> None:
    """Plot sampled intensity with FEM mesh, walls, alarms and threshold contour."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    import matplotlib.tri as mtri

    wall_regions = _normalize_walls(walls)
    fig, ax = plt.subplots(figsize=(10, 4.8))

    # Use amplitude and robust color scaling so areas away from source remain readable.
    amp = np.abs(P)
    p_lo, p_hi = clip_percentiles
    vmin = float(np.percentile(amp, p_lo))
    vmax = float(np.percentile(amp, p_hi))
    vmin = max(vmin, 1.0e-12)
    if vmax <= vmin:
        vmax = max(float(np.max(amp)), vmin * 10.0)

    scale = color_scale.lower().strip()
    if scale == "log":
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cbar_label = "|Pressure| (log scale)"
    elif scale == "linear":
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = "|Pressure|"
    else:
        raise ValueError("color_scale must be 'log' or 'linear'")

    pcm = ax.pcolormesh(X, Y, amp, shading="auto", cmap="viridis", norm=norm)
    fig.colorbar(pcm, ax=ax, label=cbar_label)

    if show_mesh:
        msh = p_h.function_space.mesh
        tdim = msh.topology.dim
        msh.topology.create_connectivity(tdim, 0)
        c_to_v = msh.topology.connectivity(tdim, 0)

        points = msh.geometry.x
        num_cells = msh.topology.index_map(tdim).size_local
        triangles = [c_to_v.links(cell) for cell in range(num_cells)]
        if len(triangles) > 0:
            tri = mtri.Triangulation(points[:, 0], points[:, 1], np.asarray(triangles, dtype=np.int32))
            ax.triplot(tri, color="white", linewidth=0.25, alpha=0.35)

    if threshold is not None:
        ax.contour(X, Y, amp, levels=[threshold], colors=["#ffcc00"], linewidths=1.4)

    for wall in wall_regions:
        rect = patches.Rectangle(
            (wall.x_min, wall.y_min),
            wall.x_max - wall.x_min,
            wall.y_max - wall.y_min,
            facecolor="#2f2f2f",
            edgecolor="#111111",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.add_patch(rect)

    if alarms:
        ax.scatter(
            [a.x for a in alarms],
            [a.y for a in alarms],
            s=45,
            c="#ff4d4d",
            edgecolors="black",
            linewidths=0.6,
            label="Alarms",
            zorder=5,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(X)), float(np.max(X)))
    ax.set_ylim(float(np.min(Y)), float(np.max(Y)))
    if alarms:
        ax.legend(loc="upper right")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved FEM visualization to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _in_rect(x: float, y: float, rect: WallRegion) -> bool:
    return rect.x_min <= x <= rect.x_max and rect.y_min <= y <= rect.y_max


def _normalize_walls(walls: Sequence[WallInput]) -> List[WallRegion]:
    normalized: List[WallRegion] = []
    for wall in walls:
        if isinstance(wall, WallRegion):
            normalized.append(wall)
            continue

        vals = tuple(float(v) for v in wall)
        if len(vals) == 4:
            # Compatible with (x, y, width, height) room layout tuples.
            x, y, width, height = vals
            normalized.append(
                WallRegion(
                    x_min=x,
                    x_max=x + width,
                    y_min=y,
                    y_max=y + height,
                    transmission=0.35,
                    reflectivity=0.65,
                )
            )
        elif len(vals) == 6:
            normalized.append(
                WallRegion(
                    x_min=vals[0],
                    x_max=vals[1],
                    y_min=vals[2],
                    y_max=vals[3],
                    transmission=vals[4],
                    reflectivity=vals[5],
                )
            )
        else:
            raise ValueError(
                "Wall input must be WallRegion or a tuple/list of length 4 "
                "(x, y, width, height) or 6 "
                "(x_min, x_max, y_min, y_max, transmission, reflectivity)."
            )
    return normalized


def _mark_walls(msh: mesh.Mesh, walls: List[WallRegion]) -> np.ndarray:
    tdim = msh.topology.dim
    num_local_cells = msh.topology.index_map(tdim).size_local
    cell_markers = np.zeros(num_local_cells, dtype=np.int32)

    if not walls:
        return cell_markers

    def in_any_wall(x: np.ndarray) -> np.ndarray:
        inside = np.zeros(x.shape[1], dtype=bool)
        for wall in walls:
            inside |= (
                (x[0] >= wall.x_min)
                & (x[0] <= wall.x_max)
                & (x[1] >= wall.y_min)
                & (x[1] <= wall.y_max)
            )
        return inside

    wall_cells = mesh.locate_entities(msh, tdim, in_any_wall)
    cell_markers[wall_cells] = 1
    return cell_markers


def _build_cellwise_scalar(
    msh: mesh.Mesh,
    cell_markers: np.ndarray,
    values: Tuple[float, float],
) -> fem.Function:
    q_space = fem.functionspace(msh, ("DG", 0))
    coeff = fem.Function(q_space)
    coeff_array = coeff.x.array
    coeff_array[:] = values[0]
    coeff_array[: cell_markers.shape[0]][cell_markers == 1] = values[1]
    coeff.x.scatter_forward()
    return coeff


def _source_function(v_space: fem.FunctionSpace, sources: List[PointSource]) -> fem.Function:
    source = fem.Function(v_space)

    def gaussian_sum(x: np.ndarray) -> np.ndarray:
        values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
        for src in sources:
            dx = x[0] - src.x
            dy = x[1] - src.y
            values += src.strength * np.exp(-(dx * dx + dy * dy) / (src.spread * src.spread))
        return values

    source.interpolate(gaussian_sum)
    return source


def _sample_solution_on_grid(
    msh: mesh.Mesh,
    solution: fem.Function,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    points = np.column_stack(
        [
            X.ravel(),
            Y.ravel(),
            np.zeros(X.size, dtype=np.float64),
        ]
    )

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    try:
        candidate_cells = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, points)
    except (TypeError, RuntimeError, ValueError):
        # Some dolfinx versions expect points with shape (3, n) instead of (n, 3).
        points = np.ascontiguousarray(points.T)
        candidate_cells = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, points)

    sampled = np.zeros(X.size, dtype=np.float64)
    point_count = X.size
    for i in range(point_count):
        cells_i = colliding_cells.links(i)
        if len(cells_i) > 0:
            point = points[:, i] if points.shape[0] == 3 else points[i, :]
            point_eval = point[np.newaxis, :]
            cell_eval = np.array([cells_i[0]], dtype=np.int32)
            eval_value = np.asarray(solution.eval(point_eval, cell_eval))
            if eval_value.size > 0:
                sampled[i] = float(eval_value.reshape(-1)[0])

    return sampled.reshape(X.shape)


def _make_linear_problem(a, L, bcs):
    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "ksp_rtol": 1.0e-8,
        "ksp_max_it": 2000,
    }
    signature = inspect.signature(LinearProblem.__init__)

    kwargs = {"bcs": bcs, "petsc_options": petsc_options}
    if "petsc_options_prefix" in signature.parameters:
        kwargs["petsc_options_prefix"] = "fire_alarm_"

    return LinearProblem(a, L, **kwargs)


def _build_boundary_bcs(msh: mesh.Mesh, V: fem.FunctionSpace, boundary_mode: str):
    mode = boundary_mode.lower().strip()
    if mode == "reflective":
        # Natural Neumann boundary (no explicit BC) approximates rigid outer walls.
        return []
    if mode == "absorbing":
        fdim = msh.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            msh,
            fdim,
            lambda x: np.ones(x.shape[1], dtype=bool),
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        return [fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)]

    raise ValueError("boundary_mode must be 'reflective' or 'absorbing'")


def solve_fire_alarm_coverage(
    room_width: float,
    room_height: float,
    nx: int,
    ny: int,
    alarms: List[PointSource],
    walls: Sequence[WallInput],
    frequency_hz: float,
    threshold: float,
    boundary_mode: str = "reflective",
) -> Tuple[fem.Function, np.ndarray, np.ndarray, np.ndarray]:
    """Solve pressure field and return pressure + grid coverage.

    Returns:
        p_h: FEniCS solution function
        X, Y: meshgrid arrays (for plotting)
        covered: boolean coverage mask on sampling grid
    """
    wall_regions = _normalize_walls(walls)

    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([room_width, room_height], dtype=np.float64)],
        [nx, ny],
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("CG", 1))

    cell_markers = _mark_walls(msh, wall_regions)

    # Air baseline constants
    rho_air = 1.2
    c_air = 343.0
    kappa_air = rho_air * c_air * c_air

    if wall_regions:
        mean_trans = float(np.mean([w.transmission for w in wall_regions]))
        mean_refl = float(np.mean([w.reflectivity for w in wall_regions]))
    else:
        mean_trans = 1.0
        mean_refl = 0.0

    # Wall properties chosen to produce partial transmission/reflection.
    # Lower transmission and higher reflectivity increase contrast and damping.
    rho_wall = rho_air * (1.0 + 4.0 * max(0.0, min(1.0, mean_refl)))
    c_wall = c_air * max(0.2, min(1.0, mean_trans))
    kappa_wall = rho_wall * c_wall * c_wall

    # Lower damping levels keep physically reasonable amplitudes in enclosed rooms.
    sigma_air = 0.02
    sigma_wall = 0.02 + 2.0 * (1.0 - max(0.0, min(1.0, mean_trans)))

    rho = _build_cellwise_scalar(msh, cell_markers, values=(rho_air, rho_wall))
    kappa = _build_cellwise_scalar(msh, cell_markers, values=(kappa_air, kappa_wall))
    sigma = _build_cellwise_scalar(msh, cell_markers, values=(sigma_air, sigma_wall))

    omega = 2.0 * np.pi * frequency_hz
    source = _source_function(V, alarms)

    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (1.0 / rho) * ufl.dot(ufl.grad(p), ufl.grad(v)) * ufl.dx + (
        ((omega * omega) / kappa + sigma) * p * v * ufl.dx
    )
    L = source * v * ufl.dx

    bcs = _build_boundary_bcs(msh, V, boundary_mode)

    problem = _make_linear_problem(a, L, bcs)
    p_h = problem.solve()

    # Sample pressure amplitude on a regular grid for coverage statistics.
    sample_nx = max(50, nx)
    sample_ny = max(50, ny)
    x = np.linspace(0.0, room_width, sample_nx)
    y = np.linspace(0.0, room_height, sample_ny)
    X, Y = np.meshgrid(x, y)

    P = _sample_solution_on_grid(msh, p_h, X, Y)

    covered = np.abs(P) >= threshold
    return p_h, X, Y, covered


def solve_fire_alarm_intensity_map(
    room_width: float,
    room_height: float,
    nx: int,
    ny: int,
    alarms: List[PointSource],
    walls: Sequence[WallInput],
    frequency_hz: float,
    threshold: float,
    boundary_mode: str = "reflective",
) -> Tuple[fem.Function, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve pressure field and return pressure + intensity map + coverage.

    Returns:
        p_h: FEniCSx solution function
        X, Y: meshgrid arrays (for plotting)
        P: sampled pressure/intensity map on the same grid as X, Y
        covered: boolean coverage mask on the sampling grid
    """
    wall_regions = _normalize_walls(walls)

    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([room_width, room_height], dtype=np.float64)],
        [nx, ny],
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("CG", 1))

    cell_markers = _mark_walls(msh, wall_regions)

    rho_air = 1.2
    c_air = 343.0
    kappa_air = rho_air * c_air * c_air

    if wall_regions:
        mean_trans = float(np.mean([w.transmission for w in wall_regions]))
        mean_refl = float(np.mean([w.reflectivity for w in wall_regions]))
    else:
        mean_trans = 1.0
        mean_refl = 0.0

    rho_wall = rho_air * (1.0 + 4.0 * max(0.0, min(1.0, mean_refl)))
    c_wall = c_air * max(0.2, min(1.0, mean_trans))
    kappa_wall = rho_wall * c_wall * c_wall

    sigma_air = 0.02
    sigma_wall = 0.02 + 2.0 * (1.0 - max(0.0, min(1.0, mean_trans)))

    rho = _build_cellwise_scalar(msh, cell_markers, values=(rho_air, rho_wall))
    kappa = _build_cellwise_scalar(msh, cell_markers, values=(kappa_air, kappa_wall))
    sigma = _build_cellwise_scalar(msh, cell_markers, values=(sigma_air, sigma_wall))

    omega = 2.0 * np.pi * frequency_hz
    source = _source_function(V, alarms)

    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (1.0 / rho) * ufl.dot(ufl.grad(p), ufl.grad(v)) * ufl.dx + (
        ((omega * omega) / kappa + sigma) * p * v * ufl.dx
    )
    L = source * v * ufl.dx

    bcs = _build_boundary_bcs(msh, V, boundary_mode)

    problem = _make_linear_problem(a, L, bcs)
    p_h = problem.solve()

    sample_nx = max(50, nx)
    sample_ny = max(50, ny)
    x = np.linspace(0.0, room_width, sample_nx)
    y = np.linspace(0.0, room_height, sample_ny)
    X, Y = np.meshgrid(x, y)

    P = _sample_solution_on_grid(msh, p_h, X, Y)
    covered = np.abs(P) >= threshold
    return p_h, X, Y, P, covered


def example_run() -> None:
    alarms = [
        PointSource(x=2.0, y=2.5, strength=2500.0, spread=0.22),
        PointSource(x=8.0, y=2.5, strength=2200.0, spread=0.22),
    ]

    walls = [
        WallRegion(
            x_min=4.8,
            x_max=5.2,
            y_min=0.5,
            y_max=4.5,
            transmission=0.35,
            reflectivity=0.65,
        )
    ]

    p_h, X, Y, P, covered = solve_fire_alarm_intensity_map(
        room_width=10.0,
        room_height=5.0,
        nx=120,
        ny=60,
        alarms=alarms,
        walls=walls,
        frequency_hz=3100.0,
        threshold=0.12,
        boundary_mode="reflective",
    )

    coverage_ratio = float(np.mean(covered))
    max_pressure = float(np.max(np.abs(P)))
    print(f"Coverage ratio: {coverage_ratio:.2%}")
    print(f"Peak |p| on sampled grid: {max_pressure:.3f}")
    plot_fire_alarm_mesh_result(
        p_h=p_h,
        X=X,
        Y=Y,
        P=P,
        walls=walls,
        alarms=alarms,
        threshold=0.12,
        output_path="fem_fire_alarm_coverage.png",
        show_plot=False,
        color_scale="log",
        clip_percentiles=(2.0, 99.2),
    )


if __name__ == "__main__":
    example_run()