"""Simple Helmholtz equation solve using FEniCSx (dolfinx).

Solves
    -Delta(u) - k^2 u = f   in Omega = (0,1)x(0,1)
    u = 0                   on partial Omega

with a manufactured exact solution u_exact = sin(pi x) sin(pi y).
"""

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, geometry, io, mesh
from dolfinx.fem.petsc import LinearProblem
from typing import Sequence, Tuple


def _wall_cell_indicator(domain, obstacle_mask: np.ndarray) -> np.ndarray:
    """Return per-cell indicator (1 for wall, 0 for free) from room mask.

    The room format used in this project is obstacle_mask[x, y] with boolean
    entries. This helper maps each triangle cell midpoint to that grid.
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    c2v = domain.topology.connectivity(tdim, 0)
    x_dim, y_dim = obstacle_mask.shape

    cell_map = domain.topology.index_map(tdim)
    n_cells = cell_map.size_local + cell_map.num_ghosts
    indicator = np.zeros(n_cells, dtype=np.int8)

    coords = domain.geometry.x
    for cell in range(n_cells):
        vertices = c2v.links(cell)
        midpoint = coords[vertices].mean(axis=0)
        ix = int(np.clip(np.floor(midpoint[0]), 0, x_dim - 1))
        iy = int(np.clip(np.floor(midpoint[1]), 0, y_dim - 1))
        indicator[cell] = 1 if obstacle_mask[ix, iy] else 0
    return indicator


def _cellwise_scalar(domain, wall_indicator: np.ndarray, air_value: float, wall_value: float) -> fem.Function:
    """Build DG0 coefficient equal to air_value/free and wall_value/wall."""
    Q = fem.functionspace(domain, ("DG", 0))
    coeff = fem.Function(Q)

    tdim = domain.topology.dim
    cell_map = domain.topology.index_map(tdim)
    n_cells = cell_map.size_local + cell_map.num_ghosts
    values = coeff.x.array

    for cell in range(n_cells):
        dof = Q.dofmap.cell_dofs(cell)[0]
        values[dof] = wall_value if wall_indicator[cell] == 1 else air_value
    coeff.x.scatter_forward()
    return coeff


def _alarm_source(x, alarm_positions: Sequence[Tuple[int, int]], strength: float, spread: float):
    """Gaussian source term from integer alarm grid coordinates."""
    source = 0.0
    for ax, ay in alarm_positions:
        cx = float(ax) + 0.5
        cy = float(ay) + 0.5
        source += strength * ufl.exp(-((x[0] - cx) ** 2 + (x[1] - cy) ** 2) / (spread * spread))
    return source


def _sample_on_room_grid(u_h: fem.Function, x_dim: int, y_dim: int) -> np.ndarray:
    """Evaluate scalar FEM solution at room-cell centers and return [x, y] map."""
    domain = u_h.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)

    out = np.zeros((x_dim, y_dim), dtype=np.float64)
    for ix in range(x_dim):
        for iy in range(y_dim):
            point = np.array([[ix + 0.5, iy + 0.5, 0.0]], dtype=np.float64)
            candidates = geometry.compute_collisions_points(tree, point)
            colliding = geometry.compute_colliding_cells(domain, candidates, point)
            links = colliding.links(0)
            if len(links) == 0:
                continue
            value = u_h.eval(point, np.array([links[0]], dtype=np.int32))
            out[ix, iy] = float(np.ravel(value)[0])
    return out


def solve_room_acoustics(
    obstacle_mask: np.ndarray,
    alarm_positions: Sequence[Tuple[int, int]],
    frequency_hz: float = 1200.0,
    source_strength: float = 2200.0,
    source_spread: float = 1.8,
    threshold: float = 3.0,
    wall_reflectivity: float = 0.85,
    wall_absorption: float = 0.15,
):
    """Solve room pressure map with the same room/wall format used in src/rooms.py.

    Args:
        obstacle_mask: Bool array [x_dim, y_dim], True where wall/obstacle exists.
        alarm_positions: List of integer (x, y) positions on the same grid.
    Returns:
        pressure_map: ndarray [x_dim, y_dim]
        covered_mask: ndarray [x_dim, y_dim] bool
        coverage_percent: scalar coverage in free space
    """
    obstacle_mask = np.asarray(obstacle_mask, dtype=bool)
    x_dim, y_dim = obstacle_mask.shape

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([float(x_dim), float(y_dim)])],
        [x_dim, y_dim],
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", 1))

    wall_indicator = _wall_cell_indicator(domain, obstacle_mask)

    rho_air = 1.2
    c_air = 343.0
    kappa_air = rho_air * c_air * c_air

    reflectivity = float(np.clip(wall_reflectivity, 0.0, 1.0))
    absorption = float(np.clip(wall_absorption, 0.0, 1.0))
    transmission = max(0.05, 1.0 - reflectivity)

    rho_wall = rho_air * (1.0 + 4.0 * reflectivity)
    c_wall = c_air * transmission
    kappa_wall = rho_wall * c_wall * c_wall

    sigma_air = 0.8
    sigma_wall = sigma_air + 25.0 * absorption

    rho = _cellwise_scalar(domain, wall_indicator, rho_air, rho_wall)
    kappa = _cellwise_scalar(domain, wall_indicator, kappa_air, kappa_wall)
    sigma = _cellwise_scalar(domain, wall_indicator, sigma_air, sigma_wall)

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(0.0, boundary_dofs, V)

    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    omega = 2.0 * np.pi * float(frequency_hz)
    s = _alarm_source(x, alarm_positions, source_strength, source_spread)

    a = ((1.0 / rho) * ufl.dot(ufl.grad(p), ufl.grad(v)) + ((omega * omega) / kappa + sigma) * p * v) * ufl.dx
    L = s * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="room_helmholtz_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    p_h = problem.solve()

    pressure_map = _sample_on_room_grid(p_h, x_dim, y_dim)
    free_mask = ~obstacle_mask
    covered_mask = (pressure_map >= threshold) & free_mask

    free_cells = int(np.count_nonzero(free_mask))
    covered_cells = int(np.count_nonzero(covered_mask))
    coverage_percent = 100.0 * covered_cells / max(1, free_cells)

    return pressure_map, covered_mask, coverage_percent


def main() -> None:
    # Wavenumber
    k = 10.0

    # Mesh and first-order Lagrange space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Dirichlet boundary condition u = 0 on all boundaries
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(0.0, boundary_dofs, V)

    # Trial/test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Exact solution and source term for verification
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = (2.0 * np.pi * np.pi - k * k) * u_exact_ufl

    # Weak form: (grad u, grad v) - k^2 (u, v) = (f, v)
    a = (ufl.dot(ufl.grad(u), ufl.grad(v)) - k * k * u * v) * ufl.dx
    L = f * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="helmholtz_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    u_h = problem.solve()
    u_h.name = "u"

    # Error norm compared to manufactured solution
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda x_in: np.sin(np.pi * x_in[0]) * np.sin(np.pi * x_in[1]))

    error_form = fem.form((u_h - u_exact) ** 2 * ufl.dx)
    error_l2_local = fem.assemble_scalar(error_form)
    error_l2 = np.sqrt(domain.comm.allreduce(error_l2_local, op=MPI.SUM))
    if domain.comm.rank == 0:
        print(f"L2 error: {error_l2:.6e}")

    # Save result for visualization in ParaView
    with io.XDMFFile(domain.comm, "helmholtz_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_h)


if __name__ == "__main__":
    main()