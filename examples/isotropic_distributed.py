"""Solve isotropic linear elasticity problem with readymade solver

Run this code with

    mpirun -n 6 isotropic_distribution.py
"""

import numpy as np
import jax
import json
import pyvista as pv

from jax import numpy as jnp
from jax.experimental.multihost_utils import process_allgather

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger_isotropic

jax.config.update("jax_enable_x64", True)

jax.distributed.initialize(cluster_detection_method="mpi4py")


def distributed_ids(grid_spec):
    nx_local, ny_local, nz_local = grid_spec.local_extents

    i_x, i_y = np.unravel_index(jax.process_index(), grid_spec.pdims)

    X, Y = np.meshgrid(
        grid_spec.dx * (1 / 2 + np.arange(i_x * nx_local, (i_x + 1) * nx_local)),
        grid_spec.dy * (1 / 2 + np.arange(i_y * ny_local, (i_y + 1) * ny_local)),
        indexing="ij",
    )
    mask_2d = np.zeros(shape=(nx_local, ny_local), dtype=np.uint8)
    with open("fibre_locations.json", encoding="utf8") as f:
        fibre_locations = json.load(f)
        fibre_positions = np.stack(
            [fibre_locations["positions_x"], fibre_locations["positions_y"]], axis=1
        )
        fibre_radii = fibre_locations["radii"]
    offset_X = [-grid_spec.Lx, 0, +grid_spec.Lx]
    offset_Y = [-grid_spec.Ly, 0, +grid_spec.Ly]
    # Ensure periodicity
    for p, r in zip(fibre_positions, fibre_radii):
        for ox in offset_X:
            for oy in offset_Y:
                mask_2d[(X + ox - p[0]) ** 2 + (Y + oy - p[1]) ** 2 < r**2] = 1
    ids = np.repeat(mask_2d[:, :, None], nz_local, axis=2)
    return ids


# Construct specifications of computational grid
nx = 4 * 48
ny = 4 * 48
nz = 4 * 24

# Parallel decomposition
px = 2
py = 3

# Save to .vtk file?
save_to_vtk = True
compute_gradient = False

grid_spec = GridSpec(Lx=1.0, Ly=1.0, Lz=0.5, nx=nx, ny=ny, nz=nz, px=px, py=py)

mesh = jax.sharding.Mesh(
    np.array(jax.devices()).reshape(px, py),
    axis_names=("x", "y"),
    axis_types=(jax.sharding.AxisType.Explicit,) * 2,
)


sharding = jax.sharding.NamedSharding(mesh, jax.P("x", "y", None))

ids = distributed_ids(grid_spec)

# Create random Lame parameters mu, lambda and mean strain vector epsilon_bar

lame_coefficients = {
    "mu": {
        "matrix": 1,
        "fibre": 0.01,
    },
    "lambda": {
        "matrix": 1,
        "fibre": 0.01,
    },
}

with jax.set_mesh(mesh):
    rank = jax.process_index()
    params = {
        key: jax.make_array_from_process_local_data(
            sharding,
            (ids * value["fibre"] + (1 - ids) * value["matrix"]).astype(jnp.float64),
        )
        for key, value in lame_coefficients.items()
    }

    epsilon_bar = np.asarray([1.0, 1.414, 3.142, 2.718, 1.732, 2.236])

    # forward solve
    epsilon, sigma = lippmann_schwinger_isotropic(
        params, epsilon_bar, grid_spec, depth=4, tol=1.0e-4, verbose=2
    )
    if save_to_vtk:
        grid = pv.ImageData(
            dimensions=np.asarray(grid_spec.extents) + 1,
            spacing=grid_spec.grid_spacings,
            origin=(0, 0, 0),
        )

        grid.cell_data["mu"] = process_allgather(params["mu"], tiled=True).ravel(
            order="F"
        )
        grid.cell_data["lambda"] = process_allgather(
            params["lambda"], tiled=True
        ).ravel(order="F")
        for k, idx in enumerate(("00", "11", "22", "01", "02", "12")):
            grid.cell_data[f"epsilon_{idx}"] = process_allgather(
                epsilon[k, ...], tiled=True
            ).ravel(order="F")
            grid.cell_data[f"sigma_{idx}"] = process_allgather(
                sigma[k, ...], tiled=True
            ).ravel(order="F")
        grid.save("lame_coefficients.vtk", binary=True)

    # sensitivity to input parameters for simple loss function
    def loss_fn(params, epsilon_bar):
        epsilon, sigma = lippmann_schwinger_isotropic(
            params, epsilon_bar, grid_spec=grid_spec, verbose=2
        )
        return jnp.sum(epsilon**2 + sigma**2)

    if compute_gradient:
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        g_params, g_epsilon_bar = grad_fn(params, epsilon_bar)

jax.distributed.shutdown()
