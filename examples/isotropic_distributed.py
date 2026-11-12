"""Solve isotropic linear elasticity problem with readymade solver

Run this code with

    mpirun -n 6 isotropic_distribution.py
"""

import numpy as np
import jax

from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger_isotropic

jax.config.update("jax_enable_x64", True)

jax.distributed.initialize(cluster_detection_method="mpi4py")

# Construct specifications of computational grid
nx = 48
ny = 48
nz = 24

# Parallel decomposition
px = 2
py = 3

grid_spec = GridSpec(Lx=1.0, Ly=1.0, Lz=0.5, nx=nx, ny=ny, nz=nz, px=px, py=py)

mesh = jax.sharding.Mesh(
    np.array(jax.devices()).reshape(px, py),
    axis_names=("x", "y"),
    axis_types=(jax.sharding.AxisType.Explicit,) * 2,
)


sharding = jax.sharding.NamedSharding(mesh, jax.P("x", "y", None))

# Create random Lame parameters mu, lambda and mean strain vector epsilon_bar

with jax.set_mesh(mesh):
    rank = jax.process_index()
    rng_local = np.random.default_rng(rank + 47273)
    params = {
        "mu": jax.make_array_from_callback(
            grid_spec.extents,
            sharding,
            lambda _: rng_local.uniform(
                low=0.8, high=1.1, size=grid_spec.local_extents
            ),
        ),
        "lambda": jax.make_array_from_callback(
            grid_spec.extents,
            sharding,
            lambda _: rng_local.uniform(
                low=0.6, high=0.7, size=grid_spec.local_extents
            ),
        ),
    }

    rng_global = np.random.default_rng(47273)
    epsilon_bar = rng_global.normal(size=6)

    # forward solve
    epsilon, sigma = lippmann_schwinger_isotropic(
        params, epsilon_bar, grid_spec, verbose=2
    )

    # sensitivity to input parameters for simple loss function
    def loss_fn(params, epsilon_bar):
        epsilon, sigma = lippmann_schwinger_isotropic(
            params, epsilon_bar, grid_spec=grid_spec, verbose=2
        )
        return jnp.sum(epsilon**2 + sigma**2)

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    g_params, g_epsilon_bar = grad_fn(params, epsilon_bar)

jax.distributed.shutdown()
