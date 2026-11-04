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

jax.distributed.initialize()

# Construct specifications of computational grid
nx = 48
ny = 48
nz = 24

# Parallel decomposition
px = 2
py = 3

grid_spec = GridSpec(Lx=1.0, Ly=1.0, Lz=0.5, nx=nx, ny=ny, nz=nz, px=px, py=py)

mesh = jax.make_mesh((px, py), axis_names=("x", "y"))
sharding = jax.sharding.NamedSharding(mesh, jax.P("x", "y", None))

# Create random Lame parameters mu, lambda and mean strain vector epsilon_bar
rng = np.random.default_rng(seed=47273)

mu = rng.uniform(low=0.8, high=1.1, size=grid_spec.extents)
lmbda = rng.uniform(low=0.6, high=0.7, size=grid_spec.extents)

with jax.set_mesh(mesh):
    params = {
        key: jax.make_array_from_callback(
            value.shape, sharding, lambda index, value=value: value[index]
        )
        for key, value in [["lambda", lmbda], ["mu", mu]]
    }

    epsilon_bar = rng.normal(size=6)

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
