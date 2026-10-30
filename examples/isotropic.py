"""Solve isotropic linear elasticity problem with readymade solver"""

import numpy as np
import jax

from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger_isotropic

jax.config.update("jax_enable_x64", True)

# Construct specifications of computational grid
nx = 32
ny = 32
nz = 16

grid_spec = GridSpec(Lx=1.0, Ly=1.0, Lz=0.5, nx=nx, ny=ny, nz=nz)

px = 4
py = 4
pz = 2
jax.config.update("jax_num_cpu_devices", px * py * pz)
mesh = jax.make_mesh((px, py, pz), ("X", "Y", "Z"))
jax.set_mesh(mesh)

# Create random Lame parameters mu, lambda and mean strain vector epsilon_bar
rng = np.random.default_rng(seed=47273)

mu = rng.uniform(low=0.8, high=1.1, size=(nx, ny, nz)).astype(np.float32)
lmbda = rng.uniform(low=0.6, high=0.7, size=(nx, ny, nz)).astype(np.float32)

params = {"lambda": lmbda, "mu": mu}
params = jax.device_put(params, jax.P(*mesh.axis_names))

epsilon_bar = rng.normal(size=6).astype(np.float32)

# forward solve
epsilon, sigma = lippmann_schwinger_isotropic(params, epsilon_bar, grid_spec, verbose=2)


# sensitivity to input parameters for simple loss function
def loss_fn(params, epsilon_bar):
    epsilon, sigma = lippmann_schwinger_isotropic(
        params, epsilon_bar, grid_spec=grid_spec, verbose=2
    )
    return jnp.sum(epsilon**2 + sigma**2)


grad_fn = jax.grad(loss_fn, argnums=(0, 1))
g_params, g_epsilon_bar = grad_fn(params, epsilon_bar)
