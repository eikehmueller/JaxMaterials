"""Solve isotropic linear elasticity problem with custom stress-strain relationship"""

import numpy as np
import jax

from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger


def compute_sigma(epsilon, params):
    """Custom implementation of linear elasticity for an isotropic material

    :arg epsilon: strain
    :arg params: dictionary with Lame parameters {"lambda": lambda, "mu": mu}
    """
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = 2 * params["mu"] * epsilon + params["lambda"] * jnp.stack(
        3 * [tr_epsilon] + 3 * [jnp.zeros(epsilon.shape[-3:], dtype=epsilon.dtype)]
    )
    return sigma


# Construct specifications of computational grid
nx = 32
ny = 32
nz = 16

grid_spec = GridSpec(nx, ny, nz, Lx=1.0, Ly=1.0, Lz=0.5)

# Create random Lame parameters mu, lambda and mean strain vector epsilon_bar
rng = np.random.default_rng(seed=47273)

mu = rng.uniform(low=0.8, high=1.1, size=(nx, ny, nz)).astype(np.float32)
lmbda = rng.uniform(low=0.6, high=0.7, size=(nx, ny, nz)).astype(np.float32)
params = {"lambda": lmbda, "mu": mu}
epsilon_bar = rng.normal(size=6).astype(np.float32)


# Lame parameters of homogeneous reference material
ref_params = {
    key: 1 / 2 * (np.min(value) + np.max(value)) for (key, value) in params.items()
}

# forward solve
epsilon, sigma = lippmann_schwinger(
    compute_sigma, params, epsilon_bar, ref_params, grid_spec=grid_spec
)


# sensitivity to input parameters for simple loss function
def loss_fn(params, epsilon_bar):
    epsilon, sigma = lippmann_schwinger(
        compute_sigma, params, epsilon_bar, ref_params, grid_spec=grid_spec
    )
    return jnp.sum(epsilon**2 + sigma**2)


grad_fn = jax.grad(loss_fn, argnums=(0, 1))
g_params, g_epsilon_bar = grad_fn(params, epsilon_bar)
