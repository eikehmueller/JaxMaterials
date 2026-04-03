import numpy as np
import jax
from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.utilities import save_to_vtk
from jaxmaterials.utilities import measure_time
from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_isotropic_jax,
    lippmann_schwinger_anisotropic_jax,
    lippmann_schwinger_isotropic_cuda,
)


jax.config.update("jax_enable_x64", True)


def initialise_material(grid_spec, fibre_radius=0.2, dtype=jnp.float64):
    """Material coefficients lambda and mu evaluated at voxel centres

    Returns two arrays of shape (nx,ny,nz)

    :arg grid_spec: grid specification
    :arg fibre_radius: radius of fibre
    :arg dtype: data type
    """
    X, Y, Z = np.meshgrid(
        grid_spec.Lx / grid_spec.nx * (1 / 2 + np.arange(grid_spec.nx)),
        grid_spec.Ly / grid_spec.ny * (1 / 2 + np.arange(grid_spec.ny)),
        grid_spec.Lz / grid_spec.nz * (1 / 2 + np.arange(grid_spec.nz)),
        indexing="ij",
    )
    mu = np.ones(shape=(grid_spec.nx, grid_spec.ny, grid_spec.nz)) + 0.5 * (
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
    )

    lmbda = np.ones(shape=(grid_spec.nx, grid_spec.ny, grid_spec.nz)) + 0.5 * (
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
    )
    return jnp.array(mu, dtype=dtype), jnp.array(lmbda, dtype=dtype)


# Domain size in all three spatial direction
Lx = 1.2
Ly = 0.8
Lz = 0.7
# Number of grid cells in all three spatial directions
nx = 64
ny = 64
nz = 64

dtype = jnp.float32
rtol = 1e-20
atol = 1e-4
depth = 0

grid_spec = GridSpec(nx, ny, nz, Lx, Ly, Lz)
mu, lmbda = initialise_material(grid_spec, dtype=dtype)
zeros = jnp.zeros(mu.shape, dtype=dtype)
stiffness_tensor = jnp.stack(
    3 * [2 * mu + lmbda] + 3 * [mu] + 3 * [lmbda] + 12 * [zeros]
)
# E_mean = jnp.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
E_mean = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5])

with measure_time("evaluation   (isotropic, Jax)"):
    epsilon_isotropic, sigma, iter = lippmann_schwinger_isotropic_jax(
        mu,
        lmbda,
        E_mean,
        grid_spec,
        maxiter=32,
        depth=depth,
        rtol=rtol,
        atol=atol,
    )
    epsilon_isotropic.block_until_ready()
print(f"  number of iterations = {iter}")
print()

with measure_time("evaluation (anisotropic, Jax)"):
    epsilon_anisotropic, sigma, iter = lippmann_schwinger_anisotropic_jax(
        stiffness_tensor,
        E_mean,
        grid_spec,
        maxiter=32,
        depth=depth,
        rtol=rtol,
        atol=atol,
    )
    epsilon_anisotropic.block_until_ready()
print(f"  number of iterations = {iter}")
print(
    "difference = ",
    jnp.linalg.norm(epsilon_anisotropic - epsilon_isotropic)
    / jnp.linalg.norm(epsilon_isotropic),
)
print()


with measure_time("evaluation  (isotropic, CUDA)"):
    epsilon, sigma, iter = lippmann_schwinger_isotropic_cuda(
        mu,
        lmbda,
        E_mean,
        grid_spec,
        maxiter=32,
        rtol=rtol,
        atol=atol,
        verbose=0,
    )
print(f"  number of iterations = {iter}")
print()

with measure_time("gradient"):
    grad_epsilon = jax.jacfwd(lippmann_schwinger_isotropic_jax, argnums=[2])
    dg = grad_epsilon(mu, lmbda, E_mean, grid_spec, depth=depth, rtol=rtol, atol=atol)
    dg[0][0].block_until_ready()
print()
