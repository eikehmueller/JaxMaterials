import numpy as np
import jax
from jax import numpy as jnp

from jaxmaterials.common import get_grid_spec
from jaxmaterials.utilities import measure_time
from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_isotropic,
    lippmann_schwinger_anisotropic,
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
Ly = 1.1
Lz = 0.7
# Number of grid cells in all three spatial directions
nx = 64
ny = 64
nz = 48

repeat = 10
grid_spec = get_grid_spec(Lx, Ly, Lz, nx=nx, ny=ny, nz=nz)

for dtype in (jnp.float32, jnp.float64):
    precision = "single precision" if dtype == jnp.float32 else "double precision"
    mu, lmbda = initialise_material(grid_spec, dtype=dtype)
    mu_ref = 1 / 2 * (np.min(mu) + np.max(mu))
    lmbda_ref = 1 / 2 * (np.min(lmbda) + np.max(lmbda))
    zeros = jnp.zeros(mu.shape, dtype=dtype)
    stiffness_tensor = jnp.stack(
        3 * [2 * mu + lmbda] + 3 * [mu] + 3 * [lmbda] + 12 * [zeros]
    )
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    with measure_time(
        f"evaluation   (isotropic, {precision})", warmup=True, repeat=repeat
    ) as (run, it):

        def body():
            epsilon_isotropic, sigma = lippmann_schwinger_isotropic(
                {"mu": mu, "lambda": lmbda}, epsilon_bar, grid_spec, verbose=it() == 0
            )
            epsilon_isotropic.block_until_ready()
            return epsilon_isotropic

        epsilon_isotropic = run(body)
    print()

    epsilon_isotropic, sigma = lippmann_schwinger_isotropic(
        {"mu": mu, "lambda": lmbda}, epsilon_bar, grid_spec, verbose=1
    )
    epsilon_isotropic.block_until_ready()

    with measure_time(
        f"evaluation (anisotropic, {precision})", repeat=repeat, warmup=True
    ) as (run, it):

        def body():
            epsilon_anisotropic, sigma = lippmann_schwinger_anisotropic(
                {"stiffness_tensor": stiffness_tensor},
                epsilon_bar,
                {"mu": mu_ref, "lambda": lmbda_ref},
                grid_spec,
                verbose=it() == 0,
            )
            epsilon_anisotropic.block_until_ready()
            return epsilon_anisotropic

        epsilon_anisotropic = run(body)
    print(
        "difference = ",
        jnp.linalg.norm(epsilon_anisotropic - epsilon_isotropic)
        / jnp.linalg.norm(epsilon_isotropic),
    )
    print()

    with measure_time(
        f"gradient (isotropic, {precision})", repeat=repeat, warmup=True
    ) as (run, it):

        def body():
            def loss_fn(mu, lmbda, epsilon_bar):
                epsilon, sigma = lippmann_schwinger_isotropic(
                    {"mu": mu, "lambda": lmbda},
                    epsilon_bar,
                    grid_spec,
                    verbose=it() == 0,
                )
                return jnp.sum(sigma**2)

            g = jax.grad(loss_fn, argnums=(0, 1, 2))(mu, lmbda, epsilon_bar)
            g[0][0].block_until_ready()

        run(body)
    print()

    with measure_time(
        f"gradient (anisotropic, {precision})", repeat=repeat, warmup=True
    ) as (run, it):

        def body():
            def loss_fn(stiffness_tensor, epsilon_bar):
                epsilon, sigma = lippmann_schwinger_anisotropic(
                    {"stiffness_tensor": stiffness_tensor},
                    epsilon_bar,
                    {"mu": mu_ref, "lambda": lmbda_ref},
                    grid_spec,
                    verbose=it() == 0,
                )
                return jnp.sum(sigma**2)

            g = jax.grad(loss_fn, argnums=(0, 1))(stiffness_tensor, epsilon_bar)
            g[0][0].block_until_ready()

        run(body)
    print()

gpu_available = False
if gpu_available:
    dtype = jnp.float32
    mu, lmbda = initialise_material(grid_spec, dtype=dtype)
    zeros = jnp.zeros(mu.shape, dtype=dtype)
    stiffness_tensor = jnp.stack(
        3 * [2 * mu + lmbda] + 3 * [mu] + 3 * [lmbda] + 12 * [zeros]
    )
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)

    with measure_time("evaluation  (isotropic, CUDA)", repeat=repeat, warmup=True) as (
        run,
        it,
    ):

        def body():
            epsilon_isotropic, sigma = lippmann_schwinger_isotropic(
                mu, lmbda, epsilon_bar, grid_spec, use_cuda=True, verbose=it() == 0
            )
            return epsilon_isotropic

        epsilon_isotropic, its = run(body)
    print()

    with measure_time(
        "evaluation  (anisotropic, CUDA)", repeat=repeat, warmup=True
    ) as (run, it):

        def body():

            epsilon_anisotropic, sigma = lippmann_schwinger_anisotropic(
                stiffness_tensor,
                epsilon_bar,
                grid_spec,
                use_cuda=True,
                verbose=it() == 0,
            )
            return epsilon_anisotropic

        epsilon_anisotropic = run(body)
    print(
        "difference = ",
        jnp.linalg.norm(epsilon_anisotropic - epsilon_isotropic)
        / jnp.linalg.norm(epsilon_isotropic),
    )
    print()
