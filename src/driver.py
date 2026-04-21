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
    lippmann_schwinger_anisotropic_cuda,
)

from jaxmaterials.solver.backend import solve_isotropic, _solve_isotropic

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
nx = 32
ny = 32
nz = 32

dtype = jnp.float32
rtol = 1e-20
atol = 1e-4
depth = 0
repeat = 10

grid_spec = GridSpec(nx, ny, nz, Lx, Ly, Lz)
mu, lmbda = initialise_material(grid_spec, dtype=dtype)
zeros = jnp.zeros(mu.shape, dtype=dtype)
stiffness_tensor = jnp.stack(
    3 * [2 * mu + lmbda] + 3 * [mu] + 3 * [lmbda] + 12 * [zeros]
)
epsilon_mean = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
with measure_time("evaluation   (isotropic, Jax)", warmup=True, repeat=repeat) as run:

    def body():
        epsilon_isotropic, sigma, iter = lippmann_schwinger_isotropic_jax(
            mu,
            lmbda,
            epsilon_mean,
            grid_spec,
            maxiter=32,
            depth=depth,
            rtol=rtol,
            atol=atol,
            dtype=dtype,
        )
        epsilon_isotropic.block_until_ready()
        return epsilon_isotropic, iter

    epsilon_isotropic, iter = run(body)

print(f"  number of iterations = {iter}")
print()

with measure_time("evaluation (anisotropic, Jax)", repeat=repeat, warmup=True) as run:

    def body():
        epsilon_anisotropic, sigma, iter = lippmann_schwinger_anisotropic_jax(
            stiffness_tensor,
            epsilon_mean,
            grid_spec,
            maxiter=32,
            depth=depth,
            rtol=rtol,
            atol=atol,
            dtype=dtype,
        )
        epsilon_anisotropic.block_until_ready()
        return epsilon_anisotropic, iter

    epsilon_anisotropic, iter = run(body)
print(f"  number of iterations = {iter}")
print(
    "difference = ",
    jnp.linalg.norm(epsilon_anisotropic - epsilon_isotropic)
    / jnp.linalg.norm(epsilon_isotropic),
)
print()


gpu_available = False
if gpu_available:
    with measure_time(
        "evaluation  (isotropic, CUDA)", repeat=repeat, warmup=True
    ) as run:

        def body():
            epsilon_isotropic, sigma, iter = lippmann_schwinger_isotropic_cuda(
                mu,
                lmbda,
                epsilon_mean,
                grid_spec,
                maxiter=32,
                rtol=rtol,
                atol=atol,
                verbose=0,
            )
            return epsilon_isotropic, iter

        epsilon_isotropic, iter = run(body)
    print(f"  number of iterations = {iter}")
    print()

    with measure_time(
        "evaluation  (anisotropic, CUDA)", repeat=repeat, warmup=True
    ) as run:

        def body():
            epsilon_anisotropic, sigma, iter = lippmann_schwinger_anisotropic_cuda(
                stiffness_tensor,
                epsilon_mean,
                grid_spec,
                maxiter=32,
                rtol=rtol,
                atol=atol,
                verbose=0,
            )
            return epsilon_anisotropic, iter

        epsilon_anisotropic, iter = run(body)
    print(f"  number of iterations = {iter}")
    print(
        "difference = ",
        jnp.linalg.norm(epsilon_anisotropic - epsilon_isotropic)
        / jnp.linalg.norm(epsilon_isotropic),
    )
    print()

with measure_time("backpropagation (isotropic)", repeat=repeat, warmup=True) as run:

    def body():
        def loss_fn(mu, lmbda, epsilon_mean):
            _, sigma = solve_isotropic(mu, lmbda, epsilon_mean, grid_spec)
            return jnp.sum(sigma**2)

        grad_loss = jax.grad(loss_fn, argnums=(0, 1, 2))
        grad_loss(mu, lmbda, epsilon_mean)

    run(body)

with measure_time("gradient (isotropic)", repeat=repeat, warmup=True) as run:

    def body():
        grad_epsilon = jax.jacfwd(lippmann_schwinger_isotropic_jax, argnums=[2])
        dg = grad_epsilon(
            mu,
            lmbda,
            epsilon_mean,
            grid_spec,
            depth=depth,
            rtol=rtol,
            atol=atol,
            dtype=dtype,
        )
        dg[0][0].block_until_ready()

    run(body)
with measure_time("gradient (anisotropic)", repeat=repeat, warmup=True) as run:

    def body():
        grad_epsilon = jax.jacfwd(lippmann_schwinger_anisotropic_jax, argnums=[1])
        dg = grad_epsilon(
            stiffness_tensor,
            epsilon_mean,
            grid_spec,
            depth=depth,
            rtol=rtol,
            atol=atol,
            dtype=dtype,
        )
        dg[0][0].block_until_ready()

    run(body)
print()
