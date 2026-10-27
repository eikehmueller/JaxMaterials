import argparse
import contextlib
import os
import sys

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxmaterials.common import get_grid_spec
from jaxmaterials.solver.dfft import distributed_fft
from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger,
    lippmann_schwinger_anisotropic,
    lippmann_schwinger_isotropic,
)
from jaxmaterials.utilities import measure_time

jax.config.update("jax_enable_x64", True)


'''
How to run the profiling scripts:

python src/profile.py --mode single --grid 256 256 256
python src/profile.py --mode multi-gpu --grid 256 256 256 [--mesh 2 2]
srun --ntasks=N --gpus-per-task=1 python src/profile.py --mode multi-node --grid 256 256 256
'''

parser = argparse.ArgumentParser(description="Profile the Lippmann-Schwinger solver")
parser.add_argument(
    "--mode",
    choices=["single", "multi-gpu", "multi-node"],
    default="single",
    help="single: one device; multi-gpu: all local devices of one process; "
    "multi-node: all devices of all processes (calls jax.distributed.initialize())",
)
parser.add_argument(
    "--mesh",
    type=int,
    nargs="+",
    default=None,
    help="mesh shape: one entry (slab) or two entries (pencil); "
    "default: slab over all devices",
)
parser.add_argument("--grid", type=int, nargs=3, default=[64, 64, 48])
parser.add_argument("--repeat", type=int, default=10)
parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
args = parser.parse_args()

if args.mode == "multi-node":
    # Cluster details are auto-detected under SLURM/Open MPI; otherwise set
    # JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES and JAX_PROCESS_ID.
    if "JAX_COORDINATOR_ADDRESS" in os.environ:
        jax.distributed.initialize(
            coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
            num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
            process_id=int(os.environ["JAX_PROCESS_ID"]),
        )
    else:
        jax.distributed.initialize()

is_root = jax.process_index() == 0
if not is_root:
    sys.stdout = open(os.devnull, "w")

mesh = None
if args.mode == "single":
    spec_axes = None
else:
    devices = np.array(jax.devices())
    mesh_shape = tuple(args.mesh) if args.mesh else (len(devices),)
    if len(mesh_shape) not in (1, 2) or np.prod(mesh_shape) != len(devices):
        raise ValueError(
            f"mesh shape {mesh_shape} must have 1 or 2 entries and "
            f"multiply to the number of devices ({len(devices)})"
        )
    spec_axes = ("a", "b")[: len(mesh_shape)]
    mesh = Mesh(devices.reshape(mesh_shape), spec_axes)
print(
    f"mode = {args.mode}, processes = {jax.process_count()}, "
    f"devices = {jax.device_count()} ({jax.devices()[0].platform}), "
    f"mesh = {None if mesh is None else dict(mesh.shape)}"
)


def make_array(func, shape, dtype, leading=0):
    """Create a (possibly distributed) array from a function of the grid indices

    Each process only evaluates the shards of its own devices.

    :arg func: function mapping a tuple of three index slices to a numpy array
    :arg shape: global shape of the grid
    :arg dtype: data type
    :arg leading: number of unsharded leading axes of the array
    """
    if mesh is None:
        return jnp.array(func(tuple(slice(None) for _ in range(3))), dtype=dtype)
    sharding = NamedSharding(mesh, P(*(leading * [None]), *spec_axes))
    return jax.make_array_from_callback(
        shape,
        sharding,
        lambda idx: np.asarray(func(tuple(idx[leading:])), dtype=dtype),
    )


def compute_sigma_custom(epsilon, params):
    """User-defined isotropic linear elasticity, passed to lippmann_schwinger()

    :arg epsilon: strain
    :arg params: dictionary with Lame parameters {"lambda": lambda, "mu": mu}
    """
    tr_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    return 2 * params["mu"] * epsilon + params["lambda"] * jnp.stack(
        3 * [tr_epsilon] + 3 * [jnp.zeros_like(tr_epsilon)]
    )


def initialise_material(grid_spec, fibre_radius=0.2, dtype=jnp.float64):
    """Material coefficients lambda and mu evaluated at voxel centres

    Returns two arrays of shape (nx,ny,nz), sharded over the mesh (if any)

    :arg grid_spec: grid specification
    :arg fibre_radius: radius of fibre
    :arg dtype: data type
    """
    n = (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    L = (grid_spec.Lx, grid_spec.Ly, grid_spec.Lz)

    def coefficient(idx):
        X, Y, Z = np.meshgrid(
            *[L[d] / n[d] * (1 / 2 + np.arange(n[d])[idx[d]]) for d in range(3)],
            indexing="ij",
        )
        return 1 + 0.5 * (
            (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < fibre_radius**2
        )

    return (
        make_array(coefficient, n, dtype),
        make_array(coefficient, n, dtype),
    )


# Domain size in all three spatial direction
Lx = 1.2
Ly = 1.1
Lz = 0.7
# Number of grid cells in all three spatial directions
nx, ny, nz = args.grid

repeat = args.repeat
grid_spec = get_grid_spec(Lx, Ly, Lz, nx=nx, ny=ny, nz=nz)

# Distributed FFTs are selected at trace time; keep the context open for the whole run
_stack = contextlib.ExitStack()
_stack.enter_context(distributed_fft(mesh, spec_axes))

for dtype in [jnp.dtype(d).type for d in args.dtypes]:
    precision = "single precision" if dtype == jnp.float32 else "double precision"
    mu, lmbda = initialise_material(grid_spec, dtype=dtype)
    mu_ref = float(1 / 2 * (jnp.min(mu) + jnp.max(mu)))
    lmbda_ref = float(1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda)))
    zeros = jnp.zeros_like(mu)
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
            jax.block_until_ready(g)

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
            jax.block_until_ready(g)

        run(body)
    print()

    ref_params = {"mu": mu_ref, "lambda": lmbda_ref}
    with measure_time(
        f"evaluation (custom, {precision})", repeat=repeat, warmup=True
    ) as (run, it):

        def body():
            epsilon_custom, sigma = lippmann_schwinger(
                compute_sigma_custom,
                {"mu": mu, "lambda": lmbda},
                epsilon_bar,
                ref_params,
                grid_spec,
                verbose=it() == 0,
            )
            epsilon_custom.block_until_ready()
            return epsilon_custom

        epsilon_custom = run(body)
    print(
        "difference = ",
        jnp.linalg.norm(epsilon_custom - epsilon_isotropic)
        / jnp.linalg.norm(epsilon_isotropic),
    )
    print()

    with measure_time(
        f"gradient (custom, {precision})", repeat=repeat, warmup=True
    ) as (run, it):

        def body():
            def loss_fn(mu, lmbda, epsilon_bar):
                epsilon, sigma = lippmann_schwinger(
                    compute_sigma_custom,
                    {"mu": mu, "lambda": lmbda},
                    epsilon_bar,
                    ref_params,
                    grid_spec,
                    verbose=it() == 0,
                )
                return jnp.sum(sigma**2)

            g = jax.grad(loss_fn, argnums=(0, 1, 2))(mu, lmbda, epsilon_bar)
            jax.block_until_ready(g)

        run(body)
    print()

_stack.close()

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
