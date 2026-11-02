import numpy as np
from jax import numpy as jnp
import jax
from jax.experimental.multihost_utils import process_allgather
from jaxmaterials.common import get_grid_spec
from jaxmaterials.solver.fourier import distributed_fft3
import pytest

jax.config.update("jax_enable_x64", True)


def test_dfft_sequential():
    """Test the FFT in sequential mode"""
    grid_spec = get_grid_spec(nx=24, ny=12, nz=6)
    rng = np.random.default_rng(seed=52523)
    local_data = rng.normal(size=grid_spec.extents)
    data = jnp.asarray(local_data)

    data_k = distributed_fft3(data)

    reference_data_k = jnp.fft.fftn(data, axes=(2, 1, 0))
    tolerance = 1.0e-12
    assert (
        jnp.linalg.norm(data_k - reference_data_k) / jnp.linalg.norm(reference_data_k)
        < tolerance
    )

    jax.distributed.shutdown()


def get_parallel_config(parallel_decomposition):
    """Return the specification of the grid, mesh and sharding"""
    if parallel_decomposition == "pencil":
        grid_spec = get_grid_spec(nx=24, ny=12, nz=6, px=2, py=3)
        mesh = jax.make_mesh((2, 3), axis_names=("x", "y"))
        sharding = jax.sharding.NamedSharding(mesh, jax.P("x", "y", None))
    elif parallel_decomposition == "slab":
        grid_spec = get_grid_spec(nx=24, ny=12, nz=6, pz=6)
        mesh = jax.make_mesh((6,), axis_names=("z",))
        sharding = jax.sharding.NamedSharding(mesh, jax.P(None, None, "z"))
    else:
        raise RuntimeError(f"Unknown parallel decomposition: {parallel_decomposition}")

    return grid_spec, mesh, sharding


@pytest.mark.mpi(ranks=6)
@pytest.mark.parametrize("parallel_decomposition", ["slab", "pencil"])
def test_dfft_distributed(parallel_decomposition):
    """Test the FFT with slab- and pencil decomposition"""
    jax.distributed.initialize()

    grid_spec, mesh, sharding = get_parallel_config(parallel_decomposition)
    jax.set_mesh(mesh)

    rng = np.random.default_rng(seed=52523 + jax.process_index())
    local_data = rng.normal(size=grid_spec.local_extents)
    distributed_data = jax.make_array_from_process_local_data(
        sharding, local_data, global_shape=grid_spec.extents
    )

    distributed_data_k = distributed_fft3(distributed_data)
    data_k = process_allgather(distributed_data_k, tiled=True)
    data = process_allgather(distributed_data, tiled=True)

    reference_data_k = jnp.fft.fftn(data, axes=(2, 1, 0))
    tolerance = 1.0e-12
    assert (
        jnp.linalg.norm(data_k - reference_data_k) / jnp.linalg.norm(reference_data_k)
        < tolerance
    )

    jax.distributed.shutdown()
