import numpy as np
from jax import numpy as jnp
import jax
from jax.experimental.multihost_utils import process_allgather
from jaxmaterials.common import get_grid_spec
from jaxmaterials.solver.fourier import sharded_fft3
import pytest
from fixtures import get_parallel_gridspec

jax.config.update("jax_enable_x64", True)


def test_sequential():
    """Check that for an unsharded array the FFT is the same as the jax native FFT"""
    grid_spec = get_grid_spec(nx=24, ny=12, nz=6)
    rng = np.random.default_rng(seed=52523)
    local_data = rng.normal(size=grid_spec.extents)
    data = jnp.asarray(local_data)

    data_k = sharded_fft3(data, direction="forward")

    reference_data_k = jnp.fft.fftn(data, axes=(2, 1, 0))
    tolerance = 1.0e-12
    assert (
        jnp.linalg.norm(data_k - reference_data_k) / jnp.linalg.norm(reference_data_k)
        < tolerance
    )


def test_inverse_sequential():
    """Check that applying the FFT followed by its inverse returns the original array"""
    grid_spec = get_grid_spec(nx=24, ny=12, nz=6)
    rng = np.random.default_rng(seed=52523)
    local_data = rng.normal(size=grid_spec.extents)
    data = jnp.asarray(local_data)

    data_k = sharded_fft3(data, direction="forward")
    data_transformed = jnp.real(sharded_fft3(data_k, direction="backward"))

    tolerance = 1.0e-12
    assert jnp.linalg.norm(data - data_transformed) / jnp.linalg.norm(data) < tolerance


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("parallel_decomposition", ["slab", "pencil"])
def test_distributed(parallel_decomposition):
    """Test the FFT with slab- and pencil decomposition"""
    jax.distributed.initialize(cluster_detection_method="mpi4py")

    grid_spec, sharding = get_parallel_gridspec(parallel_decomposition)
    jax.set_mesh(sharding.mesh)

    rng = np.random.default_rng(seed=52523 + jax.process_index())
    local_data = rng.normal(size=grid_spec.local_extents)
    distributed_data = jax.make_array_from_process_local_data(
        sharding, local_data, global_shape=grid_spec.extents
    )

    distributed_data_k = sharded_fft3(distributed_data, direction="forward")
    data_k = process_allgather(distributed_data_k, tiled=True)
    data = process_allgather(distributed_data, tiled=True)

    reference_data_k = jnp.fft.fftn(data, axes=(2, 1, 0))
    tolerance = 1.0e-12
    assert (
        jnp.linalg.norm(data_k - reference_data_k) / jnp.linalg.norm(reference_data_k)
        < tolerance
    )
    jax.distributed.shutdown()


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("parallel_decomposition", ["slab", "pencil"])
def test_inverse_distributed(parallel_decomposition):
    """Test the FFT with slab- and pencil decomposition"""
    jax.distributed.initialize(cluster_detection_method="mpi4py")
    grid_spec, sharding = get_parallel_gridspec(parallel_decomposition)
    jax.set_mesh(sharding.mesh)

    rng = np.random.default_rng(seed=52523 + jax.process_index())
    local_data = rng.normal(size=grid_spec.local_extents)
    distributed_data = jax.make_array_from_process_local_data(
        sharding, local_data, global_shape=grid_spec.extents
    )

    distributed_data_k = sharded_fft3(distributed_data, direction="forward")
    distributed_data_transformed = jnp.real(
        sharded_fft3(distributed_data_k, direction="backward")
    )

    tolerance = 1.0e-12
    assert (
        jnp.linalg.norm(distributed_data - distributed_data_transformed)
        / jnp.linalg.norm(distributed_data)
        < tolerance
    )
    jax.distributed.shutdown()
