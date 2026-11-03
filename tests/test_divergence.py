import jax
import numpy as np
import pytest

from jax.experimental.multihost_utils import process_allgather

from jaxmaterials.solver.divergence import (
    relative_divergence,
    relative_divergence_fourier,
)
from jaxmaterials.solver.fourier import get_xi, sharded_fft3

jax.config.update("jax_enable_x64", True)

from fixtures import grid_spec, rng, get_parallel_gridspec


def test_relative_divergence(grid_spec, rng):
    """Verify that the relative divergence used for exit criterion is computed
    consistently in real space and Fourier space

    :arg grid_spec: specification of computational grid
    :arg rng: random number generator
    """

    xi = get_xi(grid_spec)
    sigma = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz))
    sigma_hat = np.fft.fftn(sigma, axes=[-3, -2, -1])
    rel_div_real = relative_divergence(sigma, grid_spec)
    rel_div_fourier = relative_divergence_fourier(sigma_hat, xi)
    tolerance = 1.0e-5
    assert abs((rel_div_real - rel_div_fourier) / rel_div_real) < tolerance


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("parallel_decomposition", ["slab", "pencil"])
def test_relative_divergence_sharded(parallel_decomposition, rng):
    """Verify that the relative divergence used for exit criterion is computed
    consistently in real space and Fourier space for distributed arrays

    :arg parallel_decomposition: parallel decomposition, must by "slab" or "pencil"
    :arg rng: random number generator
    """
    jax.distributed.initialize()
    grid_spec, sharding = get_parallel_gridspec(parallel_decomposition)
    jax.set_mesh(sharding.mesh)
    sigma_sharding = sharding.update(spec=jax.P(None, *sharding.spec))
    xi = get_xi(grid_spec)
    local_sigma = rng.normal(size=(6, *grid_spec.local_extents))
    sigma = jax.make_array_from_process_local_data(
        sigma_sharding, local_sigma, global_shape=(6, *grid_spec.extents)
    )
    sigma_hat = sharded_fft3(sigma, direction="forward")
    rel_div_real = relative_divergence(process_allgather(sigma, tiled=True), grid_spec)
    rel_div_fourier = relative_divergence_fourier(sigma_hat, xi)
    tolerance = 1.0e-5
    assert abs((rel_div_real - rel_div_fourier) / rel_div_real) < tolerance
    jax.distributed.shutdown()
