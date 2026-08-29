import jax
import numpy as np

from jaxmaterials.solver.divergence import (
    relative_divergence,
    relative_divergence_fourier,
)
from jaxmaterials.solver.fourier import get_xi

jax.config.update("jax_enable_x64", True)

from fixtures import grid_spec, rng


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
