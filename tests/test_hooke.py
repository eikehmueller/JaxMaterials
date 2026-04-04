import pytest
import numpy as np
import pytest
import jax

from fixtures import initialise_material, grid_spec, rng
from jaxmaterials.solver.hooke import compute_sigma_isotropic, compute_sigma_anisotropic


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hooke(grid_spec, dtype, rng):
    """Verify that Hooke's law sigma_{ij} = C_{ijkl} epsilon_{kl} is consistent in
    isotropic and anisotropic case when applied to an isotropic material.

    :arg grid_spec: specification of computational grid
    :arg dtype: data type (single or double precision)
    :arg rng: random number generator
    """
    epsilon = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(
        dtype
    )
    mu, lmbda = initialise_material(grid_spec, rng, dtype)
    zeros = np.zeros(mu.shape, dtype=dtype)
    stiffness_tensor = np.stack(
        3 * [2 * mu + lmbda] + 3 * [mu] + 3 * [lmbda] + 12 * [zeros]
    )
    sigma_isotropic = compute_sigma_isotropic(lmbda, mu, epsilon)
    sigma_anisotropic = compute_sigma_anisotropic(stiffness_tensor, epsilon)
    rtol = 1.0e-7 if dtype == np.float32 else 1.0e-12
    assert (
        np.linalg.norm(sigma_isotropic - sigma_anisotropic)
        / np.linalg.norm(sigma_isotropic)
        < rtol
    )
