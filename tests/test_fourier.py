import pytest
import numpy as np
import pytest
import jax
from jax import numpy as jnp

from jaxmaterials.solver.fourier import (
    get_xizero,
    get_inverse_anisotropic_acoustic_tensor,
    get_anisotropic_acoustic_tensor,
    fourier_solve_isotropic,
    fourier_solve_anisotropic,
)

from fixtures import grid_spec, rng

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_acoustic_tensor(grid_spec, dtype):
    """Verify that acoustic tensor K computed for anisotropic material agrees with analytical
    expression for isotropic material.

    :arg grid_spec: specification of computational grid
    :arg dtype: data type (single or double precision)
    """
    xizero = get_xizero(grid_spec, dtype)
    mu0, lmbda0 = 0.9, 0.4
    stiffness_tensor0 = np.stack(
        3 * [2 * mu0 + lmbda0] + 3 * [mu0] + 3 * [lmbda0] + 12 * [0]
    )
    K0 = get_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)
    K0_reference = np.stack(
        [
            [
                mu0 * (xizero[1] ** 2 + xizero[2] ** 2)
                + (lmbda0 + 2 * mu0) * xizero[0] ** 2,
                (lmbda0 + mu0) * xizero[0] * xizero[1],
                (lmbda0 + mu0) * xizero[0] * xizero[2],
            ],
            [
                (lmbda0 + mu0) * xizero[0] * xizero[1],
                mu0 * (xizero[0] ** 2 + xizero[2] ** 2)
                + (lmbda0 + 2 * mu0) * xizero[1] ** 2,
                (lmbda0 + mu0) * xizero[1] * xizero[2],
            ],
            [
                (lmbda0 + mu0) * xizero[0] * xizero[2],
                (lmbda0 + mu0) * xizero[1] * xizero[2],
                mu0 * (xizero[0] ** 2 + xizero[1] ** 2)
                + (lmbda0 + 2 * mu0) * xizero[2] ** 2,
            ],
        ]
    )
    rtol = 1.0e-7 if dtype == np.float32 else 1.0e-12
    assert np.linalg.norm(K0 - K0_reference) / np.linalg.norm(K0_reference) < rtol


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inverse_acoustic_tensor(grid_spec, dtype):
    """Verify that inverse N of acoustic tensor K agrees with analytical expression when
    computed for an isotropic material

    :arg grid_spec: specification of computational grid
    :arg dtype: data type (single or double precision)
    """
    xizero = get_xizero(grid_spec, dtype)
    mu0, lmbda0 = 0.9, 0.4
    stiffness_tensor0 = np.stack(
        3 * [2 * mu0 + lmbda0] + 3 * [mu0] + 3 * [lmbda0] + 12 * [0]
    )
    N0 = get_inverse_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)
    rho = (lmbda0 + mu0) / (lmbda0 + 2 * mu0)
    mask = (xizero[0] ** 2 + xizero[1] ** 2 + xizero[2] ** 2) > 1.0e-8
    N0_reference = (
        mask
        / mu0
        * np.stack(
            [
                [
                    1 - rho * xizero[0] ** 2,
                    -rho * xizero[0] * xizero[1],
                    -rho * xizero[0] * xizero[2],
                ],
                [
                    -rho * xizero[1] * xizero[0],
                    1 - rho * xizero[1] ** 2,
                    -rho * xizero[1] * xizero[2],
                ],
                [
                    -rho * xizero[2] * xizero[0],
                    -rho * xizero[2] * xizero[1],
                    1 - rho * xizero[2] ** 2,
                ],
            ]
        )
    )
    rtol = 1.0e-7 if dtype == np.float32 else 1.0e-12
    assert np.linalg.norm(N0 - N0_reference) / np.linalg.norm(N0_reference) < rtol


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_solve(grid_spec, rng, dtype):
    """Verify that Fourier solve gives same results as analytical expression when applied to
    an isotropic material.

    :arg grid_spec: specification of computational grid
    :arg rng: random number generator
    :arg dtype: data type (single or double precision)
    """
    sigma_hat = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(
        dtype
    )
    xizero = get_xizero(grid_spec, dtype)
    mu0, lmbda0 = 0.9, 0.4
    stiffness_tensor0 = np.stack(
        3 * [2 * mu0 + lmbda0] + 3 * [mu0] + 3 * [lmbda0] + 12 * [0]
    )
    N_ref = get_inverse_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)
    r_hat_isotropic = fourier_solve_isotropic(
        sigma_hat, xizero, {"lambda": lmbda0, "mu": mu0}
    )
    r_hat_anisotropic = fourier_solve_anisotropic(sigma_hat, N_ref, xizero)
    rtol = 1.0e-6 if dtype == np.float32 else 1.0e-12
    assert (
        np.linalg.norm(r_hat_isotropic - r_hat_anisotropic)
        / np.linalg.norm(r_hat_isotropic)
        < rtol
    )
