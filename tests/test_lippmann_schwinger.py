import pytest
import numpy as np
import pytest
import jax

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.fourier import (
    get_xi,
    get_xizero,
    get_inverse_anisotropic_acoustic_tensor,
    get_anisotropic_acoustic_tensor,
    fourier_solve_isotropic,
    fourier_solve_anisotropic,
)
from jaxmaterials.solver.hooke import compute_sigma_isotropic, compute_sigma_anisotropic
from jaxmaterials.solver.lippmann_schwinger import (
    relative_divergence,
    relative_divergence_fourier,
    lippmann_schwinger_isotropic_jax,
    lippmann_schwinger_isotropic_cuda,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=[[64, 48, 32], [57, 43, 37]], ids=["even", "odd"])
def grid_spec(request):
    """Return grid specification"""
    # Domain size in all three spatial direction
    Lx = 1.2
    Ly = 0.8
    Lz = 0.9
    # Number of grid cells in all three spatial directions
    nx, ny, nz = request.param

    return GridSpec(nx, ny, nz, Lx, Ly, Lz)


@pytest.fixture
def rng():
    """Random number generator"""
    return np.random.default_rng(seed=784173)


def initialise_material(grid_spec, rng, dtype):
    """Construct random Lame parameters

    :arg grid_spec: specification of grid
    :arg rng: random number generator
    :arg dtype: data type
    """
    shape = (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    mu = np.ones(shape=shape) + rng.uniform(size=shape, low=-0.2, high=+0.2)
    lmbda = np.ones(shape=shape) + rng.uniform(size=shape, low=-0.2, high=+0.2)
    return np.array(mu, dtype=dtype), np.array(lmbda, dtype=dtype)


def test_relative_convergence(grid_spec):
    """Verify that the relative divergence is computed consistently in real space
    and Fourier space"""

    xi = get_xi(grid_spec)
    rng = np.random.default_rng(seed=8741823)
    sigma = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz))
    sigma_hat = np.fft.fftn(sigma, axes=[-3, -2, -1])
    rel_div_real = relative_divergence(sigma, grid_spec)
    rel_div_fourier = relative_divergence_fourier(sigma_hat, xi)
    tolerance = 1.0e-5
    assert abs((rel_div_real - rel_div_fourier) / rel_div_real) < tolerance


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hooke(grid_spec, dtype):
    """Verify that Hooke's law is consistent in isotropic and anisotropic case"""
    rng = np.random.default_rng(seed=8741823)
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_acoustic_tensor(grid_spec, dtype):
    """Verify that acoustic tensor is computed consistently"""
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
    """Verify that acoustic tensor is computed consistently"""
    xizero = get_xizero(grid_spec, dtype)
    mu0, lmbda0 = 0.9, 0.4
    stiffness_tensor0 = np.stack(
        3 * [2 * mu0 + lmbda0] + 3 * [mu0] + 3 * [lmbda0] + 12 * [0]
    )
    N0 = get_inverse_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)
    rho = (lmbda0 + mu0) / (lmbda0 + 2 * mu0)
    xi_nrm = (xizero[0] ** 2 + xizero[1] ** 2 + xizero[2] ** 2) > 1.0e-8
    N0_reference = (
        xi_nrm
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
def test_fourier_solve(grid_spec, dtype):
    """Verify that Fourier solve is consistent in isotropic and anisotropic case"""
    rng = np.random.default_rng(seed=8741823)
    sigma_hat = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(
        dtype
    )
    xizero = get_xizero(grid_spec, dtype)
    mu0, lmbda0 = 0.9, 0.4
    stiffness_tensor0 = np.stack(
        3 * [2 * mu0 + lmbda0] + 3 * [mu0] + 3 * [lmbda0] + 12 * [0]
    )
    N_ref = get_inverse_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)
    r_hat_isotropic = fourier_solve_isotropic(sigma_hat, lmbda0, mu0, xizero)
    r_hat_anisotropic = fourier_solve_anisotropic(sigma_hat, N_ref, xizero)
    rtol = 1.0e-6 if dtype == np.float32 else 1.0e-12
    assert (
        np.linalg.norm(r_hat_isotropic - r_hat_anisotropic)
        / np.linalg.norm(r_hat_isotropic)
        < rtol
    )


@pytest.mark.parametrize("depth", [0, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_convergence(grid_spec, rng, dtype, depth):
    """Verify that Lippmann Schwinger solver converges in small number of iterations

    :arg grid_spec: specification of computational grid
    :arg rng: random number generator
    :arg dtype: data type (single or double precision)
    :arg depth: depth of Anderson acceleration
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5])
    mu, lmbda = initialise_material(grid_spec, rng, dtype)
    atol = 1.0e-5 if dtype == np.float32 else 1.0e-12
    rtol = 1.0e-20
    _, sigma, iter = lippmann_schwinger_isotropic_jax(
        mu,
        lmbda,
        epsilon_bar,
        grid_spec,
        rtol=rtol,
        atol=atol,
        depth=depth,
        maxiter=32,
        dtype=dtype,
    )
    rel_div = relative_divergence(sigma, grid_spec)
    if dtype == np.float32:
        assert iter < 8
    else:
        if depth == 0:
            assert iter < 16
        elif depth == 2:
            assert iter < 14
        else:
            assert iter < 13
    assert rel_div < atol


def test_jax_matches_cuda(grid_spec, rng):
    """Verify that CUDA and Jax solvers give identical results

    :arg grid_spec: grid specification
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=np.float32)
    mu, lmbda = initialise_material(grid_spec, rng, dtype=np.float32)
    atol = 1e-5
    rtol = 1.0e-20
    try:
        epsilon_cuda, sigma_cuda, iter_cuda = lippmann_schwinger_isotropic_cuda(
            mu,
            lmbda,
            epsilon_bar,
            grid_spec,
            rtol=rtol,
            atol=atol,
            maxiter=32,
            verbose=0,
        )
    except:
        pytest.skip(reason="CUDA code not available")
    epsilon_jax, sigma_jax, iter_jax = lippmann_schwinger_isotropic_jax(
        mu,
        lmbda,
        epsilon_bar,
        grid_spec,
        rtol=rtol,
        atol=atol,
        depth=0,
        maxiter=32,
        dtype=np.float32,
    )
    rel_diff_epsilon_2 = np.sum((epsilon_cuda - epsilon_jax) ** 2) / np.sum(
        epsilon_jax**2
    )
    rel_diff_sigma_2 = np.sum((sigma_cuda - sigma_jax) ** 2) / np.sum(sigma_jax**2)
    assert rel_diff_epsilon_2 < 5e-3
    assert rel_diff_sigma_2 < 2e-3
