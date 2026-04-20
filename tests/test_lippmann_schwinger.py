import pytest
import numpy as np
import pytest
import jax


from jaxmaterials.solver.fourier import get_xi
from jaxmaterials.solver.backend import relative_divergence, relative_divergence_fourier

from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_isotropic_jax,
    lippmann_schwinger_anisotropic_jax,
    lippmann_schwinger_adjoint_isotropic_jax,
    lippmann_schwinger_adjoint_anisotropic_jax,
    lippmann_schwinger_isotropic_cuda,
    lippmann_schwinger_anisotropic_cuda,
)
from fixtures import initialise_material, perturbed_stiffness_tensor, grid_spec, rng

jax.config.update("jax_enable_x64", True)


def test_relative_divergence(grid_spec):
    """Verify that the relative divergence used for exit criterion is computed
    consistently in real space and Fourier space

    :arg grid_spec: specification of computational grid
    """

    xi = get_xi(grid_spec)
    rng = np.random.default_rng(seed=8741823)
    sigma = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz))
    sigma_hat = np.fft.fftn(sigma, axes=[-3, -2, -1])
    rel_div_real = relative_divergence(sigma, grid_spec)
    rel_div_fourier = relative_divergence_fourier(sigma_hat, xi)
    tolerance = 1.0e-5
    assert abs((rel_div_real - rel_div_fourier) / rel_div_real) < tolerance


@pytest.mark.parametrize("depth", [0, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_anisotropic_solve(grid_spec, rng, depth, dtype):
    """Verify that isotropic and anisotropic solvers give the same result when applied
    to an isotropic material

    :arg grid_spec: specification of computational grid
    :arg depth: depth of Anderson acceleration
    :arg dtype: data type (single or double precision)
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    mu, lmbda = initialise_material(grid_spec, rng, dtype)
    zero = np.zeros_like(mu)
    stiffness_tensor = np.stack(
        3 * [2 * mu + lmbda] + 3 * [mu] + 3 * [lmbda] + 12 * [zero]
    )

    atol = 1.0e-5 if dtype == np.float32 else 1.0e-12
    rtol = 1.0e-20
    epsilon_isotropic, sigma_isotropic, iter_isotropic = (
        lippmann_schwinger_isotropic_jax(
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
    )
    epsilon_anisotropic, sigma_anisotropic, iter_anisotropic = (
        lippmann_schwinger_anisotropic_jax(
            stiffness_tensor,
            epsilon_bar,
            grid_spec,
            rtol=rtol,
            atol=atol,
            depth=depth,
            maxiter=32,
            dtype=dtype,
        )
    )

    rtol = 1.0e-6 if dtype == np.float32 else 1.0e-12
    assert (
        np.linalg.norm(epsilon_isotropic - epsilon_anisotropic)
        / np.linalg.norm(epsilon_isotropic)
        < rtol
    )
    assert (
        np.linalg.norm(sigma_isotropic - sigma_anisotropic)
        / np.linalg.norm(sigma_isotropic)
        < rtol
    )
    assert abs(iter_isotropic - iter_anisotropic) <= 1


@pytest.mark.parametrize("depth", [0, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_convergence(grid_spec, rng, dtype, depth):
    """Verify that isotropic Lippmann Schwinger solver converges in small
    number of iterations

    :arg grid_spec: specification of computational grid
    :arg rng: random number generator
    :arg dtype: data type (single or double precision)
    :arg depth: depth of Anderson acceleration
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
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


def test_jax_matches_cuda_isotropic(grid_spec, rng):
    """Verify that CUDA and Jax solvers give identical results for isotropic materials
    (skipped if no GPU is available)

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
    assert abs(iter_jax - iter_cuda) <= 1


def test_jax_matches_cuda_anisotropic(grid_spec, rng):
    """Verify that CUDA and JAX anisotropic solvers match on anisotropic materials
    (skipped if no GPU is available).

    :arg grid_spec: grid specification
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=np.float32)
    mu, lmbda = initialise_material(grid_spec, rng, dtype=np.float32)

    stiffness_tensor = perturbed_stiffness_tensor(rng, mu, lmbda, delta=0.1)
    atol = 1e-5
    rtol = 1.0e-20
    try:
        epsilon_cuda, sigma_cuda, iter_cuda = lippmann_schwinger_anisotropic_cuda(
            stiffness_tensor,
            epsilon_bar,
            grid_spec,
            rtol=rtol,
            atol=atol,
            maxiter=32,
            verbose=0,
        )
    except Exception:
        pytest.skip(reason="CUDA code not available")

    epsilon_jax, sigma_jax, iter_jax = lippmann_schwinger_anisotropic_jax(
        stiffness_tensor,
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
    assert rel_diff_epsilon_2 < 2e-5
    assert rel_diff_sigma_2 < 2e-5
    assert abs(iter_jax - iter_cuda) <= 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adjoint_isotropic(grid_spec, rng, dtype):
    """Check that isotropic adjoint solver converges in a small number of iterations"""
    f_rhs = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    mu, lmbda = initialise_material(grid_spec, rng, dtype)
    rtol = 1.0e-5 if dtype == np.float32 else 1.0e-12
    atol = 1.0e-20
    _, iter = lippmann_schwinger_adjoint_isotropic_jax(
        mu, lmbda, f_rhs, grid_spec, rtol=rtol, atol=atol, maxiter=32, dtype=dtype
    )
    maxiter = 10 if dtype == np.float32 else 20
    assert iter < maxiter


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adjoint_anisotropic(grid_spec, rng, dtype):
    """Check that anisotropic adjoint solver converges in a small number of iterations"""
    f_rhs = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    mu, lmbda = initialise_material(grid_spec, rng, dtype)
    stiffness_tensor = perturbed_stiffness_tensor(rng, mu, lmbda, delta=0.1)
    rtol = 1.0e-5 if dtype == np.float32 else 1.0e-12
    atol = 1.0e-20
    _, iter = lippmann_schwinger_adjoint_anisotropic_jax(
        stiffness_tensor,
        f_rhs,
        grid_spec,
        rtol=rtol,
        atol=atol,
        maxiter=32,
        dtype=dtype,
    )
    maxiter = 10 if dtype == np.float32 else 20
    assert iter < maxiter
