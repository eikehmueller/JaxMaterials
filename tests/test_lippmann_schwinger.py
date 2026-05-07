import pytest
import numpy as np
import pytest
import jax
from jax import numpy as jnp

from jaxmaterials.solver.hooke import compute_sigma_isotropic
from jaxmaterials.solver.fourier import get_xi
from jaxmaterials.solver.backend import (
    relative_divergence,
    relative_divergence_fourier,
    _lippmann_schwinger_jax,
    _lippmann_schwinger_generic_jax,
)

from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_isotropic,
    lippmann_schwinger_anisotropic,
    CUDAUnavailableError,
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
    epsilon_isotropic, sigma_isotropic, its_isotropic = _lippmann_schwinger_jax(
        {"mu": mu, "lambda": lmbda},
        epsilon_bar,
        grid_spec,
        isotropic=True,
        rtol=1e-20,
        atol=1.0e-5 if dtype == np.float32 else 1.0e-12,
        depth=0,
        maxits=32,
        dynamic_stopping=True,
        dtype=dtype,
        verbose=1,
    )
    epsilon_anisotropic, sigma_anisotropic, its_anisotropic = _lippmann_schwinger_jax(
        {"stiffness_tensor": stiffness_tensor},
        epsilon_bar,
        grid_spec,
        isotropic=False,
        rtol=1e-20,
        atol=1.0e-5 if dtype == np.float32 else 1.0e-12,
        depth=0,
        maxits=32,
        dynamic_stopping=True,
        dtype=dtype,
        verbose=1,
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
    assert abs(its_isotropic - its_anisotropic) <= 1


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
    _, sigma, its = _lippmann_schwinger_jax(
        {"mu": mu, "lambda": lmbda},
        epsilon_bar,
        grid_spec,
        isotropic=True,
        rtol=1.0e-20,
        atol=atol,
        depth=depth,
        maxits=32,
        dynamic_stopping=True,
        dtype=dtype,
        verbose=1,
    )
    rel_div = relative_divergence(sigma, grid_spec)
    if dtype == np.float32:
        if depth == 0:
            assert its < 8
        else:
            assert its < 7
    else:
        if depth == 0:
            assert its < 16
        else:
            assert its < 14
    assert rel_div < atol


def test_jax_matches_cuda_isotropic(grid_spec, rng):
    """Verify that CUDA and Jax solvers give identical results for isotropic materials
    (skipped if no GPU is available)

    :arg grid_spec: grid specification
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=np.float32)
    tol = 1.0e-5
    maxits = 32
    mu, lmbda = initialise_material(grid_spec, rng, dtype=np.float32)
    try:
        epsilon_cuda, sigma_cuda = lippmann_schwinger_isotropic(
            mu,
            lmbda,
            epsilon_bar,
            grid_spec,
            tol=tol,
            maxits=maxits,
            use_cuda=True,
            verbose=1,
        )
    except CUDAUnavailableError:
        pytest.skip(reason="CUDA code not available")
    epsilon_jax, sigma_jax = lippmann_schwinger_isotropic(
        mu,
        lmbda,
        epsilon_bar,
        grid_spec,
        tol=tol,
        maxits=maxits,
        use_cuda=False,
        verbose=1,
    )
    rel_diff_epsilon_2 = np.sum((epsilon_cuda - epsilon_jax) ** 2) / np.sum(
        epsilon_jax**2
    )
    rel_diff_sigma_2 = np.sum((sigma_cuda - sigma_jax) ** 2) / np.sum(sigma_jax**2)
    assert rel_diff_epsilon_2 < 1.0e-2
    assert rel_diff_sigma_2 < 1.0e-2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_matches_generic_isotropic(grid_spec, rng, dtype):
    """Verify that JAX solver and generic JAX solver give same results

    :arg grid_spec: grid specification
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    tol = 1.0e-12 if dtype == np.float64 else 1.0e-5
    maxits = 32
    mu, lmbda = initialise_material(grid_spec, rng, dtype=dtype)
    mu0 = 1 / 2 * (jnp.min(mu) + jnp.max(mu))
    lmbda0 = 1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda))
    epsilon, sigma = lippmann_schwinger_isotropic(
        mu,
        lmbda,
        epsilon_bar,
        grid_spec,
        tol=tol,
        maxits=maxits,
        use_cuda=False,
        verbose=1,
    )
    epsilon_generic, sigma_generic, _ = _lippmann_schwinger_generic_jax(
        compute_sigma_isotropic,
        epsilon_bar,
        {"lambda": lmbda, "mu": mu},
        {"lambda": lmbda0, "mu": mu0},
        grid_spec,
        atol=1e-20,
        rtol=tol,
        depth=0,
        maxits=128,
        dynamic_stopping=True,
        dtype=dtype,
        verbose=1,
    )
    rel_diff_epsilon_2 = np.sum((epsilon - epsilon_generic) ** 2) / np.sum(
        epsilon_generic**2
    )
    rel_diff_sigma_2 = np.sum((sigma - sigma_generic) ** 2) / np.sum(sigma_generic**2)
    assert rel_diff_epsilon_2 < tol
    assert rel_diff_sigma_2 < tol


def test_jax_matches_cuda_anisotropic(grid_spec, rng):
    """Verify that CUDA and JAX anisotropic solvers match on anisotropic materials
    (skipped if no GPU is available).

    :arg grid_spec: grid specification
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=np.float32)
    mu, lmbda = initialise_material(grid_spec, rng, dtype=np.float32)
    tol = 1.0e-5
    maxits = 32
    stiffness_tensor = perturbed_stiffness_tensor(rng, mu, lmbda, delta=0.1)
    try:
        epsilon_cuda, sigma_cuda = lippmann_schwinger_anisotropic(
            stiffness_tensor,
            epsilon_bar,
            grid_spec,
            tol=tol,
            maxits=maxits,
            use_cuda=True,
            verbose=1,
        )
    except CUDAUnavailableError:
        pytest.skip(reason="CUDA code not available")

    epsilon_jax, sigma_jax = lippmann_schwinger_anisotropic(
        stiffness_tensor,
        epsilon_bar,
        grid_spec,
        tol=tol,
        maxits=maxits,
        use_cuda=False,
        verbose=1,
    )

    rel_diff_epsilon_2 = np.sum((epsilon_cuda - epsilon_jax) ** 2) / np.sum(
        epsilon_jax**2
    )
    rel_diff_sigma_2 = np.sum((sigma_cuda - sigma_jax) ** 2) / np.sum(sigma_jax**2)
    assert rel_diff_epsilon_2 < 2e-5
    assert rel_diff_sigma_2 < 2e-5
