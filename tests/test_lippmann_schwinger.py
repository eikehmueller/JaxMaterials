import pytest
import numpy as np
import pytest
import jax
import re

from jaxmaterials.solver.hooke import compute_sigma_isotropic, compute_sigma_anisotropic

from jaxmaterials.solver.divergence import relative_divergence

from jaxmaterials.solver._backend import _lippmann_schwinger_jax
from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_isotropic,
    lippmann_schwinger_anisotropic,
    CUDAUnavailableError,
)
from fixtures import (
    initialise_isotropic_material,
    reference_parameters,
    perturbed_parameters,
    grid_spec,
    rng,
)

jax.config.update("jax_enable_x64", True)


def get_niter(capfd):
    """Extract number of iterations from output

    :arg capfd: output capture
    """
    captured = capfd.readouterr()
    print(captured.out)
    m = re.search("converged after *([0-9]+) *of *[0-9]+ *iterations", captured.out)
    if m:
        its = int(m.group(1))
    else:
        assert False
    return its


@pytest.mark.parametrize("depth", [0, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_anisotropic_solve(capfd, grid_spec, rng, depth, dtype):
    """Verify that isotropic and anisotropic solvers give the same result when applied
    to an isotropic material

    :arg grid_spec: specification of computational grid
    :arg depth: depth of Anderson acceleration
    :arg dtype: data type (single or double precision)
    :arg rng: random number generator
    """
    tol = 1.0e-5 if dtype == np.float32 else 1.0e-12
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    params = initialise_isotropic_material(grid_spec, rng, dtype)
    ref_params = reference_parameters(params)
    epsilon_isotropic, sigma_isotropic = _lippmann_schwinger_jax(
        compute_sigma_isotropic,
        params,
        epsilon_bar,
        ref_params,
        grid_spec,
        tol=tol,
        depth=0,
        maxits=32,
        dynamic_stopping=True,
        verbose=1,
    )
    its_isotropic = get_niter(capfd)
    lmbda = params["lambda"]
    mu = params["mu"]
    zero = np.zeros_like(mu)
    params_anisotropic = {
        "stiffness_tensor": np.stack(
            3 * [2 * mu + lmbda] + 3 * [mu] + 3 * [lmbda] + 12 * [zero]
        )
    }
    epsilon_anisotropic, sigma_anisotropic = _lippmann_schwinger_jax(
        compute_sigma_anisotropic,
        params_anisotropic,
        epsilon_bar,
        ref_params,
        grid_spec,
        tol=tol,
        depth=0,
        maxits=32,
        dynamic_stopping=True,
        verbose=1,
    )
    its_anisotropic = get_niter(capfd)
    assert (
        np.linalg.norm(epsilon_isotropic - epsilon_anisotropic)
        / np.linalg.norm(epsilon_isotropic)
        < tol
    )
    assert (
        np.linalg.norm(sigma_isotropic - sigma_anisotropic)
        / np.linalg.norm(sigma_isotropic)
        < tol
    )
    assert abs(its_isotropic - its_anisotropic) <= 1


@pytest.mark.parametrize("depth", [0, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_convergence(capfd, grid_spec, rng, dtype, depth):
    """Verify that isotropic Lippmann Schwinger solver converges in small
    number of iterations

    :arg grid_spec: specification of computational grid
    :arg rng: random number generator
    :arg dtype: data type (single or double precision)
    :arg depth: depth of Anderson acceleration
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    params = initialise_isotropic_material(grid_spec, rng, dtype)
    ref_params = reference_parameters(params)
    tol = 1.0e-5 if dtype == np.float32 else 1.0e-12
    _, sigma = _lippmann_schwinger_jax(
        compute_sigma_isotropic,
        params,
        epsilon_bar,
        ref_params,
        grid_spec,
        tol=tol,
        depth=depth,
        maxits=32,
        dynamic_stopping=True,
        verbose=1,
    )
    its = get_niter(capfd)
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
            assert its < 15
    assert rel_div < tol


def test_jax_matches_cuda_isotropic(grid_spec, rng):
    """Verify that CUDA and Jax solvers give identical results for isotropic materials
    (skipped if no GPU is available)

    :arg grid_spec: grid specification
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=np.float32)
    tol = 1.0e-5
    maxits = 32
    params = initialise_isotropic_material(grid_spec, rng, dtype=np.float32)
    try:
        epsilon_cuda, sigma_cuda = lippmann_schwinger_isotropic(
            params,
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
        params,
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


def test_jax_matches_cuda_anisotropic(grid_spec, rng):
    """Verify that CUDA and JAX anisotropic solvers match on anisotropic materials
    (skipped if no GPU is available).

    :arg grid_spec: grid specification
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=np.float32)
    params_isotropic = initialise_isotropic_material(grid_spec, rng, dtype=np.float32)
    tol = 1.0e-5
    maxits = 32
    params = perturbed_parameters(rng, params_isotropic, delta=0.1)
    ref_params = {
        field: 1
        / 2
        * (np.min(params_isotropic[field]) + np.max(params_isotropic[field]))
        for field in params_isotropic.keys()
    }
    try:
        epsilon_cuda, sigma_cuda = lippmann_schwinger_anisotropic(
            params,
            epsilon_bar,
            ref_params,
            grid_spec,
            tol=tol,
            maxits=maxits,
            use_cuda=True,
            verbose=1,
        )
    except CUDAUnavailableError:
        pytest.skip(reason="CUDA code not available")

    epsilon_jax, sigma_jax = lippmann_schwinger_anisotropic(
        params,
        epsilon_bar,
        ref_params,
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
