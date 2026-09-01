import io
import re
from contextlib import contextmanager, redirect_stdout

import jax
import numpy as np
import pytest
from fixtures import (
    grid_spec,
    initialise_isotropic_material,
    perturbed_parameters,
    reference_parameters,
    rng,
)
from jax import numpy as jnp

from jaxmaterials.solver._backend import _lippmann_schwinger_jax
from jaxmaterials.solver.divergence import relative_divergence
from jaxmaterials.solver.hooke import compute_sigma_anisotropic, compute_sigma_isotropic
from jaxmaterials.solver.lippmann_schwinger import (
    CUDAUnavailableError,
    lippmann_schwinger_anisotropic,
    lippmann_schwinger_isotropic,
)

jax.config.update("jax_enable_x64", True)


class IterationCounter:
    """Class for counting iterations with the :func:`capture_niter` context manager

    Contains an instance of :class:`io.StringIO` which is written to
    by the context manager. This buffer can be accessed through :attr:`IterationCounter.niter`
    to extract the number of iterations.
    """

    def __init__(self):
        """Initialise new instance"""
        self.buffer = io.StringIO()

    @property
    def niter(self):
        """Extract number of iterations

        Parses the content of the :attr:`IterationCounter.buffer` attribute to extract the number
        of iterations by looking for patterns of the form ``converged after X of Y iterations``.

        Raises an exception if the pattern is contained multiple times
        """
        output = self.buffer.getvalue()
        pattern = "converged after *([0-9]+) *of *[0-9]+ *iterations"
        n_found = len(re.findall(pattern, output))
        if n_found == 0:
            raise RuntimeError(
                f"Unable to extract number of iterations from {output}: no matching pattern"
            )
        elif n_found == 1:
            m = re.search(pattern, output)
            return int(m.group(1))
        else:
            raise RuntimeError(
                f"Unable to extract number of iterations from {output}: multiple matching patterns"
            )


@contextmanager
def capture_niter():
    """Context manager for extracting the number of iterations from a output printed by a block of code

    When used like this

    .. code::python
    with capture_niter() as ctx:
        solve()
    its = ctx.niter

    the output (i.e. everything written to Python's ``stdout``) is searched for a statement of the form
    ``converged after X of Y iterations``. The number of iterations `X` can be extracted from the returned
    :class:`IterationCounter` object.

    """
    it = IterationCounter()
    with redirect_stdout(it.buffer) as f:
        yield it
    print(it.buffer.getvalue())


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
    tol = 1.0e-5 if dtype == np.float32 else 1.0e-12
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    params = initialise_isotropic_material(grid_spec, rng, dtype)
    ref_params = reference_parameters(params)
    delta_epsilon_initial = jnp.zeros((6, *grid_spec.extents), dtype=dtype)
    with capture_niter() as ctx:
        epsilon_isotropic, sigma_isotropic = _lippmann_schwinger_jax(
            compute_sigma_isotropic,
            params,
            epsilon_bar,
            delta_epsilon_initial,
            ref_params,
            grid_spec,
            tol=tol,
            depth=0,
            maxits=32,
            dynamic_stopping=True,
            verbose=1,
        )
        epsilon_isotropic.block_until_ready()
    its_isotropic = ctx.niter
    lmbda = params["lambda"]
    mu = params["mu"]
    zero = np.zeros_like(mu)
    params_anisotropic = {
        "stiffness_tensor": np.stack(
            3 * [2 * mu + lmbda] + 3 * [mu] + 3 * [lmbda] + 12 * [zero]
        )
    }
    with capture_niter() as ctx:
        epsilon_anisotropic, sigma_anisotropic = _lippmann_schwinger_jax(
            compute_sigma_anisotropic,
            params_anisotropic,
            epsilon_bar,
            delta_epsilon_initial,
            ref_params,
            grid_spec,
            tol=tol,
            depth=0,
            maxits=32,
            dynamic_stopping=True,
            verbose=1,
        )
        epsilon_anisotropic.block_until_ready()
        its_anisotropic = ctx.niter
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
def test_convergence(grid_spec, rng, dtype, depth):
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
    with capture_niter() as ctx:
        delta_epsilon_initial = jnp.zeros((6, *grid_spec.extents), dtype=dtype)
        _, sigma = _lippmann_schwinger_jax(
            compute_sigma_isotropic,
            params,
            epsilon_bar,
            delta_epsilon_initial,
            ref_params,
            grid_spec,
            tol=tol,
            depth=depth,
            maxits=32,
            dynamic_stopping=True,
            verbose=1,
        )
        sigma.block_until_ready()
    its = ctx.niter
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


@pytest.mark.parametrize("use_cuda", [False, True])
@pytest.mark.parametrize("depth", [0, 2, 4])
def test_epsilon_initialisation(grid_spec, rng, depth, use_cuda):
    """Verify that initialising epsilon improves

    Restart solve from value obtained by previous solve to loose tolerance,
    The subsequent solve should require fewer iterations that the initial
    solve from a cold start.

    :arg grid_spec: specification of computational grid
    :arg rng: random number generator
    :arg depth: depth of Anderson acceleration
    :arg use_cuda: use CUDA impementation?
    """
    if use_cuda:
        if depth > 0:
            pytest.skip("CUDA implementation does not support Anderson acceleration")
        dtype = np.float32
        tol_warmup = 1.0e-2
        tol_target = 1.0e-5
    else:
        dtype = np.float64
        tol_warmup = 1.0e-3
        tol_target = 1.0e-12
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    params = initialise_isotropic_material(grid_spec, rng, dtype)

    def _solve(tol, delta_epsilon_initial):
        try:
            with capture_niter() as ctx:
                epsilon, _ = lippmann_schwinger_isotropic(
                    params,
                    epsilon_bar,
                    grid_spec,
                    delta_epsilon_initial=delta_epsilon_initial,
                    tol=tol,
                    maxits=32,
                    depth=depth,
                    use_cuda=use_cuda,
                    verbose=1,
                )
        except CUDAUnavailableError:
            pytest.skip(reason="CUDA code not available")
        epsilon.block_until_ready()
        return epsilon, ctx.niter

    delta_epsilon_initial = None
    # Solve to tight tolerance, starting from zero initial guess
    _, niter_cold = _solve(tol_target, delta_epsilon_initial)
    # Now solve to a loose tolerance
    epsilon, niter_loose = _solve(tol_warmup, delta_epsilon_initial)
    # use resulting strain as a starting point
    delta_epsilon_initial = epsilon - np.expand_dims(epsilon_bar, (1, 2, 3))
    _, niter_warm = _solve(tol_target, delta_epsilon_initial)
    assert niter_cold >= niter_loose + niter_warm


@pytest.mark.parametrize("initialise_epsilon", [False, True])
def test_jax_matches_cuda_isotropic(grid_spec, rng, initialise_epsilon):
    """Verify that CUDA and Jax solvers give identical results for isotropic materials
    (skipped if no GPU is available)

    :arg grid_spec: grid specification
    :arg rng: random number generator
    """
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=np.float32)
    tol = 1.0e-5
    maxits = 32
    params = initialise_isotropic_material(grid_spec, rng, dtype=np.float32)
    if initialise_epsilon:
        delta_epsilon_initial = np.astype(
            rng.normal(size=(6, *grid_spec.extents)), np.float32
        )
        delta_epsilon_initial -= np.average(
            delta_epsilon_initial, axis=(1, 2, 3), keepdims=True
        )
    else:
        delta_epsilon_initial = None
    try:
        epsilon_cuda, sigma_cuda = lippmann_schwinger_isotropic(
            params,
            epsilon_bar,
            grid_spec,
            delta_epsilon_initial=delta_epsilon_initial,
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
        delta_epsilon_initial=delta_epsilon_initial,
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
