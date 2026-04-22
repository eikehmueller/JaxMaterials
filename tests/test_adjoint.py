import pytest
import numpy as np
import pytest
import jax
from jax import numpy as jnp
from jax.test_util import check_vjp
import functools


from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_adjoint_isotropic_jax,
    lippmann_schwinger_adjoint_anisotropic_jax,
)
from jaxmaterials.solver.backend import solve_isotropic, solve_anisotropic
from fixtures import (
    initialise_material,
    perturbed_stiffness_tensor,
    grid_spec,
    grid_spec_small,
    rng,
)

jax.config.update("jax_enable_x64", True)


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vjp_isotropic(grid_spec_small, rng, dtype):
    mu, lmbda = initialise_material(grid_spec_small, rng, dtype)
    epsilon_mean = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)

    def loss_fn(mu, lmbda, epsilon_mean):
        epsilon, sigma = solve_isotropic(mu, lmbda, epsilon_mean, grid_spec_small)
        return jnp.sum(sigma**2)

    rtol = 1.0e-5 if dtype == jnp.float64 else 5.0e-3
    check_vjp(
        loss_fn,
        functools.partial(jax.vjp, loss_fn),
        args=(mu, lmbda, epsilon_mean),
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vjp_anisotropic(grid_spec_small, rng, dtype):
    if dtype == np.float32:
        pytest.skip("Test currently not reliable in single precision")
    mu, lmbda = initialise_material(grid_spec_small, rng, dtype)
    epsilon_mean = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    stiffness_tensor = perturbed_stiffness_tensor(rng, mu, lmbda, delta=0.1)

    def loss_fn(stiffness_tensor, epsilon_mean):
        epsilon, sigma = solve_anisotropic(
            stiffness_tensor, epsilon_mean, grid_spec_small
        )
        return jnp.sum(sigma**2)

    rtol = 1.0e-5 if dtype == jnp.float64 else 1.0e-3
    check_vjp(
        loss_fn,
        functools.partial(jax.vjp, loss_fn),
        args=(stiffness_tensor, epsilon_mean),
        rtol=rtol,
    )
