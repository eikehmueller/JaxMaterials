import pytest
import numpy as np
import pytest
import jax
from jax import numpy as jnp
from jax.test_util import check_vjp
import functools


from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_isotropic,
    lippmann_schwinger_anisotropic,
)
from jaxmaterials.solver.backend import _lippmann_schwinger_adjoint_jax

from jaxmaterials.solver.backend import solve_impl, solve
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
    _, its = _lippmann_schwinger_adjoint_jax(
        {"mu": mu, "lambda": lmbda},
        f_rhs,
        grid_spec,
        isotropic=True,
        rtol=1.0e-5 if dtype == np.float32 else 1.0e-12,
        atol=1.0e-20,
        maxits=32,
        dynamic_stopping=True,
        dtype=dtype,
    )
    assert its < 10 if dtype == np.float32 else 20


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adjoint_anisotropic(grid_spec, rng, dtype):
    """Check that anisotropic adjoint solver converges in a small number of iterations"""
    f_rhs = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    mu, lmbda = initialise_material(grid_spec, rng, dtype)
    stiffness_tensor = perturbed_stiffness_tensor(rng, mu, lmbda, delta=0.1)
    _, its = _lippmann_schwinger_adjoint_jax(
        {"stiffness_tensor": stiffness_tensor},
        f_rhs,
        grid_spec,
        isotropic=False,
        rtol=1.0e-5 if dtype == np.float32 else 1.0e-12,
        atol=1.0e-20,
        maxits=32,
        dynamic_stopping=True,
        dtype=dtype,
    )
    assert its < 10 if dtype == np.float32 else 20


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vjp_isotropic_finite_difference(grid_spec_small, rng, dtype):
    """Compare custom gradient with adjoint method to finite difference approximation"""
    mu, lmbda = initialise_material(grid_spec_small, rng, dtype)
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)

    def loss_fn(mu, lmbda, epsilon_bar):
        epsilon, sigma = lippmann_schwinger_isotropic(
            mu, lmbda, epsilon_bar, grid_spec_small
        )
        return jnp.sum(sigma**2)

    rtol = 1.0e-5 if dtype == jnp.float64 else 5.0e-3
    check_vjp(
        loss_fn,
        functools.partial(jax.vjp, loss_fn),
        args=(mu, lmbda, epsilon_bar),
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vjp_anisotropic_finite_difference(grid_spec_small, rng, dtype):
    """Compare custom gradient with adjoint method to finite difference approximation"""
    if dtype == np.float32:
        pytest.skip("Test currently not reliable in single precision")
    mu, lmbda = initialise_material(grid_spec_small, rng, dtype)
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    stiffness_tensor = perturbed_stiffness_tensor(rng, mu, lmbda, delta=0.1)

    def loss_fn(stiffness_tensor, epsilon_bar):
        epsilon, sigma = lippmann_schwinger_anisotropic(
            stiffness_tensor, epsilon_bar, grid_spec_small
        )
        return jnp.sum(sigma**2)

    rtol = 1.0e-5 if dtype == jnp.float64 else 1.0e-3
    check_vjp(
        loss_fn,
        functools.partial(jax.vjp, loss_fn),
        args=(stiffness_tensor, epsilon_bar),
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vjp_isotropic(grid_spec_small, rng, dtype):
    """Verify that for fixed number of iteration custom gradient with adjoint method matches JAX gradient"""
    mu, lmbda = initialise_material(grid_spec_small, rng, dtype)
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)

    def loss_fn_adjoint(mu, lmbda, epsilon_bar):
        epsilon, sigma = solve(
            {"mu": mu, "lambda": lmbda},
            epsilon_bar,
            grid_spec_small,
            dynamic_stopping=False,
        )
        return jnp.sum(sigma**2)

    def loss_fn(mu, lmbda, epsilon_bar):
        epsilon, sigma = solve_impl(
            {"mu": mu, "lambda": lmbda},
            epsilon_bar,
            grid_spec_small,
            dynamic_stopping=False,
        )
        return jnp.sum(sigma**2)

    g_adjoint = jax.grad(loss_fn_adjoint, argnums=(0, 1))(mu, lmbda, epsilon_bar)
    g = jax.grad(loss_fn, argnums=(0, 1))(mu, lmbda, epsilon_bar)
    rtol = 1.0e-12 if dtype == jnp.float64 else 1.0e-3
    assert all(np.allclose(x, y, rtol=rtol) for x, y in zip(g, g_adjoint))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vjp_anisotropic(grid_spec_small, rng, dtype):
    """Verify that for fixed number of iteration custom gradient with adjoint method matches JAX gradient"""
    mu, lmbda = initialise_material(grid_spec_small, rng, dtype)
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    stiffness_tensor = perturbed_stiffness_tensor(rng, mu, lmbda, delta=0.1)

    def loss_fn_adjoint(stiffness_tensor, epsilon_bar):
        epsilon, sigma = solve(
            {"stiffness_tensor": stiffness_tensor},
            epsilon_bar,
            grid_spec_small,
            dynamic_stopping=False,
        )
        return jnp.sum(sigma**2)

    def loss_fn(stiffness_tensor, epsilon_bar):
        epsilon, sigma = solve_impl(
            {"stiffness_tensor": stiffness_tensor},
            epsilon_bar,
            grid_spec_small,
            dynamic_stopping=False,
        )
        return jnp.sum(sigma**2)

    g_adjoint = jax.grad(loss_fn_adjoint, argnums=(1,))(stiffness_tensor, epsilon_bar)
    g = jax.grad(loss_fn, argnums=(1,))(stiffness_tensor, epsilon_bar)
    # only check gradient with respect to epsilon_bar
    rtol = 1.0e-12 if dtype == jnp.float64 else 1.0e-5
    assert np.allclose(g, g_adjoint, rtol=rtol)
