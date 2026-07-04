import numpy as np
import pytest
import jax
from jax import numpy as jnp
from jax.test_util import check_vjp
import functools

from jaxmaterials.solver.hooke import compute_sigma_isotropic, compute_sigma_anisotropic
from jaxmaterials.solver.lippmann_schwinger import (
    lippmann_schwinger_isotropic,
    lippmann_schwinger_anisotropic,
)
from jaxmaterials.solver.backend import _lippmann_schwinger_adjoint_jax
from jaxmaterials.solver.backend import _lippmann_schwinger_jax

from jaxmaterials.solver.backend import solve
from fixtures import (
    initialise_isotropic_material,
    reference_parameters,
    perturbed_parameters,
    grid_spec,
    grid_spec_small,
    rng,
)


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("depth", [0, 1, 2, 4, 8])
def test_adjoint_isotropic(grid_spec, rng, dtype, depth):
    """Check that isotropic adjoint solver converges in a small number of iterations"""
    f_rhs = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    params = initialise_isotropic_material(grid_spec, rng, dtype)
    ref_params = reference_parameters(params)
    epsilon = jnp.zeros((6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype)
    _, sigma_vjp = jax.vjp(compute_sigma_isotropic, epsilon, params)
    _, its = _lippmann_schwinger_adjoint_jax(
        sigma_vjp,
        f_rhs,
        ref_params,
        grid_spec,
        tol=1.0e-6 if dtype == np.float32 else 1.0e-12,
        depth=depth,
        maxits=32,
        dynamic_stopping=True,
        verbose=1,
    )
    assert its < 10 if dtype == np.float32 else 20


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("depth", [0, 1, 2, 4, 8])
def test_adjoint_anisotropic(grid_spec, rng, dtype, depth):
    """Check that anisotropic adjoint solver converges in a small number of iterations"""
    f_rhs = rng.normal(size=(6, grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    params_isotropic = initialise_isotropic_material(grid_spec, rng, dtype)
    ref_params = reference_parameters(params_isotropic)
    params = perturbed_parameters(rng, params_isotropic, delta=0.1)
    epsilon = jnp.zeros((6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype)
    _, sigma_vjp = jax.vjp(compute_sigma_anisotropic, epsilon, params)
    _, its = _lippmann_schwinger_adjoint_jax(
        sigma_vjp,
        f_rhs,
        ref_params,
        grid_spec,
        tol=1.0e-6 if dtype == np.float32 else 1.0e-12,
        depth=depth,
        maxits=32,
        dynamic_stopping=True,
        verbose=1,
    )
    assert its < 10 if dtype == np.float32 else 21


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("depth", [0, 4])
def test_vjp_isotropic_finite_difference(grid_spec_small, rng, dtype, depth):
    """Compare custom gradient with adjoint method to finite difference approximation"""
    params = initialise_isotropic_material(grid_spec_small, rng, dtype)
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    tol = 1.0e-6 if dtype == jnp.float32 else 1.0e-12

    def loss_fn(params, epsilon_bar):
        epsilon, sigma = lippmann_schwinger_isotropic(
            params,
            epsilon_bar,
            grid_spec_small,
            tol=tol,
            depth=depth,
            verbose=1,
        )
        return jnp.sum(sigma**2)

    rtol = 1.0e-7 if dtype == jnp.float64 else 2.0e-3
    check_vjp(
        loss_fn,
        functools.partial(jax.vjp, loss_fn),
        args=(params, epsilon_bar),
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("depth", [0, 4])
def test_vjp_anisotropic_finite_difference(grid_spec_small, rng, dtype, depth):
    """Compare custom gradient with adjoint method to finite difference approximation"""
    params = initialise_isotropic_material(grid_spec_small, rng, dtype)
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    params_anisotropic = perturbed_parameters(rng, params, delta=0.1)
    tol = 1.0e-4 if dtype == jnp.float32 else 1.0e-12

    def loss_fn(params_anisotropic, epsilon_bar):
        epsilon, sigma = lippmann_schwinger_anisotropic(
            params_anisotropic,
            epsilon_bar,
            grid_spec_small,
            tol=tol,
            depth=depth,
            verbose=1,
        )
        return jnp.sum(sigma**2)

    rtol = 1.0e-5 if dtype == jnp.float64 else 5.0e-2
    check_vjp(
        loss_fn,
        functools.partial(jax.vjp, loss_fn),
        args=(params_anisotropic, epsilon_bar),
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("depth", [0, 4])
def test_vjp_isotropic(grid_spec_small, rng, dtype, depth):
    """Verify that for fixed number of iteration custom gradient with adjoint method matches JAX gradient"""
    params = initialise_isotropic_material(grid_spec_small, rng, dtype)
    ref_params = reference_parameters(params)
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    tol = 1.0e-20
    maxits = 128

    def loss_fn_adjoint(params, epsilon_bar):
        epsilon, sigma = solve(
            compute_sigma_isotropic,
            params,
            epsilon_bar,
            ref_params,
            grid_spec_small,
            tol=tol,
            maxits=maxits,
            depth=depth,
            dynamic_stopping=False,
            verbose=1,
        )
        return jnp.sum(sigma**2)

    def loss_fn(params, epsilon_bar):
        epsilon, sigma = _lippmann_schwinger_jax(
            compute_sigma_isotropic,
            params,
            epsilon_bar,
            ref_params,
            grid_spec_small,
            tol=tol,
            depth=depth,
            maxits=maxits,
            dynamic_stopping=False,
            verbose=1,
        )
        return jnp.sum(sigma**2)

    g_adjoint = jax.grad(loss_fn_adjoint, argnums=(0, 1))(params, epsilon_bar)
    g_autodiff = jax.grad(loss_fn, argnums=(0, 1))(params, epsilon_bar)
    rtol = 1.0e-12 if dtype == jnp.float64 else 1.0e-4
    assert all(
        np.linalg.norm(x - y) / np.linalg.norm(y) < rtol
        for x, y in zip(jax.tree.flatten(g_adjoint)[0], jax.tree.flatten(g_autodiff)[0])
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("depth", [0, 4])
def test_vjp_anisotropic(grid_spec_small, rng, dtype, depth):
    """Verify that for fixed number of iteration custom gradient with adjoint method matches JAX gradient"""
    params = initialise_isotropic_material(grid_spec_small, rng, dtype)
    ref_params = reference_parameters(params)
    epsilon_bar = np.array([2.1, 0.9, 0.8, 0.4, 0.9, 0.5], dtype=dtype)
    params_anisotropic = perturbed_parameters(rng, params, delta=0.1)
    tol = 1.0e-20
    maxits = 128

    def loss_fn_adjoint(params_anisotropic, epsilon_bar):
        epsilon, sigma = solve(
            compute_sigma_anisotropic,
            params_anisotropic,
            epsilon_bar,
            ref_params,
            grid_spec_small,
            tol=tol,
            maxits=maxits,
            depth=depth,
            dynamic_stopping=False,
            verbose=1,
        )
        return jnp.sum(sigma**2)

    def loss_fn(params_anisotropic, epsilon_bar):
        epsilon, sigma = _lippmann_schwinger_jax(
            compute_sigma_anisotropic,
            params_anisotropic,
            epsilon_bar,
            ref_params,
            grid_spec_small,
            tol=tol,
            depth=depth,
            maxits=maxits,
            dynamic_stopping=False,
            verbose=1,
        )
        return jnp.sum(sigma**2)

    g_adjoint = jax.grad(
        loss_fn_adjoint,
        argnums=(0, 1),
    )(params_anisotropic, epsilon_bar)
    g_autodiff = jax.grad(
        loss_fn,
        argnums=(0, 1),
    )(params_anisotropic, epsilon_bar)
    rtol = 1.0e-11 if dtype == jnp.float64 else 2.0e-4
    assert all(
        np.linalg.norm(x - y) / np.linalg.norm(y) < rtol
        for x, y in zip(jax.tree.flatten(g_adjoint)[0], jax.tree.flatten(g_autodiff)[0])
    )
