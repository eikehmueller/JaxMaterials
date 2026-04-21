"""Lippmann Schwinger solver with Anderson acceleration"""

import jax
from jax import numpy as jnp
from jaxmaterials.solver.derivatives import backward_divergence
from jaxmaterials.solver.fourier import (
    get_xizero,
    get_xi,
    get_inverse_anisotropic_acoustic_tensor,
    fourier_solve_isotropic,
    fourier_solve_anisotropic,
)
from jaxmaterials.solver.hooke import compute_sigma_isotropic, compute_sigma_anisotropic

__all__ = [
    "relative_divergence",
    "relative_divergence_fourier",
    "_lippmann_schwinger_jax",
    "_lippmann_schwinger_adjoint_jax",
    "solve_isotropic",
]


def relative_divergence(sigma, grid_spec):
    """Compute ratio of the norm of div(sigma) and the norm of the average sigma

    :arg sigma: stress
    :arg grid_spec: grid specification
    """
    dsigma = backward_divergence(sigma, grid_spec)
    dsigma_nrm = jnp.sqrt(jnp.sum(dsigma**2))
    sigma_avg = jnp.mean(sigma, axis=[1, 2, 3])
    sigma_avg_nrm = jnp.sqrt(
        jnp.sum(sigma_avg[:3] ** 2) + 2 * jnp.sum(sigma_avg[3:] ** 2)
    )
    return dsigma_nrm / (jnp.sqrt(grid_spec.number_of_voxels) * sigma_avg_nrm)


def relative_divergence_fourier(sigma_hat, xi):
    """Compute ratio of the norm of div(sigma) and the norm of the average sigma in Fourier space

    :arg sigma_hat: stress in Fourier space
    :arg xi: Fourier vectors
    """
    dsigma_hat = jnp.stack(
        [
            xi[0, ...] * sigma_hat[0, ...]
            + xi[1, ...] * sigma_hat[3, ...]
            + xi[2, ...] * sigma_hat[4, ...],
            xi[0, ...] * sigma_hat[3, ...]
            + xi[1, ...] * sigma_hat[1, ...]
            + xi[2, ...] * sigma_hat[5, ...],
            xi[0, ...] * sigma_hat[4, ...]
            + xi[1, ...] * sigma_hat[5, ...]
            + xi[2, ...] * sigma_hat[2, ...],
        ]
    )
    dsigma_nrm = jnp.sqrt(jnp.sum(jnp.abs(dsigma_hat) ** 2))
    sigma_hat_zero = jnp.real(sigma_hat[:, 0, 0, 0])
    sigma_hat_zero_nrm = jnp.sqrt(
        jnp.sum(sigma_hat_zero[:3] ** 2) + 2 * jnp.sum(sigma_hat_zero[3:] ** 2)
    )
    return dsigma_nrm / sigma_hat_zero_nrm


@jax.jit(static_argnames=["grid_spec", "isotropic", "depth", "dtype", "maxiter"])
def _lippmann_schwinger_jax(
    material_properties,
    epsilon_bar,
    grid_spec,
    isotropic,
    rtol,
    atol,
    depth,
    maxiter,
    dtype,
):
    """Lippmann Schwinger iteration with Anderson acceleration for linear elasticity

    Computational routine which should not be called directly; use the interface routines
    lippmann_schwinger_isotropic_jax() and lippmann_schwinger_anisotropic_jax() instead.
    The dictionary 'material_properties' is of the form {"lambda":lambda,"mu":mu} for an
    isotropic material and of the form {"stiffness_tensor":C} for an anisotropic material.

    :arg material_properties: dictionary with material properties
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg isotropic: isotropic material?
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=dtype)
    xi = get_xi(grid_spec, dtype=dtype)
    # reference values of Lame parameter
    if isotropic:
        mu = material_properties["mu"]
        lmbda = material_properties["lambda"]
        mu0 = 1 / 2 * (jnp.min(mu) + jnp.max(mu))
        lmbda0 = 1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda))
    else:
        stiffness_tensor = material_properties["stiffness_tensor"]
        stiffness_tensor0 = (
            1
            / 2
            * (
                jnp.min(stiffness_tensor, axis=(1, 2, 3))
                + jnp.max(stiffness_tensor, axis=(1, 2, 3))
            )
        )
        N_ref = get_inverse_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)

    # storage for solution and residual, arrays of shape (d+1,6,Nx,Ny,Nz)
    epsilon = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz),
        dtype=dtype,
    )
    epsilon = epsilon.at[0, ...].set(
        jnp.expand_dims(jnp.astype(epsilon_bar, dtype), [1, 2, 3])
    )
    residual = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype
    )
    # Anderson matrix and vectors
    A_anderson = jnp.eye(depth + 1, dtype=dtype)
    u_rhs = jnp.zeros(depth + 1, dtype=dtype)
    if isotropic:
        sigma = compute_sigma_isotropic(lmbda, mu, epsilon[0, ...])
    else:
        sigma = compute_sigma_anisotropic(stiffness_tensor, epsilon[0, ...])
    # Fourier transform sigma
    sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
    rel_error = relative_divergence_fourier(sigma_hat, xi)
    rel_error_0 = rel_error

    def exit_condition(state):
        """Check exit condition

        Let e^i = <||div(sigma^i)||> / ||<sigma^i>|| be the current normalised divergence

        This method checkes whether e^i < max (atol, rtol * e^0) or iter > maxiter

        :arg state: current iteration state (epsilon, residual, sigma, A, iter, rel_error, rel_error_0)
        """
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = state
        return (rel_error > atol) & (rel_error > rtol * rel_error_0) & (iter < maxiter)

    def loop_body(state):
        """Update strain, residual and stress according to update rule

        :arg state: current iteration state (epsilon, residual, sigma,sigma_hat, A_anderson, iter, rel_error)
        """
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = state
        # Solve reference problem hat{epsilon}_{kl} = -Gamma^0_{klij} hat{tau}_{ij}
        if isotropic:
            r_hat = fourier_solve_isotropic(sigma_hat, lmbda0, mu0, xizero)
        else:
            r_hat = fourier_solve_anisotropic(sigma_hat, N_ref, xizero)
        r = jnp.real(jnp.fft.ifftn(r_hat, axes=[-3, -2, -1]))
        residual = jnp.roll(residual, 1, axis=0)
        residual = residual.at[0, ...].set(r)
        A_anderson = jnp.roll(A_anderson, (1, 1), axis=(0, 1))
        dotproduct_scaling = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=dtype)
        A_anderson = A_anderson.at[0, :].set(
            jnp.einsum("aijk,saijk,a->s", r, residual, dotproduct_scaling)
        )
        A_anderson = A_anderson.at[:, 0].set(A_anderson[0, :])
        u_rhs = jnp.roll(u_rhs, 1)
        u_rhs = u_rhs.at[0].set(1)
        v = jnp.linalg.solve(A_anderson, u_rhs)
        alpha = v / jnp.dot(v, u_rhs)
        epsilon_tilde = jnp.einsum("s,saijk", alpha, epsilon + residual)
        epsilon = jnp.roll(epsilon, 1, axis=0)
        epsilon = epsilon.at[0, ...].set(epsilon_tilde)
        if isotropic:
            sigma = compute_sigma_isotropic(lmbda, mu, epsilon[0, ...])
        else:
            sigma = compute_sigma_anisotropic(stiffness_tensor, epsilon[0, ...])
        # Fourier transform sigma
        sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
        rel_error = relative_divergence_fourier(sigma_hat, xi)
        iter += 1
        return (
            epsilon,
            residual,
            sigma,
            sigma_hat,
            A_anderson,
            u_rhs,
            iter,
            rel_error,
        )

    epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, iter, rel_error = (
        jax.lax.while_loop(
            exit_condition,
            loop_body,
            init_val=(
                epsilon,
                residual,
                sigma,
                sigma_hat,
                A_anderson,
                u_rhs,
                0,
                rel_error_0,
            ),
        )
    )

    return epsilon[0, ...], sigma, iter


@jax.jit(static_argnames=["grid_spec", "isotropic", "dtype", "maxiter"])
def _lippmann_schwinger_adjoint_jax(
    material_properties,
    f_rhs,
    grid_spec,
    isotropic,
    rtol,
    atol,
    maxiter,
    dtype,
):
    """Lippmann Schwinger iteration for adjoint equation of linear elasticity

    Computational routine which should not be called directly.
    The dictionary 'material_properties' is of the form {"lambda":lambda,"mu":mu} for an
    isotropic material and of the form {"stiffness_tensor":C} for an anisotropic material.

    :arg material_properties: dictionary with material properties
    :arg f_rhs: right hand side function
    :arg grid_spec: grid specification as a namedtuple
    :arg isotropic: isotropic material?
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=dtype)
    # reference values of Lame parameter
    if isotropic:
        mu = material_properties["mu"]
        lmbda = material_properties["lambda"]
        mu0 = 1 / 2 * (jnp.min(mu) + jnp.max(mu))
        lmbda0 = 1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda))
    else:
        stiffness_tensor = material_properties["stiffness_tensor"]
        stiffness_tensor0 = (
            1
            / 2
            * (
                jnp.min(stiffness_tensor, axis=(1, 2, 3))
                + jnp.max(stiffness_tensor, axis=(1, 2, 3))
            )
        )
        N_ref = get_inverse_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)

    # storage for adjoint solution, array of shape (6,Nx,Ny,Nz)
    Lambda = f_rhs
    increment_nrm = jnp.linalg.norm(Lambda)

    def exit_condition(state):
        """Check exit condition

        :arg state: current iteration state (epsilon, residual, sigma, A, iter, rel_error, rel_error_0)
        """
        Lambda, increment_nrm, iter = state
        nrm = jnp.linalg.norm(Lambda)
        return (increment_nrm > atol) & (increment_nrm > rtol * nrm) & (iter < maxiter)

    def loop_body(state):
        """Update strain, residual and stress according to update rule

        :arg state: current iteration state (epsilon, residual, sigma,sigma_hat, A_anderson, iter, rel_error)
        """
        Lambda, increment_nrm, iter = state
        Lambda_hat = jnp.fft.fftn(Lambda, axes=(-3, -2, -1))
        if isotropic:
            Theta_hat = fourier_solve_isotropic(Lambda_hat, lmbda0, mu0, xizero)
        else:
            Theta_hat = fourier_solve_anisotropic(Lambda_hat, N_ref, xizero)
        Theta = jnp.real(jnp.fft.ifftn(Theta_hat, axes=(-3, -2, -1)))
        if isotropic:
            Delta = compute_sigma_isotropic(lmbda - lmbda0, mu - mu0, Theta)
        else:
            Delta = compute_sigma_anisotropic(
                stiffness_tensor - stiffness_tensor0[..., None, None, None], Theta
            )
        Lambda_prev = Lambda
        Lambda = f_rhs + Delta
        iter += 1
        increment_nrm = jnp.linalg.norm(Lambda - Lambda_prev)
        return Lambda, increment_nrm, iter

    Lambda, increment_nrm, iter = jax.lax.while_loop(
        exit_condition,
        loop_body,
        init_val=(Lambda, increment_nrm, 0),
    )

    return Lambda, iter


def solve_isotropic_impl(mu, lmbda, epsilon_bar, grid_spec):
    epsilon, sigma, _ = _lippmann_schwinger_jax(
        {"mu": mu, "lambda": lmbda},
        epsilon_bar,
        grid_spec,
        True,
        rtol=1.0e-20,
        atol=1.0e-6 if mu.dtype == jnp.float32 else 1.0e-12,
        depth=0,
        maxiter=32,
        dtype=mu.dtype,
    )
    return epsilon, sigma


def _solve_isotropic(mu, lmbda, epsilon_bar, grid_spec):
    return solve_isotropic_impl(mu, lmbda, epsilon_bar, grid_spec)


solve_isotropic = jax.custom_vjp(_solve_isotropic, nondiff_argnames=("grid_spec",))


def solve_isotropic_fwd(mu, lmbda, epsilon_bar, grid_spec):
    """Forward solve"""
    out = solve_isotropic_impl(mu, lmbda, epsilon_bar, grid_spec)
    epsilon, sigma = out
    return out, (mu, lmbda, epsilon, sigma)


def solve_isotropic_bwd(grid_spec, res, gradients):
    """Backward solve"""
    mu, lmbda, epsilon, _ = res
    mu0 = 1 / 2 * (jnp.min(mu) + jnp.max(mu))
    lmbda0 = 1 / 2 * (jnp.min(lmbda) + jnp.max(lmbda))
    # Incoming gradients are dual vectors with respect to *Euclidean*
    # inner product, need be converted to dual vectors with respect to
    # weighted dot-product is Voigt notation:
    #
    #   <a,b>_V = a_0*b_0 + a_1*b_1 + a_2*b_2
    #           + 2 * ( a_3*b_3 + a_4*b_4 + a_5*b_5 )
    g_epsilon_euclidean, g_sigma_euclidean = gradients
    voigt_weights = jnp.array([1, 1, 1, 2, 2, 2], dtype=mu.dtype)
    voigt_weights_bcast = voigt_weights[:, None, None, None]
    g_epsilon = g_epsilon_euclidean / voigt_weights_bcast
    g_sigma = g_sigma_euclidean / voigt_weights_bcast
    # solve adjoint equation
    f_rhs = -(g_epsilon + compute_sigma_isotropic(lmbda, mu, g_sigma))
    Lambda, _ = _lippmann_schwinger_adjoint_jax(
        {"mu": mu, "lambda": lmbda},
        f_rhs,
        grid_spec,
        isotropic=True,
        rtol=1.0e-5 if mu.dtype == jnp.float32 else 1.0e-12,
        atol=1.0e-20,
        maxiter=32,
        dtype=mu.dtype,
    )
    xizero = get_xizero(grid_spec, dtype=mu.dtype)
    Lambda_hat = jnp.fft.fftn(Lambda, axes=(-3, -2, -1))
    Theta_hat = fourier_solve_isotropic(Lambda_hat, lmbda0, mu0, xizero)
    A = g_sigma - jnp.real(jnp.fft.ifftn(Theta_hat, axes=(-3, -2, -1)))
    tr_A = jnp.sum(A[:3], axis=0)
    tr_epsilon = jnp.sum(epsilon[:3], axis=0)
    g_lambda = tr_A * tr_epsilon
    g_mu = 2 * jnp.sum(A * epsilon * voigt_weights_bcast, axis=0)
    # Convert back to dual vector with respect to Euclidean inner
    # product.
    g_epsilon_bar = -voigt_weights * jnp.sum(Lambda, axis=(1, 2, 3))
    return g_mu, g_lambda, g_epsilon_bar


solve_isotropic.defvjp(solve_isotropic_fwd, solve_isotropic_bwd)
