"""Lippmann Schwinger solver with Anderson acceleration"""

import contextvars
import contextlib
import jax
from jax import numpy as jnp
from jaxmaterials.solver.derivatives import backward_divergence
from jaxmaterials.solver.fourier import (
    get_xizero,
    get_xi,
    fourier_solve_isotropic,
)
from jaxmaterials.solver.hooke import compute_sigma_isotropic

__all__ = [
    "relative_divergence",
    "relative_divergence_fourier",
    "_lippmann_schwinger_jax",
    "_lippmann_schwinger_adjoint_jax",
    "solve",
    "iteration_counter",
]

# Context-specific variable for counting the number of iterations
number_of_iterations = contextvars.ContextVar("its", default=None)


@contextlib.contextmanager
def iteration_counter():
    """Context which can be used to record the number of iterations"""
    token = number_of_iterations.set(-1)
    try:
        yield number_of_iterations
    finally:
        number_of_iterations.reset(token)


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


@jax.jit(
    static_argnames=[
        "compute_sigma",
        "grid_spec",
        "depth",
        "maxits",
        "dynamic_stopping",
        "dtype",
        "verbose",
    ]
)
def _lippmann_schwinger_jax(
    compute_sigma,
    params,
    epsilon_bar,
    ref_params,
    grid_spec,
    rtol,
    atol,
    depth,
    maxits,
    dynamic_stopping,
    dtype,
    verbose,
):
    """Lippmann Schwinger iteration with Anderson acceleration for generic stress-strain
    relationship.

    Computational routine which should not be called directly; use the interface routines instead.

    The stress-strain relationship is described by the function compute_sigma which is of the
    form

        def compute_sigma(epsilon, params):
            # compute stress sigma from strain epsilon given material parameters params
            return sigma

    Here params are the material parameters, such as the Lame coefficients for an isotropic material.

    :arg compute_sigma: function which describes the stress-strain relationship
    :arg params: material parameters which are passed on to compute_sigma()
    :arg epsilon_bar: mean value of epsilon
    :arg ref_params: Lame coefficients of isotropic reference material, dictionary of the form
        {"lambda":lambda, "mu":mu}
    :arg grid_spec: specification of computational grid
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxits: maximum number of iterations
    :arg dynamic_stopping: stop based on rtol and atol. If False, stop after maxits iterations
    :arg dtype: data type
    """
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=dtype)
    xi = get_xi(grid_spec, dtype=dtype)
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
    sigma = compute_sigma(epsilon[0, ...], params)
    # Fourier transform sigma
    sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
    rel_error = relative_divergence_fourier(sigma_hat, xi)
    rel_error_0 = rel_error
    if verbose > 1:
        jax.debug.print("==== JAX forward solve ====", ordered=True)
        jax.debug.print(
            "  iteration  E = ||div(sigma)||/||sigma||  E/E_0", ordered=True
        )

    def exit_condition(state):
        """Check exit condition

        Let e^i = <||div(sigma^i)||> / ||<sigma^i>|| be the current normalised divergence

        This method checkes whether e^i < max (atol, rtol * e^0) or its > maxits

        :arg state: current iteration state (epsilon, residual, sigma, A, , rel_error, rel_error_0)
        """
        its, rel_error = state[-2:]
        if verbose > 1:
            jax.debug.print(
                "  {:6d}  {:8.2e}  {:8.2e}",
                its,
                rel_error,
                rel_error / rel_error_0,
                ordered=True,
            )

        return (rel_error > atol) & (rel_error > rtol * rel_error_0) & (its < maxits)

    def loop_body(state):
        """Update strain, residual and stress according to update rule

        :arg state: current iteration state (epsilon, residual, sigma,sigma_hat, A_anderson, its, rel_error)
        """
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, its, rel_error = state
        # Solve reference problem hat{epsilon}_{kl} = -Gamma^0_{klij} hat{tau}_{ij}
        r_hat = fourier_solve_isotropic(sigma_hat, xizero, ref_params)
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
        sigma = compute_sigma(epsilon[0, ...], params)
        # Fourier transform sigma
        sigma_hat = jnp.fft.fftn(sigma, axes=[-3, -2, -1])
        rel_error = relative_divergence_fourier(sigma_hat, xi)
        its += 1
        return (
            epsilon,
            residual,
            sigma,
            sigma_hat,
            A_anderson,
            u_rhs,
            its,
            rel_error,
        )

    init_val = (epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, 0, rel_error_0)
    if dynamic_stopping:
        loop_result = jax.lax.while_loop(exit_condition, loop_body, init_val=init_val)
    else:
        loop_result = jax.lax.fori_loop(
            0, maxits, lambda _, state: loop_body(state), init_val=init_val
        )

    epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, its, rel_error = loop_result
    if verbose > 0:
        jax.lax.cond(
            its < maxits | jnp.logical_not(dynamic_stopping),
            lambda x, y: jax.debug.print(
                "JAX forward solver converged after {:6d} of {:6d} iterations",
                x,
                y,
                ordered=True,
            ),
            lambda x, y: jax.debug.print(
                "JAX forward solver failed to converge after {:6d} iterations",
                y,
                ordered=True,
            ),
            its,
            maxits,
        )
        jax.debug.print(
            "E = ||div(sigma)||/||sigma|| = {:8.2e} E/E_0 = {:8.2e}",
            rel_error,
            rel_error / rel_error_0,
            ordered=True,
        )

    return epsilon[0, ...], sigma, its


@jax.jit(
    static_argnames=[
        "compute_sigma",
        "grid_spec",
        "maxits",
        "dynamic_stopping",
        "dtype",
        "verbose",
    ]
)
def _lippmann_schwinger_adjoint_jax(
    compute_sigma,
    params,
    epsilon,
    f_rhs,
    ref_params,
    grid_spec,
    rtol,
    atol,
    maxits,
    dynamic_stopping,
    dtype,
    verbose,
):
    """Lippmann Schwinger iteration for adjoint equation

    Computational routine which should not be called directly.

    The stress-strain relationship is described by the function compute_sigma which is of the
    form

        def compute_sigma(epsilon, params):
            # compute stress sigma from strain epsilon given material parameters params
            return sigma

    Here params are the material parameters, such as the Lame coefficients for an isotropic material.

    :arg compute_sigma: function for computing stress-strain relationship
    :arg params: parameters of stress-strain function compute_sigma()
    :arg epsilon: strain value
    :arg f_rhs: right hand side in adjoint equation
    :arg ref_params: Lame coefficients of isotropic reference material, dictionary of form
        {"lambda":lambda, "mu":mu}
    :arg grid_spec: specification of computational grid
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg maxits: maximal number of iterations
    :arg dynamic_stopping: stop based on rtol and atol. If False, stop after maxits iterations
    :arg dtype: data type
    """
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=dtype)
    # storage for adjoint solution, array of shape (6,Nx,Ny,Nz)
    Lambda = f_rhs
    _, sigma_vjp = jax.vjp(compute_sigma, epsilon, params)
    increment_nrm = jnp.linalg.norm(Lambda)
    if verbose > 1:
        jax.debug.print("==== JAX adjoint solve ====", ordered=True)
        jax.debug.print(
            "  iteration  E = ||delta(Lambda)||  ||delta(Lambda)||/||Lambda||",
            ordered=True,
        )

    def exit_condition(state):
        """Check exit condition

        :arg state: current iteration state (epsilon, residual, sigma, A, its, rel_error, rel_error_0)
        """
        Lambda, increment_nrm, its = state
        nrm = jnp.linalg.norm(Lambda)
        if verbose > 1:
            jax.debug.print(
                "  {:6d}  {:8.2e}  {:8.2e}",
                its,
                increment_nrm,
                increment_nrm / nrm,
                ordered=True,
            )

        return (increment_nrm > atol) & (increment_nrm > rtol * nrm) & (its < maxits)

    def loop_body(state):
        """Update strain, residual and stress according to update rule

        :arg state: current iteration state (epsilon, residual, sigma,sigma_hat, A_anderson, its, rel_error)
        """
        Lambda, increment_nrm, its = state
        Lambda_hat = jnp.fft.fftn(Lambda, axes=(-3, -2, -1))
        Theta_hat = fourier_solve_isotropic(Lambda_hat, xizero, ref_params)
        Theta = jnp.real(jnp.fft.ifftn(Theta_hat, axes=(-3, -2, -1)))
        dSigma_depsilon, _ = sigma_vjp(Theta)
        Delta = dSigma_depsilon - compute_sigma_isotropic(Theta, ref_params)
        Lambda_prev = Lambda
        Lambda = f_rhs + Delta
        its += 1
        increment_nrm = jnp.linalg.norm(Lambda - Lambda_prev)
        return Lambda, increment_nrm, its

    if dynamic_stopping:
        loop_result = jax.lax.while_loop(
            exit_condition, loop_body, init_val=(Lambda, increment_nrm, 0)
        )
    else:
        loop_result = jax.lax.fori_loop(
            0,
            maxits,
            lambda _, state: loop_body(state),
            init_val=(Lambda, increment_nrm, 0),
        )

    Lambda, increment_nrm, its = loop_result
    if verbose > 0:
        nrm = jnp.linalg.norm(Lambda)
        jax.lax.cond(
            its < maxits | jnp.logical_not(dynamic_stopping),
            lambda x, y: jax.debug.print(
                "JAX adjoint solver converged after {:6d} of {:6d} iterations",
                x,
                y,
                ordered=True,
            ),
            lambda x, y: jax.debug.print(
                "JAX adjoint solver failed to converge after {:6d} iterations",
                y,
                ordered=True,
            ),
            its,
            maxits,
        )
        jax.debug.print(
            "||delta(Lambda)|| = {:8.2e} ||delta(Lambda)||/||Lambda|| = {:8.2e}",
            increment_nrm,
            increment_nrm / nrm,
            ordered=True,
        )

    return Lambda, its


def solve_impl(
    compute_sigma,
    params,
    epsilon_bar,
    ref_params,
    grid_spec,
    tol,
    maxits,
    depth,
    dynamic_stopping,
    verbose=0,
):
    """Backend implementation of the forward solve

    :arg compute_sigma: stress-strain relationship
    :arg params: parameters of stress-strain function
    :arg epsilon_bar: average strain
    :arg ref_params: dictionary with Lame coefficients or symmetric stiffness tensor
    :arg grid_spec: specifications of computational grid
    :arg tol: tolerance for Lippmann Schwinger solver
    :arg maxits: maximum number of iterations
    :arg depth: depth of Anderson acceleration
    :arg dynamic_stopping: use dynamic stopping criterion? Otherwise, carry out fixed number
        of iterations as specified by maxits
    :arg verbose: verbosity level
    """
    dtype = epsilon_bar.dtype
    epsilon, sigma, its = _lippmann_schwinger_jax(
        compute_sigma,
        params,
        epsilon_bar,
        ref_params,
        grid_spec,
        rtol=1.0e-20,
        atol=tol,
        depth=depth,
        maxits=maxits,
        dynamic_stopping=dynamic_stopping,
        dtype=dtype,
        verbose=verbose,
    )
    if number_of_iterations is not None:
        number_of_iterations.set(its)
    return epsilon, sigma


def solve_fwd(
    compute_sigma,
    params,
    epsilon_bar,
    ref_params,
    grid_spec,
    tol,
    maxits,
    depth,
    dynamic_stopping,
    verbose=0,
):
    """Forward solve to compute stress and strain for given material parameters and epsilon_bar

    :arg compute_sigma: stress-strain relationship
    :arg params: parameters of stress-strain function
    :arg epsilon_bar: average strain
    :arg ref_params: dictionary with Lame coefficients or symmetric stiffness tensor
    :arg tol: tolerance for Lippmann Schwinger solver
    :arg maxits: maximum number of iterations
    :arg depth: depth of Anderson acceleration
    :arg dynamic_stopping: use dynamic stopping criterion?
    :arg verbose: verbosity level
    """
    out = solve_impl(
        compute_sigma,
        params,
        epsilon_bar,
        ref_params,
        grid_spec,
        tol,
        maxits,
        depth,
        dynamic_stopping,
        verbose,
    )
    epsilon, sigma = out
    return out, (compute_sigma, params, ref_params, grid_spec, epsilon, sigma)


def solve_bwd(tol, maxits, dynamic_stopping, verbose, res, gradients):
    """Backward solve

    Returns gradients with respect to material parameters and epsilon_bar

    :arg tol: tolerance for adjoint solve
    :arg maxits: maximum number of iterations
    :arg depth: Anderson depth of forward solve
    :arg dynamic_stopping: use dynamic stopping criterion?
    :arg verbose: verbosity level
    :arg res: results object returned by solve_fwd()
    :arg gradients: Riesz-representer of input gradients
    """
    compute_sigma, params, ref_params, grid_spec, epsilon, _ = res
    dtype = epsilon.dtype
    xizero = get_xizero(grid_spec, dtype=dtype)
    # Incoming gradients are dual vectors with respect to *Euclidean*
    # inner product, need be converted to dual vectors with respect to
    # weighted dot-product is Voigt notation:
    #
    #   <a,b>_V = a_0*b_0 + a_1*b_1 + a_2*b_2
    #           + 2 * ( a_3*b_3 + a_4*b_4 + a_5*b_5 )
    g_epsilon, g_sigma = gradients
    voigt_weights = jnp.array([1, 1, 1, 2, 2, 2], dtype=dtype)
    voigt_weights_bcast = voigt_weights[:, None, None, None]
    # solve adjoint equation
    sigma_vjp = jax.vjp(compute_sigma, epsilon, params)
    f_rhs = -(g_epsilon + sigma_vjp((g_sigma, None))[0]) / voigt_weights_bcast
    Lambda, _ = _lippmann_schwinger_adjoint_jax(
        compute_sigma,
        params,
        epsilon,
        f_rhs,
        ref_params,
        grid_spec,
        atol=1.0e-20,
        rtol=tol,
        maxits=maxits,
        dynamic_stopping=dynamic_stopping,
        dtype=dtype,
        verbose=verbose,
    )

    Lambda_hat = jnp.fft.fftn(Lambda, axes=(-3, -2, -1))
    Theta_hat = fourier_solve_isotropic(Lambda_hat, xizero, ref_params)
    S_star = g_sigma - jnp.real(jnp.fft.ifftn(Theta_hat, axes=(-3, -2, -1)))
    # Convert back to dual vector with respect to Euclidean inner product.
    g_epsilon_bar = -voigt_weights * jnp.sum(Lambda, axis=(1, 2, 3))
    # Derivative with respect to parameters
    g_params = sigma_vjp((None, S_star))[1]
    return g_epsilon_bar, g_params


# Register custom forward solve and reverse mode gradient
solve = jax.custom_vjp(
    solve_impl,
    nondiff_argnames=(
        "grid_spec",
        "tol",
        "maxits",
        "depth",
        "dynamic_stopping",
        "verbose",
    ),
)

solve.defvjp(solve_fwd, solve_bwd)
