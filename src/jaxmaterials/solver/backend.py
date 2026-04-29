"""Lippmann Schwinger solver with Anderson acceleration"""

import contextvars
import contextlib
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
        "grid_spec",
        "isotropic",
        "depth",
        "maxits",
        "dynamic_stopping",
        "dtype",
        "verbose",
    ]
)
def _lippmann_schwinger_jax(
    material_properties,
    epsilon_bar,
    grid_spec,
    isotropic,
    rtol,
    atol,
    depth,
    maxits,
    dynamic_stopping,
    dtype,
    verbose,
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
    :arg maxits: maximal number of iterations
    :arg dynamic_stopping: stop based on rtol and atol. If False, stop after maxits iterations
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
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, its, rel_error = state
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
        "grid_spec",
        "isotropic",
        "maxits",
        "dynamic_stopping",
        "dtype",
        "verbose",
    ]
)
def _lippmann_schwinger_adjoint_jax(
    material_properties,
    f_rhs,
    grid_spec,
    isotropic,
    rtol,
    atol,
    maxits,
    dynamic_stopping,
    dtype,
    verbose,
):
    """Lippmann Schwinger itsation for adjoint equation of linear elasticity

    Computational routine which should not be called directly.
    The dictionary 'material_properties' is of the form {"lambda":lambda,"mu":mu} for an
    isotropic material and of the form {"stiffness_tensor":C} for an anisotropic material.

    :arg material_properties: dictionary with material properties
    :arg f_rhs: right hand side function
    :arg grid_spec: grid specification as a namedtuple
    :arg isotropic: isotropic material?
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg maxits: maximal number of iterations
    :arg dynamic_stopping: stop based on rtol and atol. If False, stop after maxits iterations
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
    material_properties,
    epsilon_bar,
    grid_spec,
    tol,
    maxits,
    depth,
    dynamic_stopping,
    verbose=0,
):
    """Backend implementation of the forward solve

    :arg material_properties: dictionary with Lame coefficients or
        symmetric stiffness tensor
    :arg epsilon_bar: average strain
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
        material_properties,
        epsilon_bar,
        grid_spec,
        isotropic={"mu", "lambda"} == set(material_properties.keys()),
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
    if its >= maxits and dynamic_stopping:
        raise RuntimeError(
            f"Lippmann Schwinger Solver failed to converge after {maxits} iterations"
        )
    return epsilon, sigma


def solve_fwd(
    material_properties,
    epsilon_bar,
    grid_spec,
    tol,
    maxits,
    depth,
    dynamic_stopping,
    verbose=0,
):
    """Forward solve to compute stress and strain for given material parameters and epsilon_bar

    :arg material_properties: dictionary which contains either Lame parameters mu, lambda
        or the 21 independent components of the 6x6 stiffness tensor.
    :arg epsilon_bar: mean value of strain
    :arg tol: tolerance for Lippmann Schwinger solver
    :arg maxits: maximum number of iterations
    :arg depth: depth of Anderson acceleration
    :arg dynamic_stopping: use dynamic stopping criterion?
    :arg verbose: verbosity level
    """
    out = solve_impl(
        material_properties,
        epsilon_bar,
        grid_spec,
        tol,
        maxits,
        depth,
        dynamic_stopping,
        verbose,
    )
    epsilon, sigma = out
    return out, (material_properties, epsilon, sigma)


def solve_bwd(grid_spec, tol, maxits, depth, dynamic_stopping, verbose, res, gradients):
    """Backward solve

    Returns gradients with respect to material parameters and epsilon_bar

    :arg grid_spec: specification of computational grid
    :arg tol: tolerance for adjoint solve
    :arg maxits: maximum number of iterations
    :arg depth: Anderson depth of forward solve
    :arg dynamic_stopping: use dynamic stopping criterion?
    :arg verbose: verbosity level
    :arg res: results object returned by solve_fwd()
    :arg gradients: Riesz-representer of input gradients
    """
    material_properties = res[0]
    epsilon = res[1]
    dtype = epsilon.dtype
    xizero = get_xizero(grid_spec, dtype=dtype)
    isotropic = {"mu", "lambda"} == set(material_properties.keys())
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
    # Incoming gradients are dual vectors with respect to *Euclidean*
    # inner product, need be converted to dual vectors with respect to
    # weighted dot-product is Voigt notation:
    #
    #   <a,b>_V = a_0*b_0 + a_1*b_1 + a_2*b_2
    #           + 2 * ( a_3*b_3 + a_4*b_4 + a_5*b_5 )
    g_epsilon_euclidean, g_sigma_euclidean = gradients
    voigt_weights = jnp.array([1, 1, 1, 2, 2, 2], dtype=dtype)
    voigt_weights_bcast = voigt_weights[:, None, None, None]
    g_epsilon = g_epsilon_euclidean / voigt_weights_bcast
    g_sigma = g_sigma_euclidean / voigt_weights_bcast
    # solve adjoint equation
    if isotropic:
        Cg_sigma = compute_sigma_isotropic(lmbda, mu, g_sigma)
    else:
        Cg_sigma = compute_sigma_anisotropic(stiffness_tensor, g_sigma)
    f_rhs = -(g_epsilon + Cg_sigma)
    Lambda, its = _lippmann_schwinger_adjoint_jax(
        material_properties,
        f_rhs,
        grid_spec,
        isotropic=isotropic,
        atol=1.0e-20,
        rtol=tol,
        maxits=maxits,
        dynamic_stopping=dynamic_stopping,
        dtype=dtype,
        verbose=verbose,
    )
    if its > maxits:
        raise RuntimeError(
            f"Adjoint Lippmann Schwinger Solver failed to converge after {maxits} iterations"
        )

    Lambda_hat = jnp.fft.fftn(Lambda, axes=(-3, -2, -1))
    if isotropic:
        Theta_hat = fourier_solve_isotropic(Lambda_hat, lmbda0, mu0, xizero)
    else:
        Theta_hat = fourier_solve_anisotropic(Lambda_hat, N_ref, xizero)
    A = g_sigma - jnp.real(jnp.fft.ifftn(Theta_hat, axes=(-3, -2, -1)))
    # Convert back to dual vector with respect to Euclidean inner
    # product.
    g_epsilon_bar = -voigt_weights * jnp.sum(Lambda, axis=(1, 2, 3))
    if isotropic:
        tr_A = jnp.sum(A[:3], axis=0)
        tr_epsilon = jnp.sum(epsilon[:3], axis=0)
        g_lambda = tr_A * tr_epsilon
        g_mu = 2 * jnp.einsum("aijk,aijk,a->ijk", A, epsilon, voigt_weights)
        return {"mu": g_mu, "lambda": g_lambda}, g_epsilon_bar
    else:
        # indices that are used to construct the 21 components of the symmetric tensor
        # from A and epsilon
        product_indices = (
            (0, 0),  # (00,00)
            (1, 1),  # (11,11)
            (2, 2),  # (22,22)
            (3, 3),  # (01,01)
            (4, 4),  # (02,02)
            (5, 5),  # (12,12)
            (0, 1),  # (00,11)
            (0, 2),  # (00,22)
            (1, 2),  # (11,22)
            (0, 3),  # (00,01)
            (0, 4),  # (00,02)
            (0, 5),  # (00,12)
            (1, 3),  # (11,01)
            (1, 4),  # (11,02)
            (1, 5),  # (11,12)
            (2, 3),  # (22,01)
            (2, 4),  # (22,02)
            (2, 5),  # (22,12)
            (3, 4),  # (01,02)
            (3, 5),  # (01,12)
            (4, 5),  # (02,12)
        )

        def symmetrized_product(S, T, a, b):
            """Symmetrized product of two tensor components

            Returns S_a * S_b is a==b and S_a * S_b + S_b * S_a otherwise

            :arg S: first tensor
            :arg T: second tensor
            :arg a: first index
            :arg b: second index
            """
            return S[a] * T[b] if a == b else (S[a] * T[b] + S[b] * T[a])

        g_stiffness_tensor = jnp.stack(
            [
                symmetrized_product(A, epsilon, a, b)
                * voigt_weights[a]
                * voigt_weights[b]
                for (a, b) in product_indices
            ]
        )
        return {"stiffness_tensor": g_stiffness_tensor}, g_epsilon_bar


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
