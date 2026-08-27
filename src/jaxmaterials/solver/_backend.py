"""Backend implementation of Lippmann Schwinger solver with Anderson acceleration

Computational routines for solving the forward and adjoint Lippmann Schwinger equation.
The methods should not be called directly; use the interface routines in
:py:mod:`jaxmaterials.solver.lippmann_schwinger` instead.
"""

from collections.abc import Callable
from typing import Any, TypeAlias
from functools import partial
import jax
from jax import numpy as jnp
from jaxmaterials.common import GridSpec
from jaxmaterials.solver.fourier import (
    get_xizero,
    get_xi,
    fourier_solve_isotropic,
)
from jaxmaterials.solver.divergence import (
    relative_divergence_fourier,
)
from jaxmaterials.solver.hooke import compute_sigma_isotropic

__all__ = ["solve"]

PyTree: TypeAlias = Any


@jax.jit(
    static_argnames=[
        "compute_sigma",
        "grid_spec",
        "maxits",
        "depth",
        "dynamic_stopping",
        "verbose",
    ]
)
def _lippmann_schwinger_jax(
    compute_sigma: Callable[[jax.Array, PyTree], jax.Array],
    params: PyTree,
    epsilon_bar: jax.Array,
    delta_epsilon_initial: jax.Array,
    ref_params: dict[str, float],
    grid_spec: GridSpec,
    tol: float,
    maxits: int,
    depth: int,
    dynamic_stopping: bool,
    verbose: int,
) -> tuple[jax.Array, jax.Array]:
    """Lippmann Schwinger iteration with Anderson acceleration for generic stress-strain
    relationship

    The stress-strain relationship is described by the passed function ``compute_sigma()``
    which is of the form::

        def compute_sigma(epsilon, params):
            # compute stress sigma from strain epsilon given material parameters params
            return sigma

    as discussed in :py:mod:`jaxmaterials.solver.hooke`. Here ``params`` are the material parameters,
    such as the spatially varying Lame coefficients for an isotropic material.

    Parameters
    ==========
    compute_sigma :
        function which describes the stress-strain relationship, see :py:mod:`jaxmaterials.solver.hooke`
    params :
        material parameters which are passed on to ``compute_sigma()``
    epsilon_bar :
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    delta_epsilon_initial :
        initial strain perturbation :math:`\\delta\\epsilon`, which needs to average to zero. :math:`\\epsilon` is initialised to :math:`\\overline{\\epsilon}+\\delta\\epsilon`
    ref_params :
        Lame coefficients of isotropic reference material, dictionary of the form
        ``{"lambda":lambda, "mu":mu}``
    grid_spec :
        specification of computational grid
    tol :
        absolute tolerance on normalised stress divergence to check convergence
    maxits :
        maximum number of iterations
    depth :
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    dynamic_stopping :
        stop based on ``rtol`` and ``atol``? If ``False``, stop after exactly ``maxits`` iterations
    verbose :
        verbosity level

    Returns
    =======
    tuple[jax.Array, jax.Array]
        Strain :math:`\\epsilon` and stress :math:`\\sigma`
    """
    atol = tol
    rtol = 1.0e-20
    dtype = epsilon_bar.dtype
    # Fourier vectors
    xizero = get_xizero(grid_spec, dtype=dtype)
    xi = get_xi(grid_spec, dtype=dtype)
    # storage for solution and residual, arrays of shape (d+1,6,Nx,Ny,Nz)
    epsilon = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz),
        dtype=dtype,
    )
    epsilon = epsilon.at[0, ...].set(
        jnp.astype(
            jnp.expand_dims(epsilon_bar, axis=(1, 2, 3))
            + jax.lax.stop_gradient(delta_epsilon_initial),
            dtype,
        )
    )
    residual = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype
    )
    # Anderson matrix and vectors
    A_anderson = jnp.eye(depth + 1, dtype=jnp.float64)
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

    def exit_condition(
        state: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            int,
            jax.Array,
        ],
    ) -> jax.Array:
        """Check exit condition

        Let

        .. math::

            e^{(i)} = \\frac{\\langle\\|\\partial \\sigma^{(i)}\\|_2\\rangle}{\\|\\langle \\sigma^{(i)}\\rangle\\|_2}

        be the current normalised divergence

        This method checks whether :math:`e^{(i)}` is sufficiently small or the maximum number of
        iterations has been reached.

        Parameters
        ==========
        state :
            current iteration state ``(epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, its, rel_error)``

        Returns
        =======
        jax.Array
            ``True`` if :math:`e^{(i)} < \\max\\{atol, rtol\\cdot e^{(0)}\\}` or :math:`its > maxits`
        """
        its, rel_error = state[-2:]
        if verbose > 1:
            jax.debug.print(
                "  {:6d}  {:8.2e}  {:8.2e}",
                its,
                rel_error,
                rel_error / (rel_error_0 + 1.0e-20),
                ordered=True,
            )

        return (rel_error > atol) & (rel_error > rtol * rel_error_0) & (its < maxits)

    def loop_body(
        state: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            int,
            jax.Array,
        ],
    ) -> tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int, jax.Array
    ]:
        """Update strain, residual and stress according to update rule

        Parameters
        ==========
        state :
            current iteration state ``(epsilon, residual, sigma,sigma_hat, A_anderson, u_rhs, its, rel_error)``

        Returns
        =======
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int, jax.Array]
            Updated iteration state
        """
        epsilon, residual, sigma, sigma_hat, A_anderson, u_rhs, its, rel_error = state
        # Solve reference problem hat{epsilon}_{kl} = -Gamma^0_{klij} hat{tau}_{ij}
        r_hat = fourier_solve_isotropic(sigma_hat, xizero, ref_params)
        r = jnp.real(jnp.fft.ifftn(r_hat, axes=[-3, -2, -1]))
        if depth > 0:
            residual = jnp.roll(residual, 1, axis=0)
            residual = residual.at[0, ...].set(r)
            A_anderson = jnp.roll(A_anderson, (1, 1), axis=(0, 1))
            dotproduct_scaling = jnp.array(
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=jnp.float64
            )
            A_anderson = A_anderson.at[0, :].set(
                jnp.einsum(
                    "aijk,saijk,a->s",
                    jnp.astype(r, jnp.float64),
                    jnp.astype(residual, jnp.float64),
                    dotproduct_scaling,
                )
            )
            A_anderson = A_anderson.at[:, 0].set(A_anderson[0, :])
            u_rhs = jnp.roll(u_rhs, 1)
            u_rhs = u_rhs.at[0].set(1)
            v = jnp.linalg.solve(A_anderson, u_rhs)
            alpha = v / jnp.dot(v, u_rhs)
            epsilon_tilde = jnp.einsum("s,saijk", alpha, epsilon + residual)
            epsilon = jnp.roll(epsilon, 1, axis=0)
        else:
            epsilon_tilde = epsilon[0, ...] + r
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
            (its < maxits),
            lambda x, y: jax.debug.print(
                "JAX forward solver converged after {:6d} of {:6d} iterations",
                x,
                y,
                ordered=True,
            ),
            lambda x, y: None,
            its,
            maxits,
        )
        jax.lax.cond(
            (its >= maxits) & jnp.logical_not(dynamic_stopping),
            lambda x, y: jax.debug.print(
                "JAX forward stopped after {:6d} of {:6d} iterations",
                x,
                y,
                ordered=True,
            ),
            lambda x, y: None,
            its,
            maxits,
        )
        jax.debug.print(
            "E = ||div(sigma)||/||sigma|| = {:8.2e} E/E_0 = {:8.2e}",
            rel_error,
            rel_error / (rel_error_0 + 1.0e-20),
            ordered=True,
        )
    if dynamic_stopping:
        jax.lax.cond(
            its >= maxits,
            lambda x: jax.debug.print(
                "JAX forward solver failed to converge after {:6d} iterations",
                x,
                ordered=True,
            ),
            lambda x: None,
            maxits,
        )

    return epsilon[0, ...], sigma


@jax.jit(
    static_argnames=[
        "sigma_vjp",
        "grid_spec",
        "maxits",
        "depth",
        "dynamic_stopping",
        "verbose",
    ]
)
def _lippmann_schwinger_adjoint_jax(
    sigma_vjp: Callable[
        [jax.Array],
        tuple[jax.Array, PyTree],
    ],
    f_rhs: jax.Array,
    ref_params: dict[str, float],
    grid_spec: GridSpec,
    tol: float,
    maxits: int,
    depth: int,
    dynamic_stopping: bool,
    verbose: int,
) -> tuple[jax.Array, int]:
    """Lippmann Schwinger iteration for adjoint equation

    Computational routine which should not be called directly.

    The stress-strain relationship is described by the function ``compute_sigma()``
    which is of the form::

        def compute_sigma(epsilon, params):
            # compute stress sigma from strain epsilon given material parameters params
            return sigma

    as discussed in :py:mod:`jaxmaterials.solver.hooke`. Here params are the material parameters,
    such as the Lame coefficients for an isotropic material. This method gets passed
    :math:`\\delta \\sigma/\\delta \\epsilon`, derived from ``compute_sigma()`` which is derived from
    ``compute_sigma()`` with
        `jax.vjp https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html`_
    `jax.vjp https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html`_ of ``compute_sigma``.

    Parameters
    ==========
    sigma_vjp :
        vector-Jacobian product function :math:`\\delta \\sigma/\\delta \\epsilon`
    f_rhs :
        right hand side in adjoint equation, array of shape ``(6,nx,ny,nz)
    ref_params :
        Lame coefficients of isotropic reference material, dictionary of the form
        ``{"lambda":lambda, "mu":mu}``
    grid_spec :
        specification of computational grid
    tol :
        relative tolerance on normalised stress divergence to check convergence
    maxits :
        maximum number of iterations
    depth :
        depth of Anderson acceleration (depth=0: no acceleration)
    dynamic_stopping :
            stop based on ``rtol`` and ``atol``? If ``False``, stop after ``maxits`` iterations

    Returns
    =======
    tuple[jax.Array, int]
        Adjoint state :math:`\\Lambda` and number of iterations
    """
    rtol = tol
    atol = 1.0e-20
    # Fourier vectors
    dtype = f_rhs.dtype
    xizero = get_xizero(grid_spec, dtype=dtype)
    voigt_weights = jnp.array([1, 1, 1, 2, 2, 2], dtype=dtype)
    voigt_weights_bcast = voigt_weights[:, None, None, None]
    # storage for solution and residual, arrays of shape (d+1,6,Nx,Ny,Nz)
    Lambda = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz),
        dtype=dtype,
    )
    Lambda = Lambda.at[0, ...].set(f_rhs)
    residual = jnp.zeros(
        (depth + 1, 6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype
    )
    # Anderson matrix and vectors
    A_anderson = jnp.eye(depth + 1, dtype=jnp.float64)
    u_rhs = jnp.zeros(depth + 1, dtype=dtype)
    increment_nrm = jnp.linalg.norm(Lambda)
    if verbose > 1:
        jax.debug.print("==== JAX adjoint solve ====", ordered=True)
        jax.debug.print(
            "  iteration  E = ||delta(Lambda)||  ||delta(Lambda)||/||Lambda||",
            ordered=True,
        )

    def exit_condition(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int],
    ) -> jax.Array:
        """Check exit condition

        This method checks whether the relative change
        :math:`\\|\\Lambda^{(i)}-\\Lambda^{(i-1)}\\|_2` is sufficiently small or the maximum number
        of iterations has been reached.

        Parameters
        ==========
        state :
            current iteration state ``Lambda, residual, A_anderson, u_rhs, increment_nrm, its``

        Returns
        =======
        jax.Array
            ``True`` if :math:`\\|\\Lambda^{(i)}-\\Lambda^{(i-1)}\\|_2 < \\max\\{atol, rtol\\cdot \\|\\Lambda^{(i)}\\|_2\\}` or :math:`its > maxits`
        """
        Lambda = state[0]
        increment_nrm = state[4]
        its = state[5]
        nrm = jnp.linalg.norm(Lambda[0, ...])
        if verbose > 1:
            jax.debug.print(
                "  {:6d}  {:8.2e}  {:8.2e}",
                its,
                increment_nrm,
                increment_nrm / (nrm + 1.0e-20),
                ordered=True,
            )

        return (increment_nrm > atol) & (increment_nrm > rtol * nrm) & (its < maxits)

    def loop_body(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int]:
        """Update state according to update rule

        Parameters
        ==========
        state : dict
            current iteration state ``Lambda, residual, A_anderson, u_rhs, increment_nrm, its``
        """
        Lambda, residual, A_anderson, u_rhs, increment_nrm, its = state
        Lambda_hat = jnp.fft.fftn(Lambda[0, ...], axes=(-3, -2, -1))
        Theta_hat = fourier_solve_isotropic(Lambda_hat, xizero, ref_params)
        Theta = jnp.real(jnp.fft.ifftn(Theta_hat, axes=(-3, -2, -1)))
        # Convert Voigt-dual Theta to Euclidean cotangent for vjp, then map back.
        dSigma_depsilon, _ = sigma_vjp(voigt_weights_bcast * Theta)
        dSigma_depsilon = dSigma_depsilon / voigt_weights_bcast
        r = (
            Lambda[0, ...]
            - f_rhs
            - dSigma_depsilon
            + compute_sigma_isotropic(Theta, ref_params)
        )
        if depth > 0:
            residual = jnp.roll(residual, 1, axis=0)
            residual = residual.at[0, ...].set(r)
            A_anderson = jnp.roll(A_anderson, (1, 1), axis=(0, 1))
            dotproduct_scaling = jnp.array(
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=jnp.float64
            )
            A_anderson = A_anderson.at[0, :].set(
                jnp.einsum(
                    "aijk,saijk,a->s",
                    jnp.astype(r, jnp.float64),
                    jnp.astype(residual, jnp.float64),
                    dotproduct_scaling,
                )
            )
            A_anderson = A_anderson.at[:, 0].set(A_anderson[0, :])
            u_rhs = jnp.roll(u_rhs, 1)
            u_rhs = u_rhs.at[0].set(1)
            v = jnp.linalg.solve(A_anderson, u_rhs)
            alpha = v / jnp.dot(v, u_rhs)
            Lambda_tilde = jnp.einsum("s,saijk", alpha, Lambda - residual)
            Lambda = jnp.roll(Lambda, 1, axis=0)
        else:
            Lambda_tilde = Lambda[0, ...] - r
        Lambda = Lambda.at[0, ...].set(Lambda_tilde)

        its += 1
        increment_nrm = jnp.linalg.norm(r)
        return Lambda, residual, A_anderson, u_rhs, increment_nrm, its

    if dynamic_stopping:
        loop_result = jax.lax.while_loop(
            exit_condition,
            loop_body,
            init_val=(Lambda, residual, A_anderson, u_rhs, increment_nrm, 0),
        )
    else:
        loop_result = jax.lax.fori_loop(
            0,
            maxits,
            lambda _, state: loop_body(state),
            init_val=(Lambda, residual, A_anderson, u_rhs, increment_nrm, 0),
        )

    Lambda = loop_result[0][0]
    increment_nrm, its = loop_result[-2:]
    if verbose > 0:
        nrm = jnp.linalg.norm(Lambda)
        jax.lax.cond(
            (its < maxits),
            lambda x, y: jax.debug.print(
                "JAX adjoint solver converged after {:6d} of {:6d} iterations",
                x,
                y,
                ordered=True,
            ),
            lambda x, y: None,
            its,
            maxits,
        )
        jax.lax.cond(
            (its >= maxits) & jnp.logical_not(dynamic_stopping),
            lambda x, y: jax.debug.print(
                "JAX adjoint solver stopped after {:6d} of {:6d} iterations",
                x,
                y,
                ordered=True,
            ),
            lambda x, y: None,
            its,
            maxits,
        )
        jax.debug.print(
            "||delta(Lambda)|| = {:8.2e} ||delta(Lambda)||/||Lambda|| = {:8.2e}",
            increment_nrm,
            increment_nrm / (nrm + 1.0e-20),
            ordered=True,
        )
    if dynamic_stopping:
        jax.lax.cond(
            its >= maxits,
            lambda x: jax.debug.print(
                "JAX adjoint solver failed to converge after {:6d} iterations",
                x,
                ordered=True,
            ),
            lambda x: None,
            maxits,
        )

    return Lambda, its


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "compute_sigma",
        "grid_spec",
        "tol",
        "maxits",
        "depth",
        "dynamic_stopping",
        "verbose",
    ),
)
def solve(
    compute_sigma: Callable[[jax.Array, PyTree], jax.Array],
    params: PyTree,
    epsilon_bar: jax.Array,
    delta_epsilon_initial: jax.Array,
    ref_params: dict[str, float],
    grid_spec: GridSpec,
    tol: float,
    maxits: int,
    depth: int,
    dynamic_stopping: bool,
    verbose: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Reverse mode differentiable implementation of the forward solve in :py:func:`_lippmann_schwinger_jax`

    Parameters
    ==========
    compute_sigma :
        function which describes the stress-strain relationship, see :py:mod:`jaxmaterials.solver.hooke`
    params :
        material parameters which are passed on to ``compute_sigma()``
    epsilon_bar :
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    delta_epsilon_initial :
        initial strain perturbation :math:`\\delta\\epsilon`, which needs to average to zero. :math:`\\epsilon` is initialised to :math:`\\overline{\\epsilon}+\\delta\\epsilon`
    ref_params :
        Lame coefficients of isotropic reference material, dictionary of the form
        ``{"lambda":lambda, "mu":mu}`` where ``lambda``
    grid_spec :
        specification of computational grid
    tol :
        absolute tolerance on normalised stress divergence to check convergence
    maxits :
        maximum number of iterations
    depth :
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    dynamic_stopping :
        stop based on ``rtol`` and ``atol``? If ``False``, stop after exactly ``maxits`` iterations
    verbose :
        verbosity level

    Returns
    =======
    tuple[jax.Array, jax.Array]
        Strain :math:`\\epsilon` and strain :math:`\\sigma`
    """
    epsilon, sigma = _lippmann_schwinger_jax(
        compute_sigma,
        params,
        epsilon_bar,
        delta_epsilon_initial,
        ref_params,
        grid_spec,
        tol=tol,
        maxits=maxits,
        depth=depth,
        dynamic_stopping=dynamic_stopping,
        verbose=verbose,
    )
    return epsilon, sigma


def _solve_fwd(
    compute_sigma: Callable[[jax.Array, PyTree], jax.Array],
    params: PyTree,
    epsilon_bar: jax.Array,
    delta_epsilon_initial: jax.Array,
    ref_params: dict[str, float],
    grid_spec: GridSpec,
    tol: float,
    maxits: int,
    depth: int,
    dynamic_stopping: bool,
    verbose: int = 0,
) -> tuple[
    tuple[jax.Array, jax.Array],
    tuple[PyTree, jax.Array, jax.Array, jax.Array, dict[str, float]],
]:
    """Wrapped for forward solve

    Parameters
    ==========
    compute_sigma :
        function which describes the stress-strain relationship, see :py:mod:`jaxmaterials.solver.hooke`
    params :
        material parameters which are passed on to ``compute_sigma()``
    epsilon_bar :
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    delta_epsilon_initial :
        initial strain perturbation :math:`\\delta\\epsilon`, which needs to average to zero. :math:`\\epsilon` is initialised to :math:`\\overline{\\epsilon}+\\delta\\epsilon`
    ref_params :
        Lame coefficients of isotropic reference material, dictionary of the form
        ``{"lambda":lambda, "mu":mu}`` where ``lambda`` and ``mu`` are of shape ``(nx,ny,nz)``
    grid_spec :
        specification of computational grid
    tol :
        absolute tolerance on normalised stress divergence to check convergence
    maxits :
        maximum number of iterations
    depth :
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    dynamic_stopping :
        stop based on ``rtol`` and ``atol``? If ``False``, stop after exactly ``maxits`` iterations
    verbose :
        verbosity level

    Returns
    =======
    tuple[tuple[jax.Array,jax.Array], tuple[PyTree, jax.Array, jax.Array, jax.Array, dict[str, float]]]
        Tuple containing strain :math:`\\epsilon` and strain :math:`\\sigma` and information that is used by backward solve
    """
    out = _lippmann_schwinger_jax(
        compute_sigma,
        params,
        epsilon_bar,
        delta_epsilon_initial,
        ref_params,
        grid_spec,
        tol=tol,
        maxits=maxits,
        depth=depth,
        dynamic_stopping=dynamic_stopping,
        verbose=verbose,
    )
    epsilon, sigma = out
    return out, (params, epsilon, sigma, delta_epsilon_initial, ref_params)


def _solve_bwd(
    compute_sigma: Callable[[jax.Array, PyTree], jax.Array],
    grid_spec: GridSpec,
    tol: float,
    maxits: int,
    depth: int,
    dynamic_stopping: bool,
    verbose: int,
    res: tuple[PyTree, jax.Array, jax.Array, jax.Array, dict[str, float]],
    gradients: tuple[jax.Array, jax.Array],
) -> tuple[PyTree, jax.Array, jax.Array, PyTree]:
    """Backward solve based on the adjoint method

    Returns gradients with respect to material parameters and epsilon_bar

    Parameters
    ==========
    compute_sigma :
        function which describes the stress-strain relationship, see :py:mod:`jaxmaterials.solver.hooke`
    grid_spec :
        specification of computational grid
    tol : float
        absolute tolerance on normalised stress divergence to check convergence
    maxits :
        maximum number of iterations
    depth :
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    dynamic_stopping :
        stop based on ``rtol`` and ``atol``? If ``False``, stop after exactly ``maxits`` iterations
    verbose :
        verbosity level
    res :
        results object returned by :py:func:`solve_fwd()`
    gradients :
        Riesz-representer of input gradients

    Returns
    =======
    tuple[PyTree, jax.Array, jax.Array, PyTree]
        Gradients :math:`\\delta/\\delta\\Theta`, :math:`\\delta/\\delta\\overline{\\epsilion}` and :math:`\\delta/\\delta\\Theta_{rfe}` where :math:`\\Theta` and :math:`\\Theta_{ref}` are the parameters and reference parameters respectively; the latter are set to zero.
    """
    params, epsilon, _, delta_epsilon_initial, ref_params = res
    dtype = epsilon.dtype
    xizero = get_xizero(grid_spec, dtype=dtype)
    # Incoming gradients are dual vectors with respect to
    # the *Euclidean* inner product. They need be converted
    # to dual vectors with respect to weighted dot-product
    # which arises from the use of Voigt notation, namely
    #
    #   <a,b>_V = a_0*b_0 + a_1*b_1 + a_2*b_2
    #           + 2 * ( a_3*b_3 + a_4*b_4 + a_5*b_5 )
    g_epsilon, g_sigma = gradients
    voigt_weights = jnp.array([1, 1, 1, 2, 2, 2], dtype=dtype)
    voigt_weights_bcast = voigt_weights[:, None, None, None]
    # solve adjoint equation
    _, sigma_vjp = jax.vjp(compute_sigma, epsilon, params)
    f_rhs = -(g_epsilon + sigma_vjp(g_sigma)[0]) / voigt_weights_bcast
    Lambda, _ = _lippmann_schwinger_adjoint_jax(
        sigma_vjp,
        f_rhs,
        ref_params,
        grid_spec,
        tol=tol,
        maxits=maxits,
        depth=depth,
        dynamic_stopping=dynamic_stopping,
        verbose=verbose,
    )

    Lambda_hat = jnp.fft.fftn(Lambda, axes=(-3, -2, -1))
    Theta_hat = fourier_solve_isotropic(Lambda_hat, xizero, ref_params)
    S_star = g_sigma - voigt_weights_bcast * jnp.real(
        jnp.fft.ifftn(Theta_hat, axes=(-3, -2, -1))
    )
    # Convert back to dual vector with respect to Euclidean inner product.
    g_epsilon_bar = -voigt_weights * jnp.sum(Lambda, axis=(1, 2, 3))
    # Derivative with respect to parameters
    g_params = sigma_vjp(S_star)[1]
    g_delta_epsilon_initial = jax.tree.map(jnp.zeros_like, delta_epsilon_initial)
    g_ref_params = jax.tree.map(jnp.zeros_like, ref_params)
    return g_params, g_epsilon_bar, g_delta_epsilon_initial, g_ref_params


solve.defvjp(_solve_fwd, _solve_bwd)
