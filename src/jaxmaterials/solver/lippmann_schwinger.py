"""Lippmann Schwinger solver with Anderson acceleration"""

import ctypes
import numpy as np
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
    "lippmann_schwinger_isotropic_jax",
    "lippmann_schwinger_isotropic_cuda",
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


# @jax.jit(
#    static_argnames=[
#        "isotropic",
#        "grid_spec",
#        "rtol",
#        "atol",
#        "depth",
#        "maxiter",
#        "dtype",
#    ]
# )
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


def lippmann_schwinger_isotropic_jax(
    mu,
    lmbda,
    epsilon_bar,
    grid_spec,
    rtol=1e-6,
    atol=1e-20,
    depth=0,
    maxiter=32,
    dtype=jnp.float32,
):
    """Lippmann Schwinger iteration linear elasticity in isotropic material

    :arg mu: Lame parameter mu
    :arg lmbda: Lame parameter lambda
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    return _lippmann_schwinger_jax(
        {"mu": mu, "lambda": lmbda},
        epsilon_bar,
        grid_spec,
        True,
        rtol,
        atol,
        depth,
        maxiter,
        dtype,
    )


def lippmann_schwinger_anisotropic_jax(
    stiffness_tensor,
    epsilon_bar,
    grid_spec,
    rtol=1e-6,
    atol=1e-20,
    depth=0,
    maxiter=32,
    dtype=jnp.float32,
):
    """Lippmann Schwinger iteration linear elasticity in anisotropic material

    :arg stiffness_tensor: stiffness tensor,
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    return _lippmann_schwinger_jax(
        {"stiffness_tensor": stiffness_tensor},
        epsilon_bar,
        grid_spec,
        False,
        rtol,
        atol,
        depth,
        maxiter,
        dtype,
    )


def lippmann_schwinger_isotropic_cuda(
    mu,
    lmbda,
    epsilon_bar,
    grid_spec,
    rtol=1e-6,
    atol=1.0e-20,
    maxiter=32,
    verbose=0,
):
    """Wrapper for CUDA Lippmann Schwinger solver

    Required access to compiled library liblippmannschwinger.so

    The dictionary material_properties is of the form {"lambda":lambda,"mu":mu} for an
    isotropic material.

    :arg mu: Lame parameter mu
    :arg lmbda: Lame parameter lambda
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg maxiter: maximal number of iterations
    :arg verbose: verbosity level
    """
    # Check that material is isotropic
    # Load cuda library
    try:
        lib = ctypes.CDLL("liblippmannschwinger.so")
    except Exception as exc:
        raise RuntimeError(
            "Unable to load cuda library liblippmannschwinger.so. Compile and check LD_LIBRARY_PATH."
        ) from exc
    cuda_code = lib.lippmann_schwinger_solve
    cuda_code.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    cuda_code.restype = ctypes.c_int
    cells = np.array([grid_spec.nx, grid_spec.ny, grid_spec.nz], dtype=np.int32)
    extents = np.array([grid_spec.Lx, grid_spec.Ly, grid_spec.Lz], dtype=np.float32)
    epsilon = np.empty((6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32)
    sigma = np.empty((6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32)
    iter = cuda_code(
        np.asarray(mu),
        np.asarray(lmbda),
        np.asarray(epsilon_bar, dtype=np.float32),
        np.asarray(epsilon),
        np.asarray(sigma),
        cells,
        extents,
        rtol,
        atol,
        maxiter,
        verbose,
    )
    if iter == maxiter:
        raise RuntimeError(f"Solver failed to converge after {maxiter} iterations")
    return (
        epsilon,
        sigma,
        iter,
    )
