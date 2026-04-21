"""Lippmann Schwinger solver with Anderson acceleration"""

import ctypes
import numpy as np
from jax import numpy as jnp
from jaxmaterials.solver.backend import (
    _lippmann_schwinger_jax,
    _lippmann_schwinger_adjoint_jax,
)

__all__ = [
    "lippmann_schwinger_isotropic_jax",
    "lippmann_schwinger_anisotropic_jax",
    "lippmann_schwinger_isotropic_cuda",
    "lippmann_schwinger_anisotropic_cuda",
]


def _load_cuda_library():
    """Load CUDA shared library for Lippmann-Schwinger solvers.

    Prefer explicit library paths to avoid accidentally resolving an older
    system copy via the dynamic loader search path.

    Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
    """
    try:
        return ctypes.CDLL("liblippmannschwinger.so")
    except Exception as exc:
        raise RuntimeError(
            "Unable to load CUDA library liblippmannschwinger.so"
        ) from exc


def _resolve_cuda_symbol(lib, names):
    """Resolve the first available symbol from a list of candidate names.

    Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
    """
    for name in names:
        symbol = getattr(lib, name, None)
        if symbol is not None:
            return symbol
    raise RuntimeError(
        f"Unable to find any CUDA entrypoint among symbols: {', '.join(names)}"
    )


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
    """Wrapper for Anderson-accelerated Lippmann Schwinger iteration in isotropic material

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
    # Check data types
    assert mu.dtype == dtype
    assert lmbda.dtype == dtype
    assert epsilon_bar.dtype == dtype
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
    """Wrapper for Anderson-accelerated Lippmann Schwinger iteration in anisotropic material

    :arg stiffness_tensor: stiffness tensor,
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    assert stiffness_tensor.dtype == dtype
    assert epsilon_bar.dtype == dtype

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


def lippmann_schwinger_adjoint_isotropic_jax(
    mu,
    lmbda,
    f_rhs,
    grid_spec,
    rtol=1e-6,
    atol=1e-20,
    maxiter=32,
    dtype=jnp.float32,
):
    """Wrapper for adjoint Lippmann Schwinger in isotropic material

    :arg mu: Lame parameter mu
    :arg lmbda: Lame parameter lambda
    :arg f_rhs: right hand side field
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    # Check data types
    assert mu.dtype == dtype
    assert lmbda.dtype == dtype
    assert f_rhs.dtype == dtype
    return _lippmann_schwinger_adjoint_jax(
        {"mu": mu, "lambda": lmbda},
        f_rhs,
        grid_spec,
        True,
        rtol,
        atol,
        maxiter,
        dtype,
    )


def lippmann_schwinger_adjoint_anisotropic_jax(
    stiffness_tensor,
    f_rhs,
    grid_spec,
    rtol=1e-6,
    atol=1e-20,
    maxiter=32,
    dtype=jnp.float32,
):
    """Wrapper for adjoint Lippmann Schwinger in anisotropic material

    :arg stiffness_tensor: stiffness tensor,
    :arg r_rhs: right hand side function
    :arg grid_spec: grid specification as a namedtuple
    :arg rtol: relative tolerance on normalised stress divergence to check convergence
    :arg atol: absolute tolerance on normalised stress divergence to check convergence
    :arg depth: depth of Anderson acceleration
    :arg maxiter: maximal number of iterations
    :arg dtype: data type
    """
    assert stiffness_tensor.dtype == dtype
    assert f_rhs.dtype == dtype

    return _lippmann_schwinger_adjoint_jax(
        {"stiffness_tensor": stiffness_tensor},
        f_rhs,
        grid_spec,
        False,
        rtol,
        atol,
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
    :arg
    """
    # Check data types
    assert mu.dtype == np.float32
    assert lmbda.dtype == np.float32
    assert epsilon_bar.dtype == np.float32

    lib = _load_cuda_library()
    # Prefer new name, fall back to legacy symbol for backward compatibility.
    cuda_code = _resolve_cuda_symbol(
        lib,
        ["lippmann_schwinger_solve_isotropic", "lippmann_schwinger_solve"],
    )
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
        np.ascontiguousarray(mu),
        np.ascontiguousarray(lmbda),
        np.ascontiguousarray(epsilon_bar, dtype=np.float32),
        epsilon,
        sigma,
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


def lippmann_schwinger_anisotropic_cuda(
    stiffness_tensor,
    epsilon_bar,
    grid_spec,
    rtol=1e-6,
    atol=1.0e-20,
    maxiter=32,
    verbose=0,
):
    """Wrapper for CUDA Lippmann-Schwinger solver in anisotropic material.

    Required access to compiled library liblippmannschwinger.so.

    stiffness_tensor is assumed to have shape (21,nx,ny,nz).

    Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
    """
    assert stiffness_tensor.dtype == np.float32
    assert stiffness_tensor.shape == (21, grid_spec.nx, grid_spec.ny, grid_spec.nz)
    assert epsilon_bar.dtype == np.float32

    stiffness = np.ascontiguousarray(stiffness_tensor)
    lib = _load_cuda_library()
    cuda_code = _resolve_cuda_symbol(lib, ["lippmann_schwinger_solve_anisotropic"])
    cuda_code.argtypes = [
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
        stiffness,
        np.ascontiguousarray(epsilon_bar, dtype=np.float32),
        epsilon,
        sigma,
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
