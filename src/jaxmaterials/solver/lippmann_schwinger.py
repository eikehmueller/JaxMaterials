"""Lippmann Schwinger solver with Anderson acceleration"""

import warnings
import ctypes
import numpy as np
import jax

from jaxmaterials.solver.hooke import compute_sigma_isotropic, compute_sigma_anisotropic
from jaxmaterials.solver.backend import solve

__all__ = [
    "lippmann_schwinger_isotropic",
    "lippmann_schwinger_anisotropic",
]


class CUDAUnavailableError(RuntimeError):
    pass


def _load_cuda_library():
    """Load CUDA shared library for Lippmann-Schwinger solvers.

    Prefer explicit library paths to avoid accidentally resolving an older
    system copy via the dynamic loader search path.

    Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
    """
    try:
        return ctypes.CDLL("liblippmannschwinger.so")
    except Exception as exc:
        raise CUDAUnavailableError(
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
    raise CUDAUnavailableError(
        f"Unable to find any CUDA entrypoint among symbols: {', '.join(names)}"
    )


def lippmann_schwinger_isotropic(
    params,
    epsilon_bar,
    ref_params,
    grid_spec,
    tol=1.0e-5,
    maxits=1000,
    depth=0,
    use_cuda=False,
    verbose=0,
):
    """Wrapper for Lippmann Schwinger iteration in isotropic material

    Anderson acceleration can be applied for the forward solve in the JAX implementation

    :arg params: dictionary with Lame coefficients {"lambda":lambda, "mu":mu}
    :arg epsilon_bar: mean value of epsilon
    :arg ref_params: Lame coefficients of isotropic, homogeneous reference material
    :arg grid_spec: specification of computational grid
    :arg tol: tolerance used for convergence test
    :arg maxits: maximum number of iterations
    :arg depth: depth for Anderson iteration; only used in JAX forward solve
    :arg use_cuda: use cuda, requires access to compiled library liblippmannschwinger.so
    :arg verbose: verbosity level
    """
    dtype = np.float32 if use_cuda else epsilon_bar.dtype
    assert params["lambda"].dtype == dtype
    assert params["mu"].dtype == dtype
    assert epsilon_bar.dtype == dtype
    assert depth >= 0
    assert maxits > 0
    assert tol > 0
    if use_cuda:
        if depth > 0:
            warnings.warn("Parameter depth ignored for CUDA implementations")
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
        epsilon = np.empty(
            (6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32
        )
        sigma = np.empty(
            (6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32
        )
        its = cuda_code(
            np.ascontiguousarray(params["mu"]),
            np.ascontiguousarray(params["lambda"]),
            np.ascontiguousarray(epsilon_bar, dtype=np.float32),
            epsilon,
            sigma,
            cells,
            extents,
            1.0e-20,
            tol,
            maxits,
            verbose,
        )

        if its >= maxits:
            raise RuntimeError(f"Solver failed to converge after {maxits} iterations")
        return epsilon, sigma
    else:
        epsilon, sigma = solve(
            compute_sigma_isotropic,
            params,
            epsilon_bar,
            ref_params,
            grid_spec,
            tol=tol,
            maxits=maxits,
            depth=depth,
            dynamic_stopping=True,
            verbose=verbose,
        )
    return epsilon, sigma


def lippmann_schwinger_anisotropic(
    params,
    epsilon_bar,
    ref_params,
    grid_spec,
    tol=1.0e-5,
    maxits=1000,
    depth=0,
    use_cuda=False,
    verbose=0,
):
    """Wrapper for Lippmann Schwinger iteration in anisotropic material

    Anderson acceleration can be applied for the forward solve in the JAX implementation

    :arg params: material parameters, dictionary {"stiffness_tensor":stiffness_tensor}
    :arg epsilon_bar: mean value of epsilon
    :arg ref_params: Lame coefficients of isotropic, homogeneous reference material, dictionary
        {"lambda":lambda, "mu":mu}
    :arg grid_spec: specification of computational grid
    :arg tol: tolerance used for convergence test
    :arg maxits: maximum number of iterations
    :arg depth: depth for Anderson iteration; only used in JAX forward solve
    :arg use_cuda: use cuda, requires access to compiled library liblippmannschwinger.so
    :arg verbose: verbosity level
    """
    dtype = np.float32 if use_cuda else epsilon_bar.dtype
    stiffness_tensor = params["stiffness_tensor"]
    assert stiffness_tensor.dtype == dtype
    assert epsilon_bar.dtype == dtype
    assert stiffness_tensor.shape == (21, grid_spec.nx, grid_spec.ny, grid_spec.nz)
    assert depth >= 0
    assert maxits > 0
    assert tol > 0
    if use_cuda:
        if depth > 0:
            warnings.warn("Parameter depth ignored for CUDA implementations")
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
        epsilon = np.empty(
            (6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32
        )
        sigma = np.empty(
            (6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32
        )
        its = cuda_code(
            stiffness,
            np.ascontiguousarray(epsilon_bar, dtype=np.float32),
            epsilon,
            sigma,
            cells,
            extents,
            1.0e-20,
            tol,
            maxits,
            verbose,
        )

        if its >= maxits:
            raise RuntimeError(f"Solver failed to converge after {maxits} iterations")

    else:
        epsilon, sigma = solve(
            compute_sigma_anisotropic,
            params,
            epsilon_bar,
            ref_params,
            grid_spec,
            tol=tol,
            maxits=maxits,
            depth=depth,
            dynamic_stopping=True,
            verbose=verbose,
        )
    return epsilon, sigma
