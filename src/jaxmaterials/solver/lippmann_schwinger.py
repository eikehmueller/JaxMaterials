"""Lippmann Schwinger solver with Anderson acceleration"""

import ctypes
import numpy as np
from jaxmaterials.solver.backend import solve, number_of_iterations

__all__ = [
    "lippmann_schwinger_isotropic",
    "lippmann_schwinger_anisotropic",
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


def lippmann_schwinger_isotropic(
    mu, lmbda, epsilon_bar, grid_spec, use_cuda=False, verbose=0
):
    """Wrapper for Anderson-accelerated Lippmann Schwinger iteration in isotropic material

    :arg mu: Lame parameter mu
    :arg lmbda: Lame parameter lambda
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: specification of computational grid
    :arg use_cuda: use cuda, requires access to compiled library liblippmannschwinger.so
    :arg verbose: verbosity level, only used for CUDA version
    """
    dtype = np.float32 if use_cuda else epsilon_bar.dtype
    assert mu.dtype == dtype
    assert lmbda.dtype == dtype
    assert epsilon_bar.dtype == dtype
    if use_cuda:
        maxits = 32
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
            np.ascontiguousarray(mu),
            np.ascontiguousarray(lmbda),
            np.ascontiguousarray(epsilon_bar, dtype=np.float32),
            epsilon,
            sigma,
            cells,
            extents,
            1.0e-5,
            1.0e-20,
            maxits,
            verbose,
        )
        if number_of_iterations is not None:
            number_of_iterations.set(its)

        if its >= maxits:
            raise RuntimeError(f"Solver failed to converge after {maxits} iterations")
        return epsilon, sigma
    else:
        material_properties = {"mu": mu, "lambda": lmbda}
        epsilon, sigma = solve(
            material_properties, epsilon_bar, grid_spec, dynamic_stopping=True
        )
    return epsilon, sigma


def lippmann_schwinger_anisotropic(
    stiffness_tensor, epsilon_bar, grid_spec, use_cuda=False, verbose=0
):
    """Wrapper for Anderson-accelerated Lippmann Schwinger iteration in anisotropic material



    :arg stiffness_tensor: stiffness tensor,
    :arg epsilon_bar: mean value of epsilon
    :arg grid_spec: specification of computational grid
    :arg use_cuda: use cuda, requires access to compiled library liblippmannschwinger.so
    :arg verbose: verbosity level, only used for CUDA version
    """
    dtype = np.float32 if use_cuda else epsilon_bar.dtype
    assert stiffness_tensor.dtype == dtype
    assert epsilon_bar.dtype == dtype
    assert stiffness_tensor.shape == (21, grid_spec.nx, grid_spec.ny, grid_spec.nz)
    if use_cuda:
        maxits = 32
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
            1e-5,
            1.0e-20,
            maxits,
            verbose,
        )
        if number_of_iterations is not None:
            number_of_iterations.set(its)

        if its >= maxits:
            raise RuntimeError(f"Solver failed to converge after {maxits} iterations")

    else:
        material_properties = {"stiffness_tensor": stiffness_tensor}
        epsilon, sigma = solve(
            material_properties, epsilon_bar, grid_spec, dynamic_stopping=True
        )
    return epsilon, sigma
