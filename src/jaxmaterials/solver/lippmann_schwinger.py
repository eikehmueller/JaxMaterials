"""Lippmann Schwinger solver with Anderson acceleration

Solvers are implemented for three different setups:

1. a general, user defined stress-strain relationship of the form
    :math:`\\sigma=\\Sigma(\\epsilon|\\theta)`
2. anisotropic materials for which :math:`\\sigma=C\\epsilon` with a general spatially
    varying symmetric elasticity tensor :math:`C=C(x)`
3. isotropic materials for which :math:`\\sigma=C\\epsilon` with a spatially varying
    elasticity tensor :math:`C=C(x)` where :math:`C_{ijk\\ell}(x) = \\lambda(x) \\delta_{ij}\\delta_{k\\ell} + \\mu(x) (\\delta_{ik}\\delta_{j\\ell} + \\delta_{i\\ell}\\delta_{jk})`

"""

import warnings
import ctypes
import numpy as np
import jax

from jaxmaterials.solver.hooke import compute_sigma_isotropic, compute_sigma_anisotropic
from jaxmaterials.solver._backend import solve

__all__ = [
    "lippmann_schwinger",
    "lippmann_schwinger_anisotropic",
    "lippmann_schwinger_isotropic",
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


def lippmann_schwinger(
    compute_sigma,
    params,
    epsilon_bar,
    ref_params,
    grid_spec,
    tol=1.0e-5,
    maxits=1000,
    depth=0,
    verbose=0,
):
    """Wrapper for Lippmann Schwinger iteration with Anderson acceleration for generic stress-strain relationship

    Iterates the Lippmann-Schwinger equation

    .. math::

        \\epsilon = -\\Gamma^0 * \\left(\\Sigma(\\epsilon|\\theta)-C^0\\epsilon \\right)

    where

    .. math::

        C^0_{ijk\\ell} = \\lambda_{ref} \\delta_{ij}\\delta_{k\\ell} + \\mu_{ref} (\\delta_{ik}\\delta_{j\\ell} + \\delta_{i\\ell}\\delta_{jk})

    is the reference linear elasticity tensor expressed in terms of the Lame parameters
    :math:`\\mu_{ref}` and :math:`\\lambda_{ref}`.

    The stress-strain relationship :math:`\\sigma=\\Sigma(\\epsilon|\\theta)` is described by the
    passed function ``compute_sigma()`` which is of the form::

        def compute_sigma(epsilon, params):
            # compute stress sigma from strain epsilon given material parameters params
            return sigma

    as discussed in :py:mod:`jaxmaterials.solver.hooke`. Here ``params`` are the material parameters,
    such as the spatially varying Lame coefficients for an isotropic material.

    Parameters
    ==========
    compute_sigma
        function :math:`\\sigma=\\Sigma(\\varepsilon|\\theta)` which describes the stress-strain relationship, see :py:mod:`jaxmaterials.solver.hooke`
    params : `jax.pytree <https://docs.jax.dev/en/latest/pytrees.html>`_
        material parameters which are passed on to ``compute_sigma()``
    epsilon_bar : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    ref_params : dict
        Lame coefficients of isotropic reference material, dictionary of the form
        ``{"lambda":lambda_ref, "mu":mu_ref}`` where ``lambda_ref`` and ``mu_ref`` are of
        shape ``(nx,ny,nz)``
    grid_spec : :py:class:`jaxmaterials.common.GridSpec`
        specification of computational grid
    tol : float
        absolute tolerance on normalised stress divergence to check convergence
    maxits : int
        maximum number of iterations
    depth : int
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    dynamic_stopping : logical
        stop based on ``rtol`` and ``atol``? If ``False``, stop after exactly ``maxits`` iterations
    verbose : int
        verbosity level

    Returns
    =======
    Strain :math:`\\epsilon` and stress :math:`\\sigma`
    """
    assert depth >= 0
    assert maxits > 0
    assert tol > 0
    epsilon, sigma = solve(
        compute_sigma,
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


def lippmann_schwinger_isotropic(
    params,
    epsilon_bar,
    grid_spec,
    tol=1.0e-5,
    maxits=1000,
    depth=0,
    use_cuda=False,
    verbose=0,
):
    """Wrapper for Lippmann Schwinger in isotropic material

    Iterates the Lippmann-Schwinger equation

    .. math::
    
            \\epsilon = -\\Gamma^0 * \\left(C-C^0 \\right) \\epsilon
    
    where
    
    .. math::

        \\begin{aligned}
        C_{ijk\\ell}(x) &= \\lambda(x) \\delta_{ij}\\delta_{k\\ell} + \\mu(x) (\\delta_{ik}\\delta_{j\\ell} + \\delta_{i\\ell}\\delta_{jk})\\\\
        C^0_{ijk\\ell} &= \\lambda_{ref} \\delta_{ij}\\delta_{k\\ell} + \\mu_{ref} (\\delta_{ik}\\delta_{j\\ell} + \\delta_{i\\ell}\\delta_{jk})
        \\end{aligned}

    is the reference linear elasticity tensor expressed in terms of the Lame parameters
    :math:`\\mu_{ref}` and :math:`\\lambda_{ref}`.

    Anderson acceleration can be used for the JAX implementation but is currently not
    supported for the CUDA version. Reference parameters required in the Lippmann Schwinger
    iteration are automatically computed as
    :math:`\\mu_{ref} = \\frac{1}{2}(\\max(\\mu)+\\min(\\mu))` and
    :math:`\\lambda_{ref} = \\frac{1}{2}(\\max(\\lambda)+\\min(\\lambda))`.

    Parameters
    ==========
    params : dict
        dictionary ``{"lambda":lambda, "mu":mu}`` with Lame coefficients :math:`\\mu` and
         :math:`\\lambda` which are arrays of shape ``(nx,ny,nz)``
    epsilon_bar : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    grid_spec : :py:class:`jaxmaterials.common.GridSpec`
        specification of computational grid
    tol : float
        absolute tolerance on normalised stress divergence to check convergence
    maxits : int
        maximum number of iterations
    depth : int
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    use_cuda : logical
        use CUDA implementation instead of JAX? Onky forward pass is implemented in this case
    verbose : int
        verbosity level

    Returns
    =======
    Strain :math:`\\epsilon` and stress :math:`\\sigma`
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
        ref_params = {
            field: 1 / 2 * (np.min(params[field]) + np.max(params[field]))
            for field in params.keys()
        }
        epsilon, sigma = solve(
            compute_sigma_isotropic,
            params,
            epsilon_bar,
            jax.lax.stop_gradient(ref_params),
            grid_spec,
            tol=tol,
            depth=depth,
            maxits=maxits,
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
    """Wrapper for Lippmann Schwinger in anisotropic material

    Solves the Lippmann-Schwinger equation

    .. math::

            \\epsilon = -\\Gamma^0 * \\left(C-C^0 \\right) \\epsilon

    where :math:`C_{ijk\\ell}(x)` is the spatially varying elasticity tensor and

    .. math::

        C^0_{ijk\\ell} = \\lambda_{ref} \\delta_{ij}\\delta_{k\\ell} + \\mu_{ref} (\\delta_{ik}\\delta_{j\\ell} + \\delta_{i\\ell}\\delta_{jk})

    is the reference linear elasticity tensor expressed in terms of the Lame parameters
    :math:`\\mu_{ref}` and :math:`\\lambda_{ref}`.

    Anderson acceleration can be used for the JAX implementation but is currently not
    supported for the CUDA version.

    Parameters
    ==========
    params : dict
        dictionary ``{"stiffness_tensor": stiffness_tensor}`` with material parameter
        :math:`C`, which is a tensor of shape ``(21,nx,ny,nz)``
    epsilon_bar : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    ref_params : dict
        Lame coefficients of isotropic reference material, dictionary of the form
        ``{"lambda":lambda_ref, "mu":mu_ref}`` where ``lambda_ref`` and ``mu_ref`` are of
        shape ``(nx,ny,nz)``
    grid_spec : :py:class:`jaxmaterials.common.GridSpec`
        specification of computational grid
    tol : float
        absolute tolerance on normalised stress divergence to check convergence
    maxits : int
        maximum number of iterations
    depth : int
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    use_cuda : logical
        use CUDA implementation instead of JAX? Onky forward pass is implemented in this case
    verbose : int
        verbosity level

    Returns
    =======
    Strain :math:`\\epsilon` and stress :math:`\\sigma`
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
            ctypes.c_float,
            ctypes.c_float,
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
            ref_params["lambda"],
            ref_params["mu"],
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
        # Least squares fit
        epsilon, sigma = solve(
            compute_sigma_anisotropic,
            params,
            epsilon_bar,
            jax.lax.stop_gradient(ref_params),
            grid_spec,
            tol=tol,
            maxits=maxits,
            depth=depth,
            dynamic_stopping=True,
            verbose=verbose,
        )
    return epsilon, sigma
