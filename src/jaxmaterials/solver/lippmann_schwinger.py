"""Lippmann Schwinger solver with Anderson acceleration

Solvers are implemented for three different setups:

1. a general, user defined stress-strain relationship of the form
    :math:`\\sigma=\\Sigma(\\epsilon|\\theta)`
2. anisotropic materials for which :math:`\\sigma=C\\epsilon` with a general spatially
    varying symmetric elasticity tensor :math:`C=C(x)`
3. isotropic materials for which :math:`\\sigma=C\\epsilon` with a spatially varying
    elasticity tensor :math:`C=C(x)` where :math:`C_{ijk\\ell}(x) = \\lambda(x) \\delta_{ij}\\delta_{k\\ell} + \\mu(x) (\\delta_{ik}\\delta_{j\\ell} + \\delta_{i\\ell}\\delta_{jk})`

"""

from collections.abc import Callable
from typing import Any, TypeAlias
import warnings
import ctypes
import numpy as np
import jax
from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.hooke import compute_sigma_isotropic, compute_sigma_anisotropic
from jaxmaterials.solver._backend import solve

__all__ = [
    "lippmann_schwinger",
    "lippmann_schwinger_anisotropic",
    "lippmann_schwinger_isotropic",
]

PyTree: TypeAlias = Any


class CUDAUnavailableError(RuntimeError):
    """Specialised exception to signal that CUDA is unavailable"""

    pass


def _load_cuda_library() -> ctypes.CDLL:
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


def _resolve_cuda_symbol(lib: ctypes.CDLL, names: list[str]) -> Any:
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


def _expand_delta_epsilon_initial(
    epsilon_bar: jax.Array, delta_epsilon_initial: jax.Array | None
):
    """Proceess :math:`\\delta{\\epsilon}`

    If no value is given (i.e. ``delta_epsilon_initial`` is ``None``), create a
    zero field. Otherwise, verify that :math:`\\delta{\\epsilon}` indeed integrates to
    zero.

    Parameters
    ==========
    epsilon_bar :
        Average strain :math:`\\overline{\\varepsilon}`
    delta_epsilon_initial :
        Correction :math:`\\delta{\\epsilon}` to initial value of :math:`\\varepsilon`

    Returns
    =======
    jax.Array
        Zero array if ``delta_epsilon_initial`` is ``None``, ``delta_epsilon_initial`` otherwise
    """
    dtype = epsilon_bar.dtype
    if delta_epsilon_initial is None:
        _delta_epsilon_initial = jnp.zeros(shape=(6, 1, 1, 1), dtype=dtype)
    else:
        _delta_epsilon_initial = jnp.astype(delta_epsilon_initial, dtype)
        delta = 1.0e-12 if np.dtype(dtype) == np.float32 else 1.0e-6
        if (
            jnp.linalg.norm(jnp.average(_delta_epsilon_initial, axis=(1, 2, 3)))
            / jnp.linalg.norm(epsilon_bar)
            > delta
        ):
            raise RuntimeError(
                "|| <delta(epsilon)> || / || bar(epsilon) || > tolerance"
            )
    return _delta_epsilon_initial


def lippmann_schwinger(
    compute_sigma: Callable[[jax.Array, PyTree], jax.Array],
    params: PyTree,
    epsilon_bar: jax.Array,
    ref_params: dict[str, float],
    grid_spec: GridSpec,
    delta_epsilon_initial: jax.Array | None = None,
    tol: float = 1.0e-5,
    maxits: int = 1000,
    depth: int = 0,
    verbose: int = 0,
) -> tuple[jax.Array, jax.Array]:
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
    compute_sigma :
        function :math:`\\sigma=\\Sigma(\\varepsilon|\\theta)` which describes the stress-strain relationship, see :py:mod:`jaxmaterials.solver.hooke`
    params :
        material parameters which are passed on to ``compute_sigma()``
    epsilon_bar :
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    delta_epsilon_initial :
        initial strain perturbation :math:`\\delta\\epsilon`, which needs to average to zero. :math:`\\epsilon` is initialised to :math:`\\overline{\\epsilon}+\\delta\\epsilon`
    ref_params :
        Lame coefficients of isotropic reference material, dictionary of the form
        ``{"lambda":lambda_ref, "mu":mu_ref}``
    grid_spec :
        specification of computational grid
    tol :
        absolute tolerance on normalised stress divergence to check convergence
    maxits :
        maximum number of iterations
    depth :
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    verbose :
        verbosity level

    Returns
    =======
    tuple[jax.Array,jax.Array]
        Strain :math:`\\epsilon` and stress :math:`\\sigma`
    """
    assert depth >= 0
    assert maxits > 0
    assert tol > 0

    epsilon, sigma = solve(
        compute_sigma,
        params,
        epsilon_bar,
        _expand_delta_epsilon_initial(epsilon_bar, delta_epsilon_initial),
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
    params: dict[str, jax.Array],
    epsilon_bar: jax.Array,
    grid_spec: GridSpec,
    delta_epsilon_initial: jax.Array | None = None,
    tol: float = 1.0e-5,
    maxits: int = 1000,
    depth: int = 0,
    use_cuda: bool = False,
    verbose: int = 0,
) -> tuple[jax.Array, jax.Array]:
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
    params :
        dictionary ``{"lambda":lambda, "mu":mu}`` with Lame coefficients :math:`\\mu` and
         :math:`\\lambda` which are arrays of shape ``(nx,ny,nz)``
    epsilon_bar :
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    grid_spec :
        specification of computational grid
    delta_epsilon_initial :
        initial strain perturbation :math:`\\delta\\epsilon`, which needs to average to zero. :math:`\\epsilon` is initialised to :math:`\\overline{\\epsilon}+\\delta\\epsilon`
    tol : 
        absolute tolerance on normalised stress divergence to check convergence
    maxits : 
        maximum number of iterations
    depth : 
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    use_cuda :
        use CUDA implementation instead of JAX? Onky forward pass is implemented in this case
    verbose : 
        verbosity level

    Returns
    =======
    tuple[jax.Array, jax.Array]
        Strain :math:`\\epsilon` and stress :math:`\\sigma`
    """
    dtype = np.float32 if use_cuda else epsilon_bar.dtype
    assert params["lambda"].dtype == dtype
    assert params["mu"].dtype == dtype
    assert epsilon_bar.dtype == dtype
    assert depth >= 0
    assert maxits > 0
    assert tol > 0
    _delta_epsilon_initial = _expand_delta_epsilon_initial(
        epsilon_bar, delta_epsilon_initial
    )
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
        epsilon_cuda = np.empty(
            (6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32
        )
        sigma_cuda = np.empty(
            (6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32
        )
        its = cuda_code(
            np.ascontiguousarray(params["mu"]),
            np.ascontiguousarray(params["lambda"]),
            np.ascontiguousarray(epsilon_bar, dtype=np.float32),
            np.ascontiguousarray(
                jnp.broadcast_to(_delta_epsilon_initial, shape=(6, *grid_spec.extents)),
                dtype=np.float32,
            ),
            epsilon_cuda,
            sigma_cuda,
            cells,
            extents,
            1.0e-20,
            tol,
            maxits,
            verbose,
        )

        if its >= maxits:
            raise RuntimeError(f"Solver failed to converge after {maxits} iterations")
        return jnp.asarray(epsilon_cuda), jnp.asarray(sigma_cuda)
    else:
        ref_params = {
            field: 1 / 2 * (np.min(params[field]) + np.max(params[field]))
            for field in params.keys()
        }
        epsilon_jax, sigma_jax = solve(
            compute_sigma_isotropic,
            params,
            epsilon_bar,
            _delta_epsilon_initial,
            jax.lax.stop_gradient(ref_params),
            grid_spec,
            tol=tol,
            depth=depth,
            maxits=maxits,
            dynamic_stopping=True,
            verbose=verbose,
        )
        return epsilon_jax, sigma_jax


def lippmann_schwinger_anisotropic(
    params: dict[str, jax.Array],
    epsilon_bar: jax.Array,
    ref_params: dict[str, float],
    grid_spec: GridSpec,
    delta_epsilon_initial: jax.Array | None = None,
    tol: float = 1.0e-5,
    maxits: int = 1000,
    depth: int = 0,
    use_cuda: bool = False,
    verbose: int = 0,
) -> tuple[jax.Array, jax.Array]:
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
    params :
        dictionary ``{"stiffness_tensor": stiffness_tensor}`` with material parameter
        :math:`C`, which is a tensor of shape ``(21,nx,ny,nz)``
    epsilon_bar :
        mean value :math:`\\overline{\\epsilon}` of strain :math:`\\epsilon`, array of shape ``(6,)``
    ref_params :
        Lame coefficients of isotropic reference material, dictionary of the form
        ``{"lambda":lambda_ref, "mu":mu_ref}`` where ``lambda_ref`` and ``mu_ref`` are of
        shape ``(nx,ny,nz)``
    grid_spec :
        specification of computational grid
    delta_epsilon_initial :
        initial strain perturbation :math:`\\delta\\epsilon`, which needs to average to zero. :math:`\\epsilon` is initialised to :math:`\\overline{\\epsilon}+\\delta\\epsilon`
    tol :
        absolute tolerance on normalised stress divergence to check convergence
    maxits :
        maximum number of iterations
    depth :
        depth of Anderson acceleration; depth=0 corresponds to no Anderson acceleration
    use_cuda :
        use CUDA implementation instead of JAX? Only forward pass is implemented in this case
    verbose :
        verbosity level

    Returns
    =======
    tuple[jax.Array, jax.Array]
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
    _delta_epsilon_initial = _expand_delta_epsilon_initial(
        epsilon_bar, delta_epsilon_initial
    )
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
        epsilon_cuda = np.empty(
            (6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32
        )
        sigma_cuda = np.empty(
            (6, grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=np.float32
        )
        its = cuda_code(
            stiffness,
            np.ascontiguousarray(epsilon_bar, dtype=np.float32),
            np.ascontiguousarray(
                jnp.broadcast_to(_delta_epsilon_initial, shape=(6, *grid_spec.extents)),
                dtype=np.float32,
            ),
            ref_params["lambda"],
            ref_params["mu"],
            epsilon_cuda,
            sigma_cuda,
            cells,
            extents,
            1.0e-20,
            tol,
            maxits,
            verbose,
        )

        if its >= maxits:
            raise RuntimeError(f"Solver failed to converge after {maxits} iterations")
        return jnp.asarray(epsilon_cuda), jnp.asarray(sigma_cuda)
    else:
        epsilon_jax, sigma_jax = solve(
            compute_sigma_anisotropic,
            params,
            epsilon_bar,
            _delta_epsilon_initial,
            jax.lax.stop_gradient(ref_params),
            grid_spec,
            tol=tol,
            maxits=maxits,
            depth=depth,
            dynamic_stopping=True,
            verbose=verbose,
        )
        return epsilon_jax, sigma_jax
