"""Functionality for computations in Fourier space

See discussion in Section :ref:`sec:fourier` for details.
"""

from typing import cast

import jax
import numpy as np
from jax import numpy as jnp

from jaxmaterials.common import GridSpec

__all__ = [
    "fourier_solve_anisotropic",
    "fourier_solve_isotropic",
    "get_anisotropic_acoustic_tensor",
    "get_inverse_anisotropic_acoustic_tensor",
    "get_xi",
    "get_xizero",
]


def get_xi(grid_spec: GridSpec, dtype: np.typing.DTypeLike = np.float64) -> np.ndarray:
    """Construct un-normalised Fourier momentum vectors :math:`\\widetilde{\\boldsymbol{\\xi}}`

    Let :math:`\\boldsymbol{k} = (k_0,k_1,k_2)` with :math:`k_i = 0,1,...,N_i-1` be a three-dimensional Fourier index. The normalised momentum vector is :math:`\\xi_i = 2 \\pi k_i / N_i`, with :math:`0 \\le \\xi_i < 2\\pi`.

    For a given :math:`\\boldsymbol{k}` we then have that

    .. math::

        \\begin{aligned}
            \\widetilde{\\xi}_0 &= \\frac{2}{h_0} \\sin\\left(\\frac{\\xi_0}{2}\\right) \\cos\\left(\\frac{\\xi_1}{2}\\right) \\cos\\left(\\frac{\\xi_2}{2}\\right)\\\\
            \\widetilde{\\xi}_1 &= \\frac{2}{h_1} \\cos\\left(\\frac{\\xi_0}{2}\\right) \\sin\\left(\\frac{\\xi_1}{2}\\right) \\cos\\left(\\frac{\\xi_2}{2}\\right)\\\\
            \\widetilde{\\xi}_2 &= \\frac{2}{h_2} \\cos\\left(\\frac{\\xi_0}{2}\\right) \\cos\\left(\\frac{\\xi_1}{2}\\right) \\sin\\left(\\frac{\\xi_2}{2}\\right)
        \\end{aligned}

    Parameters
    ==========
    grid_spec :
            specification of computational grid
    
    dtype :
        data type

    Returns
    =======
    numpy.ndarray
        Tensor of shape ``(3,N_0,N_1,N_2)`` and type ``dtype`` with :math:`\\widetilde{\\boldsymbol{\\xi}}` for all Fourier modes.
    """
    # Normalised momentum vectors in all three spatial directions
    K = [
        2 * np.pi * np.arange(n) / n for n in (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    ]
    # Grid with normalised momentum vectors
    xi_grid = np.meshgrid(*K, indexing="ij")
    # Grid with tilde(xi)
    xi = np.stack(
        [
            2
            * grid_spec.nx
            / grid_spec.Lx
            * np.sin(xi_grid[0] / 2)
            * np.cos(xi_grid[1] / 2)
            * np.cos(xi_grid[2] / 2),
            2
            * grid_spec.ny
            / grid_spec.Ly
            * np.cos(xi_grid[0] / 2)
            * np.sin(xi_grid[1] / 2)
            * np.cos(xi_grid[2] / 2),
            2
            * grid_spec.nz
            / grid_spec.Lz
            * np.cos(xi_grid[0] / 2)
            * np.cos(xi_grid[1] / 2)
            * np.sin(xi_grid[2] / 2),
        ]
    )
    return xi.astype(dtype)


def get_xizero(
    grid_spec: GridSpec, dtype: np.typing.DTypeLike = np.float64
) -> np.ndarray:
    """Construct the normalised Fourier momentum vectors :math:`\\mathring{\\widetilde{\\boldsymbol{\\xi}}}`

    Computes

    .. math::

        \\mathring{\\widetilde{\\xi}}_i =
        \\begin{cases} 
            0 & \\text{if}\\; \\|\\widetilde{\\boldsymbol{\\xi}}\\|=0\\\\
            \\widetilde{\\xi}_i/\\|\\widetilde{\\boldsymbol{\\xi}}\\| & \\text{otherwise}
        \\end{cases}

    with :math:`\\widetilde{\\boldsymbol{\\xi}}` as defined in :py:func:`get_xi`.

    Parameters
    ==========
    grid_spec :
            specification of computational grid
    
    dtype :
        data type

    Returns
    =======
    numpy.ndarray
        Tensor of shape ``(3,N_0,N_1,N_2)`` and type ``dtype`` with :math:`\\mathring{\\widetilde{\\boldsymbol{\\xi}}}` for all Fourier modes.
    """
    xi_tilde = get_xi(grid_spec, dtype=dtype)
    # Normalise tilde(xi) to obtain xi^0
    xi_nrm = np.linalg.norm(xi_tilde, axis=0)
    tolerance = 1.0e-12 if np.dtype(dtype) == np.float64 else 1.0e-6
    xi_nrm[xi_nrm < tolerance] = 1  # avoid division by zero
    return cast(np.ndarray, (xi_tilde / xi_nrm).astype(dtype))


def get_anisotropic_acoustic_tensor(
    xizero: np.ndarray, stiffness_tensor0: np.ndarray
) -> np.ndarray:
    """Acoustic tensor :math:`K^0` for a homogeneous anisotropic reference material

    Assemble the :math:`3\\times3` acoustic tensor :math:`K^0` for a homogeneous isotropic reference material characterised by the stiffness tensor :math:`C^0` as described in :ref:`appendix_Gamma0_anisotropic`.

    Parameters
    ==========
    xizero :
        Normalised Fourier momentum vectors :math:`\\mathring{\\widetilde{\\boldsymbol{\\xi}}}` computed with :py:func:`get_xizero`
    stiffness_tensor0 :
        Stiffness tensor :math:`C^0`, vector with 21 components

    Returns
    =======
    numpy.ndarray
        Acoustic tensor :math:`K^0` with shape ``(3,3,Nx,Ny,Nz)`` for all Fourier modes
    """
    return np.stack(
        [
            np.stack(
                [
                    stiffness_tensor0[0] * xizero[0] ** 2
                    + stiffness_tensor0[3] * xizero[1] ** 2
                    + stiffness_tensor0[4] * xizero[2] ** 2
                    + 2 * stiffness_tensor0[9] * xizero[0] * xizero[1]
                    + 2 * stiffness_tensor0[10] * xizero[0] * xizero[2]
                    + 2 * stiffness_tensor0[18] * xizero[1] * xizero[2],
                    stiffness_tensor0[3] * xizero[0] * xizero[1]
                    + stiffness_tensor0[6] * xizero[0] * xizero[1]
                    + stiffness_tensor0[9] * xizero[0] ** 2
                    + stiffness_tensor0[11] * xizero[0] * xizero[2]
                    + stiffness_tensor0[12] * xizero[1] ** 2
                    + stiffness_tensor0[13] * xizero[1] * xizero[2]
                    + stiffness_tensor0[18] * xizero[0] * xizero[2]
                    + stiffness_tensor0[19] * xizero[1] * xizero[2]
                    + stiffness_tensor0[20] * xizero[2] ** 2,
                    stiffness_tensor0[4] * xizero[0] * xizero[2]
                    + stiffness_tensor0[7] * xizero[0] * xizero[2]
                    + stiffness_tensor0[10] * xizero[0] ** 2
                    + stiffness_tensor0[11] * xizero[0] * xizero[1]
                    + stiffness_tensor0[15] * xizero[1] * xizero[2]
                    + stiffness_tensor0[16] * xizero[2] ** 2
                    + stiffness_tensor0[18] * xizero[0] * xizero[1]
                    + stiffness_tensor0[19] * xizero[1] ** 2
                    + stiffness_tensor0[20] * xizero[1] * xizero[2],
                ]
            ),
            np.stack(
                [
                    stiffness_tensor0[3] * xizero[0] * xizero[1]
                    + stiffness_tensor0[6] * xizero[0] * xizero[1]
                    + stiffness_tensor0[9] * xizero[0] ** 2
                    + stiffness_tensor0[11] * xizero[0] * xizero[2]
                    + stiffness_tensor0[12] * xizero[1] ** 2
                    + stiffness_tensor0[13] * xizero[1] * xizero[2]
                    + stiffness_tensor0[18] * xizero[0] * xizero[2]
                    + stiffness_tensor0[19] * xizero[1] * xizero[2]
                    + stiffness_tensor0[20] * xizero[2] ** 2,
                    stiffness_tensor0[1] * xizero[1] ** 2
                    + stiffness_tensor0[3] * xizero[0] ** 2
                    + stiffness_tensor0[5] * xizero[2] ** 2
                    + 2 * stiffness_tensor0[12] * xizero[0] * xizero[1]
                    + 2 * stiffness_tensor0[14] * xizero[1] * xizero[2]
                    + 2 * stiffness_tensor0[19] * xizero[0] * xizero[2],
                    stiffness_tensor0[5] * xizero[1] * xizero[2]
                    + stiffness_tensor0[8] * xizero[1] * xizero[2]
                    + stiffness_tensor0[13] * xizero[0] * xizero[1]
                    + stiffness_tensor0[14] * xizero[1] ** 2
                    + stiffness_tensor0[15] * xizero[0] * xizero[2]
                    + stiffness_tensor0[17] * xizero[2] ** 2
                    + stiffness_tensor0[18] * xizero[0] ** 2
                    + stiffness_tensor0[19] * xizero[0] * xizero[1]
                    + stiffness_tensor0[20] * xizero[0] * xizero[2],
                ]
            ),
            np.stack(
                [
                    stiffness_tensor0[4] * xizero[0] * xizero[2]
                    + stiffness_tensor0[7] * xizero[0] * xizero[2]
                    + stiffness_tensor0[10] * xizero[0] ** 2
                    + stiffness_tensor0[11] * xizero[0] * xizero[1]
                    + stiffness_tensor0[15] * xizero[1] * xizero[2]
                    + stiffness_tensor0[16] * xizero[2] ** 2
                    + stiffness_tensor0[18] * xizero[0] * xizero[1]
                    + stiffness_tensor0[19] * xizero[1] ** 2
                    + stiffness_tensor0[20] * xizero[1] * xizero[2],
                    stiffness_tensor0[5] * xizero[1] * xizero[2]
                    + stiffness_tensor0[8] * xizero[1] * xizero[2]
                    + stiffness_tensor0[13] * xizero[0] * xizero[1]
                    + stiffness_tensor0[14] * xizero[1] ** 2
                    + stiffness_tensor0[15] * xizero[0] * xizero[2]
                    + stiffness_tensor0[17] * xizero[2] ** 2
                    + stiffness_tensor0[18] * xizero[0] ** 2
                    + stiffness_tensor0[19] * xizero[0] * xizero[1]
                    + stiffness_tensor0[20] * xizero[0] * xizero[2],
                    stiffness_tensor0[2] * xizero[2] ** 2
                    + stiffness_tensor0[4] * xizero[0] ** 2
                    + stiffness_tensor0[5] * xizero[1] ** 2
                    + 2 * stiffness_tensor0[16] * xizero[0] * xizero[2]
                    + 2 * stiffness_tensor0[17] * xizero[1] * xizero[2]
                    + 2 * stiffness_tensor0[20] * xizero[0] * xizero[1],
                ]
            ),
        ]
    )


def get_inverse_anisotropic_acoustic_tensor(
    xizero: np.ndarray, stiffness_tensor0: np.ndarray
) -> np.ndarray:
    """Inverse :math:`N^0 = (K^0)^{-1}` of the acoustic tensor :math:`K^0` for homogeneous anisotropic reference material

    Assemble the :math:`3\\times3` acoustic tensor :math:`K^0` for a homogeneous isotropic reference material characterised with :py:func:`get_anisotropic_acoustic_tensor` and invert it for each Fourier mode.

    Parameters
    ==========
    xizero : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        Normalised Fourier momentum vectors :math:`\\mathring{\\widetilde{\\boldsymbol{\\xi}}}` computed with :py:func:`get_xizero`
    stiffness_tensor0 : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        Stiffness tensor :math:`C^0`, vector with 21 components

    Returns
    =======
    numpy.ndarray
        Inverse acoustic tensor :math:`N^0 = (K^0)^{-1}` with shape ``(3,3,Nx,Ny,Nz)`` for all Fourier modes
    """
    K0 = get_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)
    K0_transpose = np.transpose(K0, axes=(2, 3, 4, 0, 1))
    xi_nrm = (xizero[0] ** 2 + xizero[1] ** 2 + xizero[2] ** 2) > 1.0e-8
    # Avoid inverting a singular zero-frequency block, which can produce NaN gradients.
    eye3 = np.eye(3, dtype=K0_transpose.dtype)
    K0_safe = np.where(xi_nrm[..., None, None], K0_transpose, eye3)
    N0_transpose = np.linalg.inv(K0_safe)
    return cast(np.ndarray, xi_nrm * np.transpose(N0_transpose, axes=(3, 4, 0, 1, 2)))


def fourier_solve_isotropic(
    tau_hat: jax.Array, xizero: np.ndarray, ref_params: dict[str, float]
) -> jax.Array:
    """Solve residual equation for homogeneous isotropic reference material in Fourier space

    For given :math:`\\widehat{\\tau}` compute

    .. math::

        \\hat{\\varepsilon}_{k\\ell} = -\\widehat{\\Gamma}^0_{k\\ell ij} \\widehat{\\tau}_{ij}

    for a homogeneous isotropic reference material which is characterised by the two Lame parameters :math:`\\lambda^0` and :math:`\\mu^0`. See Section :ref:`sec:fourier` for details.

    Parameters
    ==========
    tau_hat :
        right hand side :math:`\\widehat{\\tau}`, tensor of shape ``(6,nx,ny,nz)``
    xizero :
            Normalised Fourier momentum vectors :math:`\\mathring{\\widetilde{\\boldsymbol{\\xi}}}` computed with :py:func:`get_xizero`
    ref_params :
        Lame coefficients :math:`\\lambda^0`, :math:`\\mu^0` of isotropic reference material passed in the form ``{"lambda":lambda0, "mu":mu0}``

    Returns
    =======
    jax.Array
        Solution :math:`\\hat{\\varepsilon}_{k\\ell}`, tensor of shape ``(6,nx,ny,nz)``
    """
    lambda0 = ref_params["lambda"]
    mu0 = ref_params["mu"]
    epsilon_hat_A = jnp.stack(
        [
            xizero[0, ...] ** 2 * tau_hat[0, ...]
            + xizero[0, ...]
            * (xizero[2, ...] * tau_hat[4, ...] + xizero[1, ...] * tau_hat[3]),
            xizero[1, ...] ** 2 * tau_hat[1, ...]
            + xizero[1, ...]
            * (xizero[2, ...] * tau_hat[5, ...] + xizero[0, ...] * tau_hat[3]),
            xizero[2, ...] ** 2 * tau_hat[2, ...]
            + xizero[2, ...]
            * (xizero[1, ...] * tau_hat[5, ...] + xizero[0, ...] * tau_hat[4]),
            1
            / 2
            * (
                xizero[0, ...] * xizero[1, ...] * (tau_hat[0, ...] + tau_hat[1, ...])
                + (xizero[0, ...] ** 2 + xizero[1, ...] ** 2) * tau_hat[3, ...]
                + xizero[2, ...]
                * (xizero[0, ...] * tau_hat[5, ...] + xizero[1, ...] * tau_hat[4, ...])
            ),
            1
            / 2
            * (
                xizero[0, ...] * xizero[2, ...] * (tau_hat[0, ...] + tau_hat[2, ...])
                + (xizero[0, ...] ** 2 + xizero[2, ...] ** 2) * tau_hat[4, ...]
                + xizero[1, ...]
                * (xizero[0, ...] * tau_hat[5, ...] + xizero[2, ...] * tau_hat[3, ...])
            ),
            1
            / 2
            * (
                xizero[1, ...] * xizero[2, ...] * (tau_hat[1, ...] + tau_hat[2, ...])
                + (xizero[1, ...] ** 2 + xizero[2, ...] ** 2) * tau_hat[5, ...]
                + xizero[0, ...]
                * (xizero[1, ...] * tau_hat[4, ...] + xizero[2, ...] * tau_hat[3, ...])
            ),
        ]
    )
    Xi = jnp.stack(
        [
            xizero[0, ...] ** 2,
            xizero[1, ...] ** 2,
            xizero[2, ...] ** 2,
            xizero[0, ...] * xizero[1, ...],
            xizero[0, ...] * xizero[2, ...],
            xizero[1, ...] * xizero[2, ...],
        ]
    )
    Xi_dot_tau = (
        xizero[0, ...] ** 2 * tau_hat[0, ...]
        + xizero[1, ...] ** 2 * tau_hat[1, ...]
        + xizero[2, ...] ** 2 * tau_hat[2, ...]
        + 2
        * (
            xizero[0, ...] * xizero[1, ...] * tau_hat[3, ...]
            + xizero[0, ...] * xizero[2, ...] * tau_hat[4, ...]
            + xizero[1, ...] * xizero[2, ...] * tau_hat[5, ...]
        )
    )
    epsilon_hat_B = Xi * Xi_dot_tau
    return jnp.asarray(
        1
        / mu0
        * (-epsilon_hat_A + (lambda0 + mu0) / (lambda0 + 2 * mu0) * epsilon_hat_B)
    )


def fourier_solve_anisotropic(
    tau_hat: jax.Array, N_reference: np.ndarray, xizero: np.ndarray
) -> jax.Array:
    """Solve residual equation for homogeneous anisotropic reference material in Fourier space

    For given :math:`\\widehat{\\tau}` compute

    .. math::

        \\hat{\\varepsilon}_{k\\ell} = -\\widehat{\\Gamma}^0_{k\\ell ij} \\widehat{\\tau}_{ij}

    for a homogeneous anisotropic reference material. See Section :ref:`appendix_Gamma0_anisotropic` for details.

    Parameters
    ==========
    tau_hat :
        right hand side :math:`\\widehat{\\tau}`, tensor of shape ``(6,nx,ny,nz)``
    N_reference :
        Inverse :math:`N^0 = (K^0)^{-1}` of acoustic tensor for reference material as computed with :py:func:`get_inverse_anisotropic_acoustic_tensor`
    xizero :
        Normalised Fourier momentum vectors :math:`\\mathring{\\widetilde{\\boldsymbol{\\xi}}}` computed with :py:func:`get_xizero`

    Returns
    =======
    jax.Array
        Solution :math:`\\hat{\\varepsilon}_{k\\ell}`, tensor of shape ``(6,nx,ny,nz)``
    """
    Gamma0 = jnp.stack(
        [
            jnp.stack(
                [
                    N_reference[0, 0] * xizero[0] ** 2,
                    N_reference[0, 1] * xizero[0] * xizero[1],
                    N_reference[0, 2] * xizero[0] * xizero[2],
                    xizero[0]
                    * (N_reference[0, 0] * xizero[1] + N_reference[0, 1] * xizero[0]),
                    xizero[0]
                    * (N_reference[0, 0] * xizero[2] + N_reference[0, 2] * xizero[0]),
                    xizero[0]
                    * (N_reference[0, 1] * xizero[2] + N_reference[0, 2] * xizero[1]),
                ]
            ),
            jnp.stack(
                [
                    N_reference[0, 1] * xizero[0] * xizero[1],
                    N_reference[1, 1] * xizero[1] ** 2,
                    N_reference[1, 2] * xizero[1] * xizero[2],
                    xizero[1]
                    * (N_reference[0, 1] * xizero[1] + N_reference[1, 1] * xizero[0]),
                    xizero[1]
                    * (N_reference[0, 1] * xizero[2] + N_reference[1, 2] * xizero[0]),
                    xizero[1]
                    * (N_reference[1, 1] * xizero[2] + N_reference[1, 2] * xizero[1]),
                ]
            ),
            jnp.stack(
                [
                    N_reference[0, 2] * xizero[0] * xizero[2],
                    N_reference[1, 2] * xizero[1] * xizero[2],
                    N_reference[2, 2] * xizero[2] ** 2,
                    xizero[2]
                    * (N_reference[0, 2] * xizero[1] + N_reference[1, 2] * xizero[0]),
                    xizero[2]
                    * (N_reference[0, 2] * xizero[2] + N_reference[2, 2] * xizero[0]),
                    xizero[2]
                    * (N_reference[1, 2] * xizero[2] + N_reference[2, 2] * xizero[1]),
                ]
            ),
            jnp.stack(
                [
                    xizero[0]
                    * (N_reference[0, 0] * xizero[1] + N_reference[0, 1] * xizero[0])
                    / 2,
                    xizero[1]
                    * (N_reference[0, 1] * xizero[1] + N_reference[1, 1] * xizero[0])
                    / 2,
                    xizero[2]
                    * (N_reference[0, 2] * xizero[1] + N_reference[1, 2] * xizero[0])
                    / 2,
                    N_reference[0, 0] * xizero[1] ** 2 / 2
                    + N_reference[0, 1] * xizero[0] * xizero[1]
                    + N_reference[1, 1] * xizero[0] ** 2 / 2,
                    N_reference[0, 0] * xizero[1] * xizero[2] / 2
                    + N_reference[0, 1] * xizero[0] * xizero[2] / 2
                    + N_reference[0, 2] * xizero[0] * xizero[1] / 2
                    + N_reference[1, 2] * xizero[0] ** 2 / 2,
                    N_reference[0, 1] * xizero[1] * xizero[2] / 2
                    + N_reference[0, 2] * xizero[1] ** 2 / 2
                    + N_reference[1, 1] * xizero[0] * xizero[2] / 2
                    + N_reference[1, 2] * xizero[0] * xizero[1] / 2,
                ]
            ),
            jnp.stack(
                [
                    xizero[0]
                    * (N_reference[0, 0] * xizero[2] + N_reference[0, 2] * xizero[0])
                    / 2,
                    xizero[1]
                    * (N_reference[0, 1] * xizero[2] + N_reference[1, 2] * xizero[0])
                    / 2,
                    xizero[2]
                    * (N_reference[0, 2] * xizero[2] + N_reference[2, 2] * xizero[0])
                    / 2,
                    N_reference[0, 0] * xizero[1] * xizero[2] / 2
                    + N_reference[0, 1] * xizero[0] * xizero[2] / 2
                    + N_reference[0, 2] * xizero[0] * xizero[1] / 2
                    + N_reference[1, 2] * xizero[0] ** 2 / 2,
                    N_reference[0, 0] * xizero[2] ** 2 / 2
                    + N_reference[0, 2] * xizero[0] * xizero[2]
                    + N_reference[2, 2] * xizero[0] ** 2 / 2,
                    N_reference[0, 1] * xizero[2] ** 2 / 2
                    + N_reference[0, 2] * xizero[1] * xizero[2] / 2
                    + N_reference[1, 2] * xizero[0] * xizero[2] / 2
                    + N_reference[2, 2] * xizero[0] * xizero[1] / 2,
                ]
            ),
            jnp.stack(
                [
                    xizero[0]
                    * (N_reference[0, 1] * xizero[2] + N_reference[0, 2] * xizero[1])
                    / 2,
                    xizero[1]
                    * (N_reference[1, 1] * xizero[2] + N_reference[1, 2] * xizero[1])
                    / 2,
                    xizero[2]
                    * (N_reference[1, 2] * xizero[2] + N_reference[2, 2] * xizero[1])
                    / 2,
                    N_reference[0, 1] * xizero[1] * xizero[2] / 2
                    + N_reference[0, 2] * xizero[1] ** 2 / 2
                    + N_reference[1, 1] * xizero[0] * xizero[2] / 2
                    + N_reference[1, 2] * xizero[0] * xizero[1] / 2,
                    N_reference[0, 1] * xizero[2] ** 2 / 2
                    + N_reference[0, 2] * xizero[1] * xizero[2] / 2
                    + N_reference[1, 2] * xizero[0] * xizero[2] / 2
                    + N_reference[2, 2] * xizero[0] * xizero[1] / 2,
                    N_reference[1, 1] * xizero[2] ** 2 / 2
                    + N_reference[1, 2] * xizero[1] * xizero[2]
                    + N_reference[2, 2] * xizero[1] ** 2 / 2,
                ]
            ),
        ]
    )
    return -jnp.einsum("abijk,bijk->aijk", Gamma0, tau_hat)
