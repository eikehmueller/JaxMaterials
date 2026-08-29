"""Relative convergence used for convergence tests

See discussion in Section :ref:`sec:stopping_criterion` on how the relative divergence is used to check whether the Lippmann Schwinger solver has converged.
"""

import jax
import numpy as np
from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.derivatives import backward_divergence

__all__ = ["relative_divergence", "relative_divergence_fourier"]


def relative_divergence(sigma: jax.Array, grid_spec: GridSpec) -> jax.Array:
    """Relative divergence of stress :math:`\\sigma` in real space

    Computes the ratio

    .. math::

        R(\\sigma) = \\frac{\\sqrt{\\langle\\|D^-\\cdot \\sigma\\|^2\\rangle}}{\\|\\langle \\sigma \\rangle\\|}

    defined in Section :ref:`sec:stopping_criterion`.

    Parameters
    ==========
    sigma :
        stress :math:`\\sigma`, array of shape ``(6,nx,ny,nz)``
    grid_spec :
        specification of computational grid

    Returns
    =======
    float :
        ratio :math:`R(\\sigma)`
    """
    dsigma = backward_divergence(sigma, grid_spec)
    dsigma_nrm = jnp.sqrt(jnp.sum(dsigma**2))
    sigma_avg = jnp.mean(sigma, axis=[1, 2, 3])
    sigma_avg_nrm = jnp.sqrt(
        jnp.sum(sigma_avg[:3] ** 2) + 2 * jnp.sum(sigma_avg[3:] ** 2)
    )
    return dsigma_nrm / (jnp.sqrt(grid_spec.number_of_voxels) * sigma_avg_nrm)


def relative_divergence_fourier(sigma_hat: jax.Array, xi: np.ndarray) -> jax.Array:
    """Relative divergence of stress :math:`\\sigma` in Fourier space

    Computes the ratio

    .. math::

        \\widehat{R}(\\widehat{\\sigma}) = \\frac{\\sqrt{N\\langle\\|\\xi\\cdot \\widehat{\\sigma}\\|^2\\rangle}}{\\|\\widehat{\\sigma}_{\\boldsymbol{\\xi}=0}\\|}

    Defined in Section :ref:`sec:stopping_criterion`. Here :math:`\\boldsymbol{\\xi}` are the Fourier vectors computed in :py:func:`jaxmaterials.solver.fourier.get_xi`.

    Parameters
    ==========
    sigma_hat :
        stress :math:`\\sigma` in Fourier space, array of shape ``(6,nx,ny,nz)``
    xi :
        Fourier vectors, array of shape ``(6,nx,ny,nz)``

    Returns
    =======
    float
        ratio :math:`\\widehat{R}(\\widehat{\\sigma})`
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
