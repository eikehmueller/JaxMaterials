"""Methods for relating strain :math:`\\epsilon` and stress :math:`\\sigma=\\Sigma(\\epsilon|\\theta)`

Implementation of Hooke's law for isotropic and anisotropic materials. All methods conform to the
common interface::

    def compute_sigma(epsilon, params):
        # compute sigma from epsilon and params
        return sigma

where ``epsilon`` is the strain :math:`\\epsilon` and `params` is
a `jax.pytree <https://docs.jax.dev/en/latest/pytrees.html>`_ with the parameters :math:`\\theta`.
"""

from jax import numpy as jnp

__all__ = [
    "compute_sigma_isotropic",
    "compute_sigma_anisotropic",
    "compute_sigma_inelastic",
]


def compute_sigma_isotropic(epsilon, params):
    """Compute stress from strain for isotropic material

    Returns :math:`\\sigma_{ij} = C_{ijkl}\\epsilon_{kl}` for an isotropic material characterised
    by the two Lame parameters :math:`\\lambda` and :math:`\\mu`.
    Voigt notation ``{00,11,22,01,02,12}`` is used to index the stress and strain tensor.

    Parameters
    ==========
    epsilon : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        strain field of shape ``(6,nx,ny,nz)``, using Voight notation for indexing.
    params : dict
        Lame parameters, dictionary of the form ``{"lambda":lambda, "mu":mu}`` where
        ``lambda`` and ``mu`` are of shape ``(nx,ny,nz)``
    """
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = 2 * params["mu"] * epsilon + params["lambda"] * jnp.stack(
        3 * [tr_epsilon] + 3 * [jnp.zeros(epsilon.shape[-3:], dtype=epsilon.dtype)]
    )
    return sigma


def compute_sigma_anisotropic(epsilon, params):
    """Compute stress from strain for anisotropic material

    Returns :math:`\\sigma_{ij} = C_{ijkl}\\epsilon_{kl}` for an anisotropic material characterised
    by the 21 entries of the stiffness tensor :math:``C``.

    Voigt notation ``{00,11,22,01,02,12}`` is used to index the stress and strain tensor.
    The 21 independent entries of the stiffness tensor are given by

    
    .. math::
      \\begin{aligned}
        C_{0}  &= C_{00,00},&   C_{1}  &= C_{11,11},&   C_{2}  &= C_{22,22},\\\\
        C_{3}  &= C_{01,01},&   C_{4}  &= C_{02,02},&   C_{5}  &= C_{12,12},\\\\
        C_{6}  &= C_{00,11},&   C_{7}  &= C_{00,22},&   C_{8}  &= C_{11,22},\\\\
        C_{9}  &= C_{00,01},&   C_{10} &= C_{00,02},&   C_{11} &= C_{00,12},\\\\
        C_{12} &= C_{11,01},&   C_{13} &= C_{11,02},&   C_{14} &= C_{11,12},\\\\
        C_{15} &= C_{22,01},&   C_{16} &= C_{22,02},&   C_{17} &= C_{22,12},\\\\
        C_{18} &= C_{01,02},&   C_{19} &= C_{01,12},&   C_{20} &= C_{02,12}.
      \\end{aligned}

    Parameters
    ==========
    epsilon : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        strain field :math:`\\epsilon` of shape ``(6,nx,ny,nz)``, using Voigt notation for indexing.
    params : dict
        dictionary of the form ``{"stiffness_tensor":stiffness_tensor}``
        with vector representation of stiffness tensor :math:`C` which is passed as an array of
        shape ``(21,nx,ny,nz)``
    """
    stiffness_tensor = params["stiffness_tensor"]
    sigma = jnp.stack(
        [
            stiffness_tensor[0] * epsilon[0]
            + stiffness_tensor[6] * epsilon[1]
            + stiffness_tensor[7] * epsilon[2]
            + 2 * stiffness_tensor[9] * epsilon[3]
            + 2 * stiffness_tensor[10] * epsilon[4]
            + 2 * stiffness_tensor[11] * epsilon[5],
            stiffness_tensor[1] * epsilon[1]
            + stiffness_tensor[6] * epsilon[0]
            + stiffness_tensor[8] * epsilon[2]
            + 2 * stiffness_tensor[12] * epsilon[3]
            + 2 * stiffness_tensor[13] * epsilon[4]
            + 2 * stiffness_tensor[14] * epsilon[5],
            stiffness_tensor[2] * epsilon[2]
            + stiffness_tensor[7] * epsilon[0]
            + stiffness_tensor[8] * epsilon[1]
            + 2 * stiffness_tensor[15] * epsilon[3]
            + 2 * stiffness_tensor[16] * epsilon[4]
            + 2 * stiffness_tensor[17] * epsilon[5],
            2 * stiffness_tensor[3] * epsilon[3]
            + stiffness_tensor[9] * epsilon[0]
            + stiffness_tensor[12] * epsilon[1]
            + stiffness_tensor[15] * epsilon[2]
            + 2 * stiffness_tensor[18] * epsilon[4]
            + 2 * stiffness_tensor[19] * epsilon[5],
            2 * stiffness_tensor[4] * epsilon[4]
            + stiffness_tensor[10] * epsilon[0]
            + stiffness_tensor[13] * epsilon[1]
            + stiffness_tensor[16] * epsilon[2]
            + 2 * stiffness_tensor[18] * epsilon[3]
            + 2 * stiffness_tensor[20] * epsilon[5],
            2 * stiffness_tensor[5] * epsilon[5]
            + stiffness_tensor[11] * epsilon[0]
            + stiffness_tensor[14] * epsilon[1]
            + stiffness_tensor[17] * epsilon[2]
            + 2 * stiffness_tensor[19] * epsilon[3]
            + 2 * stiffness_tensor[20] * epsilon[4],
        ]
    )
    return sigma


def compute_sigma_inelastic(epsilon, params):
    """Compute stress from strain for inelastic material

    Returns :math:`\\sigma_{ij} = \\sigma_{ij}(\\epsilon_{kl})` for an isotropic material.
    Voigt notation ``{00,11,22,01,02,12}`` is used for the stress and strain tensor.

    Parameters
    ==========
    epsilon : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        strain field :math:`\\epsilon`
    params : dict
        Lame parameters :math:`\\lambda` and :math:`\\mu`, dictionary
        of the form ``{"lambda":lambda, "mu":mu}``, where ``lambda`` and ``mu`` are
        of shape ``(nx,ny,nz)``
    """
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = 2 * params["mu"] / (
        1 + 0.1 * jnp.linalg.norm(tr_epsilon)
    ) * epsilon + params["lambda"] * jnp.stack(
        3 * [tr_epsilon] + 3 * [jnp.zeros(epsilon.shape[-3:], dtype=epsilon.dtype)]
    )
    return sigma
