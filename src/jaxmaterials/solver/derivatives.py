"""Implementation of discrete derivatives on structured grid"""

from jax import numpy as jnp

__all__ = ["backward_derivative", "backward_divergence"]


def backward_derivative(g, grid_spec, direction):
    """Discrete backward derivative :math:`\\partial_i g` of function :math:`g(x)`

    For ``direction=0`` the backward derivative is defined as the finite difference

    .. math::

        D^-_0g(x) = \\frac{1}{h_0} \\left(S_{1,2}g(x) - S_{1,2}g(x-e_0)\\right)

    where :math:`h_0` is the grid spacing in direction 0 and :math:`S_{1,2}` denotes
    averaging over the other two directions. More generally

    .. math::

        S_{i,j}g(x) = \\frac{1}{4}\\left(g(x)+g(x-e_i)+g(x-e_j)+g(x-e_i-e_j)\\right)

    Similarly we have for the discrete backward derivatives in the other directions:

    .. math::
    
        \\begin{aligned}
            D^-_1g(x) &= \\frac{1}{h_1} \\left(S_{0,2}g(x) - S_{0,2}g(x-e_1)\\right)\\\\
            D^-_2g(x) &= \\frac{1}{h_2} \\left(S_{0,1}g(x) - S_{0,1}g(x-e_2)\\right)
        \\end{aligned}

    Parameters
    ==========
    g : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        function to take the derivative of. Assumed to be of shape ``(*,nx,ny,nz)``
    grid_spec : :py:class:`jaxmaterials.common.GridSpec`
        specification of computational grid
    direction : int
        direction in which to take the derivative, can be ``0``, ``1`` or ``2``

    Returns
    =======
    Tensor of shape ``(*,nx,ny,nz)`` with finite difference :math:`D^-_i g` in direction :math:`i`
    """
    if direction == 0:
        dg = (
            0.25
            * grid_spec.nx
            / grid_spec.Lx
            * (
                g
                + jnp.roll(g, 1, axis=-2)
                + jnp.roll(g, 1, axis=-1)
                + jnp.roll(g, (1, 1), axis=(-2, -1))
                - jnp.roll(g, 1, axis=-3)
                - jnp.roll(g, (1, 1), axis=(-3, -2))
                - jnp.roll(g, (1, 1), axis=(-3, -1))
                - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
            )
        )
    elif direction == 1:
        dg = (
            0.25
            * grid_spec.ny
            / grid_spec.Ly
            * (
                g
                + jnp.roll(g, 1, axis=-3)
                + jnp.roll(g, 1, axis=-1)
                + jnp.roll(g, (1, 1), axis=(-3, -1))
                - jnp.roll(g, 1, axis=-2)
                - jnp.roll(g, (1, 1), axis=(-3, -2))
                - jnp.roll(g, (1, 1), axis=(-2, -1))
                - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
            )
        )
    elif direction == 2:
        dg = (
            0.25
            * grid_spec.nz
            / grid_spec.Lz
            * (
                g
                + jnp.roll(g, 1, axis=-3)
                + jnp.roll(g, 1, axis=-2)
                + jnp.roll(g, (1, 1), axis=(-3, -2))
                - jnp.roll(g, 1, axis=-1)
                - jnp.roll(g, (1, 1), axis=(-3, -1))
                - jnp.roll(g, (1, 1), axis=(-2, -1))
                - jnp.roll(g, (1, 1, 1), axis=(-3, -2, -1))
            )
        )
    else:
        raise IndexError(f"Invalid direction: {direction}")
    return dg


def backward_divergence(sigma, grid_spec):
    """Discrete backward divergence :math:`\\partial_i\\sigma_{ij}` of symmetric :math:`3\\times 3` tensor-valued function :math:`\\sigma_{ij}(x)`

    The components of the tensor are assumed to be represented in vector form
    using Voigt notation:

    .. math::
    
        (\\sigma_{00}, \\sigma_{11}, \\sigma_{22}, \\sigma_{01}, \\sigma_{02}, \\sigma_{12})

    The resulting divergence is a vector

    .. math::

        \\begin{pmatrix}
            \\frac{d\\sigma_{00}}{dx_0} + \\frac{d\\sigma_{01}}{dx_1} + \\frac{d\\sigma_{02}}{dx_2} \\\\
            \\frac{d\\sigma_{10}}{dx_0} + \\frac{d\\sigma_{11}}{dx_1} + \\frac{d\\sigma_{12}}{dx_2} \\
            \\frac{d\\sigma_{20}}{dx_0} + \\frac{d\\sigma_{21}}{dx_1} + \\frac{d\\sigma_{22}}{dx_2}
        \\end{pmatrix}

    Parameters
    ==========
    sigma : `numpy.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
        tensor to take the divergence of. Assumed to be of shape ``(6,nx,ny,nz)``
    grid_spec : :py:class:`jaxmaterials.common.GridSpec`
        specification of computational grid

    Returns
    =======
    Tensor of shape ``(3,nx,ny,nz)`` with the three components of the divergence vector
    """
    return jnp.stack(
        [
            backward_derivative(sigma[0, ...], grid_spec, 0)
            + backward_derivative(sigma[3, ...], grid_spec, 1)
            + backward_derivative(sigma[4, ...], grid_spec, 2),
            backward_derivative(sigma[3, ...], grid_spec, 0)
            + backward_derivative(sigma[1, ...], grid_spec, 1)
            + backward_derivative(sigma[5, ...], grid_spec, 2),
            backward_derivative(sigma[4, ...], grid_spec, 0)
            + backward_derivative(sigma[5, ...], grid_spec, 1)
            + backward_derivative(sigma[2, ...], grid_spec, 2),
        ]
    )
