"""Implementation of discrete derivatives"""

from jax import numpy as jnp

__all__ = ["backward_derivative", "backward_divergence"]


def backward_derivative(g, grid_spec, direction):
    """Compute the backward derivative in a specified direction

    :arg g: function to take the derivative of. Assumed to be of shape (*,N_0,N_1,N_2)
    :arg grid_spec: namedtuple with grid specification
    :arg direction: direction in which to take the derivative: 0, 1 or 2
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
    """Compute backward divergence of symmetric 3x3 tensor sigma_{ij}

    The componets of the tensor are assumed to be represented in vector form
    using Voigt notation:

    (sigma_{00}, sigma_{11}, sigma_{22}, sigma_{12}, sigma_{02}, sigma_{01})

    Returns a tensor of shape (3,N_0,N_1,N_2) with the three components of the
    divergence vector:

    [ dsigma_{00}/dx_0 + dsigma_{01}/dx_1 + dsigma_{02}/dx_2 ]
    [ dsigma_{10}/dx_0 + dsigma_{11}/dx_1 + dsigma_{12}/dx_2 ]
    [ dsigma_{20}/dx_0 + dsigma_{21}/dx_1 + dsigma_{22}/dx_2 ]

    :arg sigma: tensor representation using Voigt notation
    :arg grid_spec: namedtuple with grid specification
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
