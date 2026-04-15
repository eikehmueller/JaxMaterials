"""Functionality for computations in Fourier space"""

import numpy as np
from jax import numpy as jnp

__all__ = [
    "get_xizero",
    "get_xi",
    "get_anisotropic_acoustic_tensor",
    "get_inverse_anisotropic_acoustic_tensor",
    "fourier_solve_isotropic",
    "fourier_solve_anisotropic",
]


def get_xi(grid_spec, dtype=jnp.float64):
    """Construct the un-normalised momentum vectors in Fourier space

    Let k = (k_0,k_1,k_2) with k_d = 0,1,...,N_d-1 be a three-dimensional Fourier index.

    The normalised momentum vector is xi_d = 2 pi k_d / N_d, with 0 <= xi_0 < 2pi

    For a given k we then have that

    tilde(xi)_0 = 2/h_0 * sin(xi_0/2) * cos(xi_1/2) * cos(xi_2/2)
    tilde(xi)_1 = 2/h_1 * cos(xi_0/2) * sin(xi_1/2) * cos(xi_2/2)
    tilde(xi)_2 = 2/h_2 * cos(xi_0/2) * cos(xi_1/2) * sin(xi_2/2)

    This function returns a tensor of shape (3,N_0,N_1,N_2) which contains
    the normalised xi^0 = tilde(xi) for all Fourier modes.

     :arg grid_spec: namedtuple with grid specifications
     :arg dtype: data type

    """
    # Normalised momentum vectors in all three spatial directions
    K = [
        2 * np.pi * np.arange(n) / n for n in (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    ]
    # Grid with normalised momentum vectors
    xi = np.meshgrid(*K, indexing="ij")
    # Grid with tilde(xi)
    xi = np.stack(
        [
            2
            * grid_spec.nx
            / grid_spec.Lx
            * np.sin(xi[0] / 2)
            * np.cos(xi[1] / 2)
            * np.cos(xi[2] / 2),
            2
            * grid_spec.ny
            / grid_spec.Ly
            * np.cos(xi[0] / 2)
            * np.sin(xi[1] / 2)
            * np.cos(xi[2] / 2),
            2
            * grid_spec.nz
            / grid_spec.Lz
            * np.cos(xi[0] / 2)
            * np.cos(xi[1] / 2)
            * np.sin(xi[2] / 2),
        ]
    )
    return xi.astype(dtype)


def get_xizero(grid_spec, dtype=jnp.float64):
    """Construct the normalised momentum vectors in Fourier space

    Let k = (k_0,k_1,k_2) with k_d = 0,1,...,N_d-1 be a three-dimensional Fourier index.

    The normalised momentum vector is xi_d = 2 pi k_d / N_d, with 0 <= xi_0 < 2pi

    For a given k we then have that

    tilde(xi)_0 = 2/h_0 * sin(xi_0/2) * cos(xi_1/2) * cos(xi_2/2)
    tilde(xi)_1 = 2/h_1 * cos(xi_0/2) * sin(xi_1/2) * cos(xi_2/2)
    tilde(xi)_2 = 2/h_2 * cos(xi_0/2) * cos(xi_1/2) * sin(xi_2/2)

    This function returns a tensor of shape (3,nx,ny,nz) which contains
    the normalised xi^0 = tilde(xi) / ||tilde(xi)|| for all Fourier modes.

     :arg grid_spec: namedtuple with grid specifications
     :arg dtype: data type

    """
    xi_tilde = get_xi(grid_spec, dtype=dtype)
    # Normalise tilde(xi) to obtain xi^0
    xi_nrm = np.linalg.norm(xi_tilde, axis=0)
    tolerance = 1.0e-12 if dtype == jnp.float64 else 1.0e-6
    xi_nrm[xi_nrm < tolerance] = 1  # avoid division by zero
    return (xi_tilde / xi_nrm).astype(dtype)


def get_anisotropic_acoustic_tensor(xizero, stiffness_tensor0):
    """Acoustic tensor for a homogeneous anisotropic reference material

    Assemble the 3x3 acoustic tensor for each Fourier mode for a homogeneous isotropic
    reference material characterised by the stiffness tensor C^{0}.
    The stiffness tensor is passed as a vector which contains the 21 independent entries
    of the reference stiffness tensor.

    Returns a rank 5 tensor of shape (3,3,Nx,Ny,Nz) with the acoustic tensor

    :arg xizero: Fourier vectors
    :arg stiffness_tensor0: stiffness tensor
    """
    return jnp.stack(
        [
            jnp.stack(
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
            jnp.stack(
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
            jnp.stack(
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


def get_inverse_anisotropic_acoustic_tensor(xizero, stiffness_tensor0):
    """Inverse of the acoustic tensor for a homogeneous anisotropic reference material

    Assemble the 3x3 acoustic tensor for each Fourier mode for a homogeneous isotropic
    reference material characterised by the stiffness tensor C^{0} and invert it.
    The stiffness tensor is passed as a vector which contains the 21 independent entries of the
    reference stiffness tensor.
    Returns a rank 5 tensor of shape (3,3,Nx,Ny,Nz)

    :arg xizero: Fourier vectors
    :arg stiffness_tensor0: stiffness tensor
    """
    K0 = get_anisotropic_acoustic_tensor(xizero, stiffness_tensor0)
    K0_transpose = jnp.transpose(K0, axes=(2, 3, 4, 0, 1))
    xi_nrm = (xizero[0] ** 2 + xizero[1] ** 2 + xizero[2] ** 2) > 1.0e-8
    # Avoid inverting a singular zero-frequency block, which can produce NaN gradients.
    eye3 = jnp.eye(3, dtype=K0_transpose.dtype)
    K0_safe = jnp.where(xi_nrm[..., None, None], K0_transpose, eye3)
    N0_transpose = jnp.linalg.inv(K0_safe)
    return xi_nrm * jnp.transpose(N0_transpose, axes=(3, 4, 0, 1, 2))


def fourier_solve_isotropic(tau_hat, lmbda0, mu0, xizero):
    """Solve residual equation for homogeneous isotropic reference material in Fourier space

    Computes hat(epsilon)_{kl} = -Gamma^0_{klij} hat(tau)_{ij} for a homogeneous isotropic
    reference material which is characterised by the two Lame parameters lambda^0 and mu^0.

    :arg tau_hat: The residual hat(tau) in Fourier space
    :arg lmbda0: coefficient lambda^0 of homogeneous reference material
    :arg mu0: coefficient mu^0 of homogeneous reference material
    :arg xizero: Normalised momentum vectors
    """
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
    return (
        1 / mu0 * (-epsilon_hat_A + (lmbda0 + mu0) / (lmbda0 + 2 * mu0) * epsilon_hat_B)
    )


def fourier_solve_anisotropic(tau_hat, N_reference, xizero):
    """Solve residual equation for homogeneous anisotropic reference material in Fourier space

    Computes hat(epsilon)_{kl} = -Gamma^0_{klij} hat(tau)_{ij} for a homogeneous anisotropic
    reference material which is characterised by the reference stiffness tensor C^{0}. This
    is implicitly contained in the inverse of the acoustic tensor

    :arg tau_hat: The residual hat(tau) in Fourier space
    :arg N_reference: inverse of acoustic tensor for reference material
    :arg xizero: Normalised momentum vectors
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
