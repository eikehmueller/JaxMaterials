import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxmaterials.solver.derivatives import backward_derivative, backward_divergence
from fixtures import grid_spec, grid_spec_highres, rng

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_derivative_constant(grid_spec, direction, dtype):
    """Test backward derivative with constant function

    (test written by github copilot)

    The derivative should be 0.
    """
    # Constant function
    f = jnp.ones((grid_spec.nx, grid_spec.ny, grid_spec.nz), dtype=dtype)

    # Compute derivative
    df = backward_derivative(f, grid_spec, direction)

    # Expected derivative is 0
    expected = jnp.zeros_like(df)

    rtol = 1.0e-6 if dtype == np.float32 else 1.0e-12
    assert jnp.allclose(df, expected, rtol=rtol)


@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_derivative_sine(grid_spec_highres, direction, dtype):
    """Test backward derivative with sinusoidal function f(x,y,z) = sin(2π x/Lx) + sin(2π y/Ly) + sin(2π z/Lz)

    (test originally written by github copilot and modified)

    The derivative in direction d should be proportional to
    (2π/L_d) * cos(2π coord_d/L_d).
    """
    # Create coordinate arrays
    hx = grid_spec_highres.Lx / grid_spec_highres.nx
    x = jnp.linspace(
        hx / 2, grid_spec_highres.Lx - hx / 2, grid_spec_highres.nx, dtype=dtype
    )
    hy = grid_spec_highres.Ly / grid_spec_highres.ny
    y = jnp.linspace(
        hy / 2, grid_spec_highres.Ly - hy / 2, grid_spec_highres.ny, dtype=dtype
    )
    hz = grid_spec_highres.Lz / grid_spec_highres.nz
    z = jnp.linspace(
        hz / 2, grid_spec_highres.Lz - hz / 2, grid_spec_highres.nz, dtype=dtype
    )

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    # Sinusoidal function that is periodic
    f = (
        jnp.sin(2 * jnp.pi * X / grid_spec_highres.Lx)
        * jnp.sin(2 * jnp.pi * Y / grid_spec_highres.Ly)
        * jnp.sin(2 * jnp.pi * Z / grid_spec_highres.Lz)
    )

    # Compute derivative
    df = backward_derivative(f, grid_spec_highres, direction)

    # Expected derivative
    if direction == 0:
        expected = (
            (2 * jnp.pi / grid_spec_highres.Lx)
            * jnp.cos(2 * jnp.pi * X / grid_spec_highres.Lx)
            * jnp.sin(2 * jnp.pi * Y / grid_spec_highres.Ly)
            * jnp.sin(2 * jnp.pi * Z / grid_spec_highres.Lz)
        )
    elif direction == 1:
        expected = (
            (2 * jnp.pi / grid_spec_highres.Ly)
            * jnp.sin(2 * jnp.pi * X / grid_spec_highres.Lx)
            * jnp.cos(2 * jnp.pi * Y / grid_spec_highres.Ly)
            * jnp.sin(2 * jnp.pi * Z / grid_spec_highres.Lz)
        )
    else:  # direction == 2
        expected = (
            (2 * jnp.pi / grid_spec_highres.Lz)
            * jnp.sin(2 * jnp.pi * X / grid_spec_highres.Lx)
            * jnp.sin(2 * jnp.pi * Y / grid_spec_highres.Ly)
            * jnp.cos(2 * jnp.pi * Z / grid_spec_highres.Lz)
        )
    atol = 0.1
    assert jnp.allclose(df, expected, atol=atol)


def test_backward_derivative_invalid_direction(grid_spec):
    """Test that backward_derivative raises error for invalid direction

    (test written by github copilot)
    """
    f = jnp.ones((grid_spec.nx, grid_spec.ny, grid_spec.nz))

    with pytest.raises((ValueError, IndexError)):
        backward_derivative(f, grid_spec, 3)  # Invalid direction
