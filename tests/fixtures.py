"""Fixtures and common functionality used by all tests"""

import pytest
import numpy as np
import pytest

from jaxmaterials.common import GridSpec


@pytest.fixture(params=[[64, 48, 32], [57, 43, 37]], ids=["even", "odd"])
def grid_spec(request):
    """Return grid specification"""
    # Domain size in all three spatial direction
    Lx = 1.2
    Ly = 0.8
    Lz = 0.9
    # Number of grid cells in all three spatial directions
    nx, ny, nz = request.param

    return GridSpec(nx, ny, nz, Lx, Ly, Lz)


@pytest.fixture
def rng():
    """Random number generator"""
    return np.random.default_rng(seed=784173)


def initialise_material(grid_spec, rng, dtype):
    """Construct random Lame parameters

    :arg grid_spec: specification of grid
    :arg rng: random number generator
    :arg dtype: data type
    """
    shape = (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    mu = np.ones(shape=shape) + rng.uniform(size=shape, low=-0.2, high=+0.2)
    lmbda = np.ones(shape=shape) + rng.uniform(size=shape, low=-0.2, high=+0.2)
    return np.array(mu, dtype=dtype), np.array(lmbda, dtype=dtype)
