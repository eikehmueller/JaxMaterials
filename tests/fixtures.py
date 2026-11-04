"""Fixtures and common functionality used by all tests"""

import pytest
import jax
import numpy as np

from jaxmaterials.common import GridSpec, get_grid_spec


@pytest.fixture(params=[[64, 48, 32], [57, 43, 37]], ids=["even", "odd"])
def grid_spec(request):
    """Return grid specification"""
    # Domain size in all three spatial direction
    Lx = 1.2
    Ly = 0.8
    Lz = 0.9
    # Number of grid cells in all three spatial directions
    nx, ny, nz = request.param

    return get_grid_spec(Lx, Ly, Lz, nx=nx, ny=ny, nz=nz)


@pytest.fixture(params=[[8, 6, 4], [7, 5, 3]], ids=["even", "odd"])
def grid_spec_small(request):
    """Return grid specification"""
    # Domain size in all three spatial direction
    Lx = 1.2
    Ly = 0.8
    Lz = 0.9
    # Number of grid cells in all three spatial directions
    nx, ny, nz = request.param

    return get_grid_spec(Lx, Ly, Lz, nx=nx, ny=ny, nz=nz)


@pytest.fixture(params=[[512, 384, 256], [453, 347, 297]], ids=["even", "odd"])
def grid_spec_highres(request):
    """Return grid specification"""
    # Domain size in all three spatial direction
    Lx = 1.2
    Ly = 0.8
    Lz = 0.9
    # Number of grid cells in all three spatial directions
    nx, ny, nz = request.param

    return get_grid_spec(Lx, Ly, Lz, nx=nx, ny=ny, nz=nz)


@pytest.fixture
def rng():
    """Random number generator"""
    return np.random.default_rng(seed=784173)


def initialise_isotropic_material(grid_spec, rng, dtype):
    """Construct random Lame coefficients

    :arg grid_spec: specification of grid
    :arg rng: random number generator
    :arg dtype: data type
    """
    shape = (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    lmbda = 1.1 * np.ones(shape=shape) + rng.uniform(size=shape, low=-0.2, high=+0.2)
    mu = 0.9 * np.ones(shape=shape) + rng.uniform(size=shape, low=-0.2, high=+0.2)
    return {"lambda": np.array(lmbda, dtype=dtype), "mu": np.array(mu, dtype=dtype)}


def reference_parameters(params):
    """Compute Lame reference coefficients for isotropic, homogeneous material

    :arg params: dictionary with Lame parameters
    """
    return {
        key: 1 / 2 * (np.min(value) + np.max(value)) for key, value in params.items()
    }


def perturbed_parameters(rng, params, delta=0.1):
    """Build mildly anisotropic stiffness tensor around isotropic baseline

    :arg rng: random number generator
    :arg params: Lame coefficients of isotropic material
    :arg delta: magnitude of perturbation
    """
    lmbda = params["lambda"]
    mu = params["mu"]
    perturb = lambda scale: scale + rng.uniform(-delta, delta, size=mu.shape)
    return {
        "stiffness_tensor": np.stack(
            [
                perturb(2 * mu + lmbda),
                perturb(2 * mu + lmbda),
                perturb(2 * mu + lmbda),
                perturb(mu),
                perturb(mu),
                perturb(mu),
                perturb(lmbda),
                perturb(lmbda),
                perturb(lmbda),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
                perturb(0),
            ]
        ).astype(mu.dtype)
    }


def get_parallel_gridspec(
    parallel_decomposition: str | None,
) -> tuple[GridSpec, jax.sharding.NamedSharding]:
    """Return the specification of the grid and sharding

    Parameters
    ==========
    parallel_decomposition : str | None
        Type of decomposition, must be either "slab", "pencil" or None

    Returns
    =======
    tuple of grid specification sharding
    """
    if parallel_decomposition == "pencil":
        grid_spec = get_grid_spec(nx=24, ny=12, nz=6, px=2, py=3)
        mesh = jax.make_mesh((2, 3), axis_names=("x", "y"))
        sharding = jax.sharding.NamedSharding(mesh, jax.P("x", "y", None))
    elif parallel_decomposition == "slab":
        grid_spec = get_grid_spec(nx=24, ny=12, nz=6, pz=6)
        mesh = jax.make_mesh((6,), axis_names=("z",))
        sharding = jax.sharding.NamedSharding(mesh, jax.P(None, None, "z"))
    elif parallel_decomposition is None:
        grid_spec = get_grid_spec(nx=24, ny=12, nz=6)
        sharding = None
    else:
        raise RuntimeError(f"Unknown parallel decomposition: {parallel_decomposition}")

    return grid_spec, sharding
