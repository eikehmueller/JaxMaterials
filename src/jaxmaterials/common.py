"""Common definitions required throughout the code"""

import numpy as np
import typing

__all__ = ["GridSpec", "get_grid_spec"]


class GridSpec:
    """Specification of structured grid for the 3d domain :math:`[0,L_x] \\times [0,L_y] \\times [0,L_z]`

    The number of voxels in the three coordinate directions are :math:`n_x`, :math:`n_y`
    and :math:`n_z` respectively.
    """

    def __init__(
        self, Lx: float, Ly: float, Lz: float, nx: int, ny: int, nz: int
    ) -> None:
        """Initialise instance

        Parameters
        ==========
        Lx :
            domain size :math:`L_x` in x-direction
        Ly :
            domain size :math:`L_y` in y-direction
        Lz :
            domain size :math:`L_z` in z-direction
        nx :
            number of voxels :math:`n_x` in x-direction
        ny :
            number of voxels :math:`n_y` in y-direction
        nz :
            number of voxels :math:`n_z` in z-direction
        """
        assert nx > 0
        assert ny > 0
        assert nz > 0
        assert Lx > 0
        assert Ly > 0
        assert Lz > 0
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

    @property
    def number_of_voxels(self) -> int:
        """Total number of voxels :math:`N = n_x\\cdot n_y \\cdot n_z`"""
        return self._nx * self._ny * self._nz

    @property
    def Lx(self) -> float:
        """Size of grid in x-direction"""
        return self._Lx

    @property
    def Ly(self) -> float:
        """Size of grid in y-direction"""
        return self._Ly

    @property
    def Lz(self) -> float:
        """Size of grid in z-direction"""
        return self._Lz

    @property
    def nx(self) -> int:
        """Number of voxels in x-direction"""
        return self._nx

    @property
    def ny(self) -> int:
        """Number of voxels in y-direction"""
        return self._ny

    @property
    def nz(self) -> int:
        """Number of voxels in z-direction"""
        return self._nz

    @property
    def dx(self) -> float:
        """Extent :math:`h_x` of voxels in x-direction"""
        return self._Lx / self._nx

    @property
    def dy(self) -> float:
        """Extent :math:`h_y` of voxels in y-direction"""
        return self._Ly / self._ny

    @property
    def dz(self) -> float:
        """Extent :math:`h_z` of voxels in z-direction"""
        return self._Lz / self._nz

    @property
    def grid_spacings(self) -> tuple[float, float, float]:
        """Tuple of grid spacings :math:`(h_x,h_y,h_z)` with :math:`h_i=L_i/n_i`"""
        return (self.dx, self.dy, self.dz)

    @property
    def voxel_centers(self) -> np.ndarray:
        """Array with voxel centres"""
        x = np.linspace(self.dx / 2, self._Lx - self.dx / 2, self._nx)
        y = np.linspace(self.dy / 2, self._Ly - self.dy / 2, self._ny)
        z = np.linspace(self.dz / 2, self._Lz - self.dz / 2, self._nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)


def get_grid_spec(
    Lx: float,
    Ly: float,
    Lz: float,
    /,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
) -> GridSpec:
    """Factory for constructing grid specification

    In each direction, the size of the grid can be specified either by giving the number
    of voxels or the size of the voxels in this direction.

    Example::

        get_grid_spec(0.9, 0.7, 0.4, nx=8, ny=16, nz=8)
        get_grid_spec(0.9, 0.7, 0.4, dx=0.1, dy=0.2, dz=0.05)
        get_grid_spec(0.9, 0.7, 0.4, nx=8, dy=0.2, dz=0.05)

    Parameters
    ==========
    Lx :
        domain size :math:`L_x` in x-direction
    Ly :
        domain size :math:`L_y` in x-direction
    Lz :
        domain size :math:`L_z` in z-direction
    nx :
        number of voxels :math:`n_x` in x-direction
    ny :
        number of voxels :math:`n_y` in y-direction
    nz :
        number of voxels :math:`n_z` in z-direction
    dx :
        voxel size :math:`h_x` in x-direction
    dy :
        voxel size :math:`h_y` in y-direction
    dz :
        voxel size :math:`h_z` in z-direction

    Returns
    =======
    GridSpec :
        Specification of computational grid
    """

    def _valid(n: int | None, h: float | None) -> bool:
        return ((n is None) and isinstance(h, float)) or (
            isinstance(n, int) and (h is None)
        )

    assert _valid(nx, dx) and _valid(ny, dy) and _valid(nz, dz)
    _nx = typing.cast(int, nx if dx is None else int(Lx / dx))
    _ny = typing.cast(int, ny if dy is None else int(Ly / dy))
    _nz = typing.cast(int, nz if dz is None else int(Lz / dz))
    return GridSpec(Lx, Ly, Lz, _nx, _ny, _nz)
