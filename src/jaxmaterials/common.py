"""Common definitions required throughout the code"""

import numpy as np

__all__ = ["GridSpec", "get_grid_spec"]


class GridSpec:
    """Specification of structured grid for the 3d domain :math:`[0,L_x] \\times [0,L_y] \\times [0,L_z]`

    The number of voxels in the three coordinate directions are :math:`n_x`, :math:`n_y`
    and :math:`n_z` respectively.
    """

    def __init__(self, Lx, Ly, Lz, nx, ny, nz):
        """Initialise instance

        Parameters
        ----------
        Lx : float
            domain size :math:`L_x` in x-direction
        Ly : float
            domain size :math:`L_y` in y-direction
        Lz : float
            domain size :math:`L_z` in z-direction
        nx : int
            number of voxels :math:`n_x` in x-direction
        ny : int
            number of voxels :math:`n_y` in y-direction
        nz : int
            number of voxels :math:`n_z` in z-direction
        """
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

    @property
    def number_of_voxels(self):
        """Total number of voxels :math:`N = n_x\\cdot n_y \\cdot n_z`"""
        return self._nx * self._ny * self._nz

    @property
    def Lx(self):
        """Size of grid in x-direction"""
        return self._Lx

    @property
    def Ly(self):
        """Size of grid in y-direction"""
        return self._Ly

    @property
    def Lz(self):
        """Size of grid in z-direction"""
        return self._Lz

    @property
    def nx(self):
        """Number of voxels in x-direction"""
        return self._nx

    @property
    def ny(self):
        """Number of voxels in y-direction"""
        return self._ny

    @property
    def nz(self):
        """Number of voxels in z-direction"""
        return self._nz

    @property
    def dx(self):
        """Extent :math:`h_x` of voxels in x-direction"""
        return self._Lx / self._nx

    @property
    def dy(self):
        """Extent :math:`h_y` of voxels in y-direction"""
        return self._Ly / self._ny

    @property
    def dz(self):
        """Extent :math:`h_z` of voxels in z-direction"""
        return self._Lz / self._nz

    @property
    def grid_spacings(self):
        """Tuple of grid spacings :math:`(h_x,h_y,h_z)` with :math:`h_i=L_i/n_i`"""
        return (self.dx, self.dy, self.dz)

    @property
    def voxel_centers(self):
        """Array with voxel centres"""
        x = np.linspace(self.dx / 2, self._Lx - self.dx / 2, self._nx)
        y = np.linspace(self.dy / 2, self._Ly - self.dy / 2, self._ny)
        z = np.linspace(self.dz / 2, self._Lz - self.dz / 2, self._nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)


def get_grid_spec(Lx, Ly, Lz, /, nx=None, ny=None, nz=None, dx=None, dy=None, dz=None):
    """Factory for constructing grid specification

    In each direction, the size of the grid can be specified either by giving the number
    of voxels or the size of the voxels in this direction.

    Parameters
    ==========
    Lx : float
        domain size :math:`L_x` in x-direction
    Ly : float
        domain size :math:`L_y` in x-direction
    Lz : float
        domain size :math:`L_z` in z-direction
    nx : int
        number of voxels :math:`n_x` in x-direction
    ny : int
        number of voxels :math:`n_y` in y-direction
    nz : int
        number of voxels :math:`n_z` in z-direction
    dx : int
        voxel size :math:`h_x` in x-direction
    dy : int
        voxel size :math:`h_y` in y-direction
    dz : int
        voxel size :math:`h_z` in z-direction

    Returns
    =======
    Instance of :py:class:`jaxmaterials.common.GridSpec`
    """

    def _valid(n, h):
        return ((n is None) and isinstance(h, float)) or (
            isinstance(n, int) and (h is None)
        )

    assert _valid(nx, dx) and _valid(ny, dy) and _valid(nz, dz)
    _nx = nx if dx is None else int(Lx / dx)
    _ny = ny if dy is None else int(Ly / dy)
    _nz = nz if dz is None else int(Lz / dz)
    return GridSpec(Lx, Ly, Lz, _nx, _ny, _nz)
