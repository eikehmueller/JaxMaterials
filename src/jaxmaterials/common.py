"""Common definitions required throughout the code"""

__all__ = ["GridSpec"]


class GridSpec:
    """Specification of structured grid for the 3d domain :math:`[0,L_x] \\times [0,L_y] \\times [0,L_z]`

    The number of voxels in the three coordinate directions are :math:`n_x`, :math:`n_y`
    and :math:`n_z` respectively.
    """

    def __init__(self, nx, ny, nz, Lx, Ly, Lz):
        """Initialise instance

        Parameters
        ----------
        nx : int
            number of voxels :math:`n_x` in x-direction
        ny : int
            number of voxels :math:`n_y` in y-direction
        nz : int
            number of voxels :math:`n_z` in z-direction
        Lx : float
            domain size :math:`L_x` in x-direction
        Ly : float
            domain size :math:`L_y` in y-direction
        Lz : float
            domain size :math:`L_z` in z-direction
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

    @property
    def number_of_voxels(self):
        """Total number of voxels :math:`N = n_x\\cdot n_y \\cdot n_z`"""
        return self.nx * self.ny * self.nz

    @property
    def grid_spacings(self):
        """Tuple of grid spacings :math:`(h_x,h_y,h_z)` with :math:`h_i=L_i/n_i`"""
        return (self.Lx / self.nx, self.Ly / self.ny, self.Lz / self.nz)
