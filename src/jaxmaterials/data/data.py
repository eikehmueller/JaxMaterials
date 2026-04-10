from jax import numpy as jnp
import numpy as np
import torch
import tqdm
import h5py
from jaxmaterials.data.distributions_fibres import (
    FibreDistribution2d,
)
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger_isotropic_cuda
from matplotlib import pyplot as plt

__all__ = [
    "IsotropicMaterialDataset",
    "LayeredFibresDatasetGenerator",
    "visualise_fibres",
]


class IsotropicMaterialDataset(torch.utils.data.Dataset):
    """Dataset as loaded from disk"""

    def __init__(self, filename, features_last=False):
        """Initialise dataset

        :arg filename: name of hdf5 file with data
        :arg features_last: Transpose array to return features in last dimension?
        """
        with h5py.File(filename, "r") as f:
            lame_parameters = f["base/lame_parameters"]
            self.number_of_cells = list(lame_parameters.shape[-3:])
            if features_last:
                lame_parameters = np.transpose(lame_parameters, axes=(0, 2, 3, 4, 1))
            self._lame_parameters = np.array(lame_parameters)
            self._sigma_bar = np.array(f["base/sigma_bar"])
            self.domain_size = np.array(f.attrs["domain_size"])
            self.n_samples = self._lame_parameters.shape[0]

    def __len__(self):
        """Number of samples"""
        return self.n_samples

    def __getitem__(self, idx):
        """Get data

        Return material properties (mu,lambda) and effective elasticity tensor C_{eff}.
        Depending on the ordering, the resulting tensors are of shape
        (N_x,N_y,N_z,2), (6,6) or (2,N_x,N_y,N_z), (6,6)

        :arg idx: index"""
        return self._lame_parameters[idx, ...], self._sigma_bar[idx, ...]

    @property
    def mean(self):
        """Compute pointwise mean"""
        return np.mean(self._lame_parameters, axis=(0,)), np.mean(
            self._sigma_bar, axis=(0,)
        )

    @property
    def std(self):
        """Compute pointwise standard deviation"""
        return np.std(self._lame_parameters, axis=(0,)), np.std(
            self._sigma_bar, axis=(0,)
        )


class LayeredFibresDatasetGenerator:
    """Dataset of material described by alternating layers of fibres

    The domain of size [L_x, L_y, L_z] is vertically divided into two parts:

    1. the regions [0,d_void] and [L_z-d_void,L_z] contain a uniform material
       with (mu_void, lambda_void)
    2. the region [d_void,L_z-d_void] contains n_layers layers of fibres
       alternatively oriented in the X- and Y-direction. The depths of these
       layers are random. The layers have Lame parameters
       (mu_fibre, lambda_fibre), with the surrounding material
       being characterised by (mu_material, lambda_material).

    The returned samples (X,y) describe the Lame-coefficients (in X) and
    resulting effective 6x6 material tensor C_{eff} (in y) defined by

        bar(sigma)_i = (C_{eff})_{ij} bar(epsilon)_j

    where the independent components of the strain epsilon and stress sigma
    are represented by six numbers each:

        (E_{00}, E_{11}, E_{22}, E_{23}, E_{13}, E_{12})

    Hence the shapes of (X,y) are:

    * Lame coefficients (mu,lambda) = X [2, N_x, N_y, N_z]
    * Effective elasticity tensor C_{eff} = y [6, 6]
    """

    def __init__(
        self,
        grid_spec,
        nlayers,
        d_void,
        n_samples,
        radius_distribution,
        volume_fraction=0.3,
        mu_fibre=1.0,
        mu_material=0.5,
        mu_void=0.1,
        lambda_fibre=0.1,
        lambda_material=0.1,
        lambda_void=0.01,
        rng=None,
        dtype=np.float64,
        verbose=False,
    ):
        """Initialise instance

        :arg grid_spec: specification of computational grid
        :arg radius_distribution: instance of FibreDistribution2d
        :arg n_layers: number of layers with fibres
        :arg d_void: thickness of empty layer at bottom and top of the domain
        :arg n_samples: number of samples in dataset
        :arg volume_fraction: fraction of volume occupied by the fibres
        :arg mu_fibre: Lame parameter in fibres
        :arg mu_material: Lame parameter in material
        :arg mu_void: Lame parameter in void
        :arg lambda_fibre: Lame parameter in fibres
        :arg lambda_material: Lame parameter in material
        :arg lambda_void: Lame parameter in void
        :arg dtype: Data type
        """
        super().__init__()
        # domain
        self.grid_spec = grid_spec
        self.nlayers = nlayers
        self.d_void = d_void
        # number of samples
        self.n_samples = n_samples
        # fibre radius distribution
        self.radius_distribution = radius_distribution
        self.volume_fraction = volume_fraction
        # material properties
        self.mu_fibre = mu_fibre
        self.mu_material = mu_material
        self.mu_void = mu_void
        self.lambda_fibre = lambda_fibre
        self.lambda_material = lambda_material
        self.lambda_void = lambda_void
        # random number generator and data type
        self.rng = np.random.default_rng(seed=64217) if rng is None else rng
        self.dtype = dtype
        self.verbose = verbose

    def generate(self):
        # Generate data
        self._lame_parameters = np.empty(
            shape=(
                self.n_samples,
                2,
                self.grid_spec.nx,
                self.grid_spec.ny,
                self.grid_spec.nz,
            ),
            dtype=self.dtype,
        )
        self._sigma_bar = np.empty(shape=(self.n_samples, 6, 6), dtype=self.dtype)
        generator_range = range(self.n_samples)

        if self.verbose:
            print("Generating data...")
            generator_range = tqdm.tqdm(generator_range)
        for j in generator_range:
            _, fibre_positions, fibre_radii, fibre_orientations = (
                self.generate_fibre_positions()
            )
            self._lame_parameters[j, ...] = self.material_properties(
                fibre_positions, fibre_radii, fibre_orientations
            )

            mu, lmbda = (
                self._lame_parameters[j, 0, ...],
                self._lame_parameters[j, 1, ...],
            )
            for k in range(6):
                E_mean = np.zeros(shape=(6,), dtype=self.dtype)
                E_mean[k] = 1.0
                rtol = 1.0e-4
                atol = 1.0e-4
                epsilon, sigma, iter = lippmann_schwinger_isotropic_cuda(
                    lmbda, mu, E_mean, self.grid_spec, rtol=rtol, atol=atol, maxiter=100
                )
                self._sigma_bar[j, k, :] = jnp.mean(sigma, axis=(1, 2, 3))

    def generate_fibre_positions(self):
        """Generate fibre positions and radii in the centre of the domain

        returns the following information:

            * a list [z_0=d_void,z_1,z_2,...,z_{nlayers}=L_z-d_void] of length n_layers+1
              with the boundaries of the layers
            * a list of length n_layers, where each entry is an array of shape (n_j,2) with the
              positions of the n_j fibres in layer j, projected onto the XZ or YZ planes,
              depending on the orientation.
            * a list of length n_layers, where each entry is an array of shape (n_j,) with the
              radii of the n_j fibres in layer j.
            * a list of length n_layers, where each entry describes the orientation (0 or 1)

        """
        layer_depths = np.asarray([0.5 * self.radius_distribution.r_max])
        while np.any(layer_depths < 3 * self.radius_distribution.r_max):
            layer_boundaries = np.asarray(
                sorted(
                    [self.d_void, self.grid_spec.Lz - self.d_void]
                    + [
                        float(z)
                        for z in self.rng.uniform(
                            size=self.nlayers - 1,
                            low=self.d_void,
                            high=self.grid_spec.Lz - self.d_void,
                        )
                    ]
                )
            )
            layer_depths = layer_boundaries[1:] - layer_boundaries[:-1]
        offset = self.rng.integers(low=0, high=2, size=1)[0]
        fibre_orientations = []
        fibre_positions = []
        fibre_radii = []
        domain_size = [self.grid_spec.Lx, self.grid_spec.Ly, self.grid_spec.Lz]
        for k in range(self.nlayers):
            orientation = (k + offset) % 2
            fibre_orientations.append(orientation)
            seed = self.rng.integers(low=0, high=2, size=1)[0]
            fibre_distribution_2d = FibreDistribution2d(
                domain_size=[
                    domain_size[orientation],
                    layer_depths[k] - 2 * self.radius_distribution.r_max,
                ],
                volume_fraction=self.volume_fraction,
                r_fibre_dist=self.radius_distribution,
                seed=seed,
                fast_code=True,
            )
            X, R = next(iter(fibre_distribution_2d))
            lower_boundary_indices = (
                np.less_equal(X[..., 0], self.radius_distribution.r_max),
            )
            upper_boundary_indices = np.greater_equal(
                X[..., 0],
                domain_size[orientation] - self.radius_distribution.r_max,
            )
            X = np.concatenate(
                [
                    X,
                    X[lower_boundary_indices]
                    + np.asarray([domain_size[orientation], 0]),
                    X[upper_boundary_indices]
                    - np.asarray([domain_size[orientation], 0]),
                ],
                axis=0,
            )
            R = np.concatenate(
                [R, R[lower_boundary_indices], R[upper_boundary_indices]], axis=0
            )
            vertical_offset = (
                np.asarray([0, layer_boundaries[k] + self.radius_distribution.r_max]),
            )
            X += vertical_offset
            fibre_positions.append(X)
            fibre_radii.append(R)
        return layer_boundaries, fibre_positions, fibre_radii, fibre_orientations

    def material_properties(self, fibre_positions, fibre_radii, fibre_orientations):
        """Construct fields with Lame parameters (mu, lambda) throughout domain

        Returns array of shape (2,N_x,N_y,N_z) with the two Lame parameters mu, lambda at the
        voxel centres.

        The input is usually generated by the generate_fibre_positions() method.

        :arg fibre_positions: list with arrays of fibre positions in each vertical layer
        :arg fibre_radii: list with arrays of fibre radii in each vertical layer
        :arg fibre_orientations: orientations of fibres
        """
        X, Y, Z = np.meshgrid(
            self.grid_spec.Lx
            / self.grid_spec.nx
            * (1 / 2 + np.arange(self.grid_spec.nx)),
            self.grid_spec.Ly
            / self.grid_spec.ny
            * (1 / 2 + np.arange(self.grid_spec.ny)),
            self.grid_spec.Lz
            / self.grid_spec.nz
            * (1 / 2 + np.arange(self.grid_spec.nz)),
            indexing="ij",
        )
        data = np.zeros(
            shape=(2, self.grid_spec.nx, self.grid_spec.ny, self.grid_spec.nz),
            dtype=self.dtype,
        )
        for j, (v_void, v_fibre, v_material) in enumerate(
            zip(
                [self.mu_void, self.lambda_void],
                [self.mu_fibre, self.lambda_fibre],
                [self.mu_material, self.lambda_material],
            )
        ):
            data[j, ...] = v_material
            data[j, ...] += (
                (Z < self.d_void) + (Z > self.grid_spec.Lz - self.d_void)
            ) * (v_void - v_material)
            for fibre_position, fibre_radius, orientation in zip(
                fibre_positions, fibre_radii, fibre_orientations
            ):
                coord = Y if orientation else X
                for p, r in zip(fibre_position, fibre_radius):
                    data[j, ...] += ((coord - p[0]) ** 2 + (Z - p[1]) ** 2 < r**2) * (
                        v_fibre - v_material
                    )
        return np.asarray(data)

    def save_hdf5(self, filename):
        """Save the dataset to disk

        :arg filename: name of file to save to"""
        assert hasattr(self, "_lame_parameters") and hasattr(self, "_sigma_bar"), (
            "Data not not generated yet, call generate() first."
        )
        with h5py.File(filename, "w") as f:
            group = f.create_group("base")
            group.create_dataset("lame_parameters", data=self._lame_parameters)
            group.create_dataset("sigma_bar", data=self._sigma_bar)
            f.attrs["domain_size"] = [
                self.grid_spec.Lx,
                self.grid_spec.Ly,
                self.grid_spec.Lz,
            ]


def visualise_fibres(
    grid_spec,
    layer_boundaries,
    fibre_positions,
    fibre_radii,
    fibre_orientations,
    filename="fibres.pdf",
):
    """Auxilliary method for visualising the location of the fibres in 2d

    Generate plot of the projection of the fibres onto the XZ and YZ plane.
    The input is usually created by the generate_fibre_positions() method.

    :arg grid_spec: specification of grid
    :arg layer_boundaries: the locations of the boundaries between layers
    :arg fibre_positions: list with arrays of fibre positions in each vertical layer
    :arg fibre_radii: list with arrays of fibre radii in each vertical layer
    :arg fibre_orientations: orientations of fibres
    """
    horizontal_size = max(grid_spec.Lx, grid_spec.Ly)
    layer_depths = layer_boundaries[1:] - layer_boundaries[:-1]
    d_void = layer_boundaries[0]
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0, horizontal_size)
    ax.set_ylim(-0.02 / 2, grid_spec.Lz + 0.02 / 2)
    ax.set_aspect("equal")

    ax.add_patch(
        plt.Rectangle(
            xy=[0, 0],
            width=grid_spec.Lx,
            height=d_void,
            color="gray",
            linewidth=0,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            xy=[0, grid_spec.Lz - d_void],
            width=grid_spec.Lx,
            height=d_void,
            color="gray",
            linewidth=0,
        )
    )
    domain_size = [grid_spec.Lx, grid_spec.Ly, grid_spec.Lz]
    for X, R, orientation, depth, boundary in zip(
        fibre_positions,
        fibre_radii,
        fibre_orientations,
        layer_depths,
        layer_boundaries,
    ):
        color = "red" if orientation else "blue"
        for p, r in zip(X, R):
            ax.add_patch(
                plt.Circle(
                    p,
                    r,
                    color=color,
                    linewidth=0,
                )
            )
        ax.add_patch(
            plt.Rectangle(
                xy=[0, boundary],
                width=domain_size[orientation],
                height=depth,
                color=color,
                alpha=0.5,
                linewidth=0,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                xy=[domain_size[orientation], boundary],
                width=horizontal_size - domain_size[orientation],
                height=depth,
                color="white",
                linewidth=0,
            )
        )
    plt.savefig(filename, bbox_inches="tight")
