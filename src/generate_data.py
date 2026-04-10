"""Main script for visualising layered fibre dataset"""

import numpy as np
from jaxmaterials.common import GridSpec
from jaxmaterials.data.distributions_fibres import FibreRadiusDistribution
from jaxmaterials.data.data import (
    IsotropicMaterialDataset,
    LayeredFibresDatasetGenerator,
    visualise_fibres,
)
from jaxmaterials.utilities import save_to_vtk
from matplotlib import pyplot as plt

# Domain
Lx = 0.5
Ly = 0.5
Lz = 0.2
d_void = 0.01
N = 32
nx = 5 * N
ny = 5 * N
nz = 2 * N
grid_spec = GridSpec(nx, ny, nz, Lx, Ly, Lz)

# Distribution of fibre radii
r_avg = 7.5e-3
r_min = 5.0e-3
r_max = 10.0e-3
sigma = 0.5e-3
volume_fraction = 0.3
radius_distribution = FibreRadiusDistribution(
    r_avg=r_avg, r_min=r_min, r_max=r_max, sigma=sigma, gaussian=True
)
# number of layers with fibres
nlayers = 3

n_samples = 256
rng = np.random.default_rng(seed=5713853)

dataset_generator = LayeredFibresDatasetGenerator(
    grid_spec,
    nlayers,
    d_void,
    n_samples,
    radius_distribution,
    volume_fraction=volume_fraction,
    mu_fibre=0.5,
    mu_material=1.0,
    mu_void=0.1,
    lambda_fibre=0.1,
    lambda_material=0.5,
    lambda_void=1.0,
    dtype=np.float32,
    rng=rng,
    verbose=True,
)

# Visualise the projected fibre locations for samples
visualise = False
if visualise:
    for k in range(8):
        layer_boundaries, fibre_positions, fibre_radii, fibre_orientations = (
            dataset_generator.generate_fibre_positions()
        )
        visualise_fibres(
            grid_spec,
            layer_boundaries,
            fibre_positions,
            fibre_radii,
            fibre_orientations,
            filename=f"fibres_{k + 1:03d}.pdf",
        )

filename = "fibres.h5"
dataset_generator.generate()
dataset_generator.save_hdf5(filename)
dataset = IsotropicMaterialDataset(filename)

# nsamples = len(dataset)
# sigma = np.empty((nsamples, 6, 6))
# for k in range(nsamples):
#    _, sigma_bar = dataset[k]
#    sigma[k, :, :] = sigma_bar

# fig, axs = plt.subplots(6, 6, figsize=(20, 20))
# for i in range(6):
#    for j in range(6):
#        axs[i, j].plot(sigma[:, i, j])
# plt.savefig("sigma_bar.pdf", bbox_inches="tight")

# Save gridded material properties to vtk files
savetovtk = False
if savetovtk:
    for k in range(8):
        data, sigma_bar = dataset[k]

        save_to_vtk(
            {"mu": data[0], "lambda": data[1]},
            grid_spec,
            filename=f"data_{k + 1:03d}.vtk",
        )
