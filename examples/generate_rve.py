from pathlib import Path
import numpy as np
from jaxmaterials.common import GridSpec

from jaxpeds.data.distributions_fibres import (
    FibreRadiusDistribution,
    FibreDistribution2d,
)
import json
from collections.abc import Sequence


def saveVolvtk_amitex(
    volume: np.ndarray,
    filename: str | Path,
    dtype: str = "uint8",
    choice: str = "cell",
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> None:
    """Write a legacy binary STRUCTURED_POINTS VTK volume for AMITEX.

    This function was written by Yang Chen, University of Bath
    """
    type_map = {
        "uint8": ("unsigned_char", ">u1"),
        "int8": ("char", ">i1"),
        "uint16": ("unsigned_short", ">u2"),
        "int16": ("short", ">i2"),
        "uint32": ("unsigned_int", ">u4"),
        "int32": ("int", ">i4"),
        "uint64": ("unsigned_long", ">u8"),
        "int64": ("long", ">i8"),
        "single": ("float", ">f4"),
        "double": ("double", ">f8"),
    }
    data = np.asarray(volume)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3 or dtype not in type_map or choice not in {"cell", "point"}:
        raise ValueError(
            "Expected a 3-D volume, supported VTK dtype, and cell or point data"
        )
    vtk_type, numpy_type = type_map[dtype]
    nx, ny, nz = data.shape
    dimensions = (nx + 1, ny + 1, nz + 1) if choice == "cell" else (nx, ny, nz)
    data_count = nx * ny * nz
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        header = (
            "# vtk DataFile Version 4.2\n"
            "mesh_grid\n"
            "BINARY\n"
            "DATASET STRUCTURED_POINTS\n"
            f"DIMENSIONS {dimensions[0]} {dimensions[1]} {dimensions[2]}\n"
            "ORIGIN 0 0 0\n"
            f"SPACING {float(spacing[0]):e} {float(spacing[1]):e} {float(spacing[2]):e}\n"
            f"{'CELL_DATA' if choice == 'cell' else 'POINT_DATA'} {data_count}\n"
            f"SCALARS data {vtk_type}\n"
            "LOOKUP_TABLE default\n"
        )
        stream.write(header.encode("ascii"))
        data.ravel(order="F").astype(numpy_type, copy=False).tofile(stream)


def generate_fibre_distribution(filename):
    file_path = Path(filename)
    if not file_path.exists():
        scale = 4
        r_avg = scale * 7.5e-3
        r_min = scale * 5.0e-3
        r_max = scale * 10.0e-3
        sigma = scale * 0.5e-3
        volume_fraction = 0.3
        radius_distribution = FibreRadiusDistribution(
            r_avg=r_avg, r_min=r_min, r_max=r_max, sigma=sigma, gaussian=True
        )
        fibre_distribution_2d = FibreDistribution2d(
            domain_size=[grid_spec.Lx, grid_spec.Ly],
            volume_fraction=volume_fraction,
            r_fibre_dist=radius_distribution,
            seed=1874183,
            fast_code=True,
        )

        fibre_positions, fibre_radii = next(iter(fibre_distribution_2d))
        with open(filename, "w", encoding="utf8") as f:
            json.dump(
                {
                    "positions_x": list(fibre_positions[:, 0]),
                    "positions_y": list(fibre_positions[:, 1]),
                    "radii": list(fibre_radii),
                },
                f,
                indent=4,
            )


# Domain
Lx = 1.0
Ly = 1.0
Lz = 0.5

nx = 1024
ny = 1024
nz = 512


grid_spec = GridSpec(Lx, Ly, Lz, nx=nx, ny=ny, nz=nz)

filename = "fibre_locations.json"

generate_fibre_distribution(filename)

with open(filename, encoding="utf8") as f:
    fibre_locations = json.load(f)
    fibre_positions = np.stack(
        [fibre_locations["positions_x"], fibre_locations["positions_y"]], axis=1
    )
    fibre_radii = fibre_locations["radii"]

voxel_centres = np.meshgrid(
    grid_spec.dx * (1 / 2 + np.arange(grid_spec.nx)),
    grid_spec.dy * (1 / 2 + np.arange(grid_spec.ny)),
    indexing="ij",
)


X, Y = voxel_centres
mask_2d = np.ones(shape=(grid_spec.nx, grid_spec.ny), dtype=np.uint8)

offset_X = [-grid_spec.Lx, 0, +grid_spec.Lx]
offset_Y = [-grid_spec.Ly, 0, +grid_spec.Ly]
# Ensure periodicity
for p, r in zip(fibre_positions, fibre_radii):
    for ox in offset_X:
        for oy in offset_Y:
            mask_2d[(X + ox - p[0]) ** 2 + (Y + oy - p[1]) ** 2 < r**2] = 2


data = np.repeat(mask_2d[:, :, None], grid_spec.nz, axis=2)

saveVolvtk_amitex(data, "rve.vtk", spacing=grid_spec.grid_spacings)
