"""Utility functions for profiling and saving results to disk"""

import numpy as np
from contextlib import contextmanager
import time

__all__ = ["measure_time", "save_to_vtk"]


@contextmanager
def measure_time(label, repeat=1, warmup=False):
    """Context manager for measuring the time it takes to execute a block of code

    Parameters
    ==========
    label : str
        label for the time measurement
    repeat : int
        number of repetitions used for timing
    warmup_call : logical
        include a warmup call at the beginning which is not timed?
    """
    timings = []

    def run(func, *args, **kwargs):
        if warmup:
            func(*args, **kwargs)
        timings.append(time.perf_counter())
        for _ in range(repeat):
            result = func(*args, **kwargs)
        timings.append(time.perf_counter())
        return result

    yield run

    if timings:
        t_elapsed = (timings[1] - timings[0]) / repeat
        print(f"time [{label}] = {t_elapsed:8.3f} s")


def save_to_vtk(data, grid_spec, filename, location="centre"):
    """Save fields to VTK file

    Parameters
    ==========
    data : dict
        dictionary of the form ``{"label_1":field_1, "label_2":field_2, ...}`` where the labels
        are strings and each ``field_i`` is an array of shape ``(nx,ny,nz)``
    grid_spec : :py:class:`jaxmaterials.common.GridSpec`
        Specification of computational grid
    filename : str
        name of file to save to
    location : str
        location of data within voxel. Currently only "centre" is supported
    """
    assert location == "centre"
    shape = next(iter(data.values())).shape
    nx, ny, nz = shape
    with open(filename, mode="w", encoding="utf8") as f:
        print("# vtk DataFile Version 2.0", file=f)
        print("data", file=f)
        print("ASCII", file=f)
        print("DATASET RECTILINEAR_GRID", file=f)
        print(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}", file=f)
        for n, extent, dim_label in zip(
            shape, (grid_spec.Lx, grid_spec.Ly, grid_spec.Lz), "XYZ"
        ):
            print(f"{dim_label}_COORDINATES {n + 1} float", file=f)
            print(
                " ".join([f"{x:12.8f}" for x in np.linspace(0, extent, num=n + 1)]),
                file=f,
            )
        print("", file=f)
        print(f"CELL_DATA {nx * ny * nz}", file=f)
        for key, value in data.items():
            print(f"SCALARS {key} float 1", file=f)
            print("LOOKUP_TABLE default", file=f)
            print(
                "\n".join([f"{v:12.8f}" for v in value.flatten(order="F")]),
                file=f,
            )
