"""Utility functions for profiling and saving results to disk"""

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import numpy as np

from jaxmaterials.common import GridSpec

__all__ = ["measure_time", "save_to_vtk"]


@contextmanager
def measure_time(
    label: str, repeat: int = 1, warmup: bool = False
) -> Generator[tuple[Callable[..., Any], Callable[[], int]]]:
    """Context manager for measuring the time it takes to execute a block of code

    Parameters
    ==========
    label :
        label for the time measurement
    repeat :
        number of repetitions used for timing
    warmup :
        include a warmup call at the beginning which is not timed?

    Yields
    ======
    Tuple of callable wrapper around code and function for returning the number of iterations
    """
    timings = []
    it = [0]

    def get_iter() -> int:
        return it[0]

    def run(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if warmup:
            func(*args, **kwargs)
        timings.append(time.perf_counter())
        for _ in range(repeat):
            it[0] += 1
            result = func(*args, **kwargs)
        timings.append(time.perf_counter())
        return result

    yield run, get_iter

    if timings:
        t_elapsed = (timings[1] - timings[0]) / repeat
        print(f"time [{label}] = {t_elapsed:8.3f} s")


def save_to_vtk(
    data: dict[str, np.ndarray],
    grid_spec: GridSpec,
    filename: str,
    location: str = "centre",
) -> None:
    """Save fields to VTK file

    Parameters
    ==========
    data :
        dictionary of the form ``{"label_1":field_1, "label_2":field_2, ...}`` where the labels
        are strings and each ``field_i`` is an array of shape ``(nx,ny,nz)``
    grid_spec :
        Specification of computational grid
    filename :
        name of file to save to
    location :
        location of data within voxel. Currently only ``"centre"`` is supported
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
        print(file=f)
        print(f"CELL_DATA {nx * ny * nz}", file=f)
        for key, value in data.items():
            print(f"SCALARS {key} float 1", file=f)
            print("LOOKUP_TABLE default", file=f)
            print(
                "\n".join([f"{v:12.8f}" for v in value.flatten(order="F")]),
                file=f,
            )
