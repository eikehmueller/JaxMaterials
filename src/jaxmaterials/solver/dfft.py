"""Distributed 3D FFT with slab or pencil decomposition (2DECOMP-style).

The XLA partitioner does not distribute multi-dimensional FFTs: it all-gathers the
full array onto every device. Inside :func:`distributed_fft` the functions
:func:`fftn3` and :func:`ifftn3` instead use the transpose algorithm, with local
1D/2D FFTs and ``all_to_all`` transposes, on arrays of shape ``(..., nx, ny, nz)``.

Slab (one mesh axis ``a``): input sharded ``(x: a)``::

    FFT(y, z) -> all_to_all(a) -> FFT(x)          output sharded ``(y: a)``

Pencil (two mesh axes ``a``, ``b``): input sharded ``(x: a, y: b)``::

    FFT(z) -> all_to_all(b) -> FFT(y) -> all_to_all(a) -> FFT(x)
                                                  output sharded ``(y: a, z: b)``

The outputs have the same global shape and element order as ``jnp.fft.fftn``,
only the layout differs; elementwise operations in Fourier space (Green
operators) need no changes since XLA aligns them with that layout. The inverse
transform accepts the forward output layout and returns the input layout.
Outside the context the functions are plain ``jnp.fft`` calls.

Divisibility: slab needs ``nx % pa == 0`` and ``ny % pa == 0``; pencil needs
``nx % pa == 0``, ``ny % pa == 0``, ``ny % pb == 0`` and ``nz % pb == 0``.
"""

import contextlib
import threading

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map as _shard_map
except ImportError:  # older JAX
    from jax.experimental.shard_map import shard_map as _shard_map

_state = threading.local()


@contextlib.contextmanager
def distributed_fft(mesh, axes):
    """Use distributed FFTs on the three trailing (grid) axes.

    :arg mesh: ``jax.sharding.Mesh`` (``None`` disables the distributed FFT)
    :arg axes: mesh axis name(s): ``"a"`` or ``("a",)`` for a slab decomposition
        of grid axis 0, ``("a", "b")`` for a pencil decomposition of grid axes 0, 1

    The setting is read at trace time, so enter the context inside the traced
    function (or around the first call of a jitted function).
    """
    if isinstance(axes, str):
        axes = (axes,)
    enabled = mesh is not None and axes is not None and len(axes) > 0
    if enabled and len(axes) > 2:
        raise ValueError("at most two mesh axes (slab or pencil) are supported")
    previous = getattr(_state, "config", None)
    _state.config = (mesh, tuple(axes)) if enabled else None
    # Entering the abstract mesh also changes JAX's trace-context cache key, so
    # inner jitted functions are retraced rather than reusing a layout-agnostic trace.
    try:
        with (
            jax.sharding.use_abstract_mesh(mesh.abstract_mesh)
            if enabled
            else contextlib.nullcontext()
        ):
            yield
    finally:
        _state.config = previous


def _config():
    return getattr(_state, "config", None)


def _spec(nd, assignment):
    """PartitionSpec from {array axis counted from the end: mesh axis name}."""
    spec = [None] * nd
    for axis, name in assignment.items():
        spec[nd + axis] = name
    return P(*spec)


def fftn3(a):
    """FFT over the last three axes."""
    config = _config()
    if config is None:
        return jnp.fft.fftn(a, axes=(-3, -2, -1))
    mesh, names = config
    nd = a.ndim
    x_, y_, z_ = nd - 3, nd - 2, nd - 1

    if len(names) == 1:
        (na,) = names
        in_spec, out_spec = _spec(nd, {-3: na}), _spec(nd, {-2: na})

        def local(x):
            x = jnp.fft.fftn(x, axes=(y_, z_))
            x = jax.lax.all_to_all(x, na, y_, x_, tiled=True)
            return jnp.fft.fft(x, axis=x_)

    else:
        na, nb = names
        in_spec = _spec(nd, {-3: na, -2: nb})
        out_spec = _spec(nd, {-2: na, -1: nb})

        def local(x):
            x = jnp.fft.fft(x, axis=z_)
            x = jax.lax.all_to_all(x, nb, z_, y_, tiled=True)
            x = jnp.fft.fft(x, axis=y_)
            x = jax.lax.all_to_all(x, na, y_, x_, tiled=True)
            return jnp.fft.fft(x, axis=x_)

    return _shard_map(local, mesh=mesh, in_specs=in_spec, out_specs=out_spec)(a)


def ifftn3(a):
    """Inverse of :func:`fftn3` (input in its output layout)."""
    config = _config()
    if config is None:
        return jnp.fft.ifftn(a, axes=(-3, -2, -1))
    mesh, names = config
    nd = a.ndim
    x_, y_, z_ = nd - 3, nd - 2, nd - 1

    if len(names) == 1:
        (na,) = names
        in_spec, out_spec = _spec(nd, {-2: na}), _spec(nd, {-3: na})

        def local(x):
            x = jnp.fft.ifft(x, axis=x_)
            x = jax.lax.all_to_all(x, na, x_, y_, tiled=True)
            return jnp.fft.ifftn(x, axes=(y_, z_))

    else:
        na, nb = names
        in_spec = _spec(nd, {-2: na, -1: nb})
        out_spec = _spec(nd, {-3: na, -2: nb})

        def local(x):
            x = jnp.fft.ifft(x, axis=x_)
            x = jax.lax.all_to_all(x, na, x_, y_, tiled=True)
            x = jnp.fft.ifft(x, axis=y_)
            x = jax.lax.all_to_all(x, nb, y_, z_, tiled=True)
            return jnp.fft.ifft(x, axis=z_)

    return _shard_map(local, mesh=mesh, in_specs=in_spec, out_specs=out_spec)(a)
