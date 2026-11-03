import numpy as np
from jax import numpy as jnp
import jax
from jax.experimental.multihost_utils import process_allgather
from jaxmaterials.common import get_grid_spec
from jaxmaterials.solver.fourier import get_xizero, sharded_fft3

jax.distributed.initialize()
jax.config.update("jax_enable_x64", True)

rank = jax.process_index()
if rank == 0:
    print(f"Running on {jax.process_count()} processes")


pencil_decomposition = True

if pencil_decomposition:
    grid_spec = get_grid_spec(nx=12, ny=12, nz=6, px=2, py=3)
    mesh = jax.make_mesh((2, 3), axis_names=("x", "y"))
    sharding = jax.sharding.NamedSharding(mesh, jax.P("x", "y", None))
else:
    grid_spec = get_grid_spec(nx=12, ny=12, nz=6, pz=6)
    mesh = jax.make_mesh((6,), axis_names=("z",))
    sharding = jax.sharding.NamedSharding(mesh, jax.P(None, None, "z"))

jax.set_mesh(mesh)

xi = get_xizero(grid_spec, dtype=jnp.float64)

print("xi : ", jax.typeof(xi))

rng = np.random.default_rng(seed=52523 + rank)
local_data = rng.normal(size=grid_spec.local_extents)
data = jax.make_array_from_process_local_data(
    sharding, local_data, global_shape=grid_spec.extents
)

if rank == 0:
    print(jax.typeof(data))


data_t = sharded_fft3(data, direction="forward")
global_data_t = process_allgather(data_t, tiled=True)

global_data = process_allgather(data, tiled=True)
print(global_data.shape)
_y = jnp.fft.fftn(global_data, axes=(2, 1, 0))
print(jnp.linalg.norm(global_data_t - _y) / jnp.linalg.norm(_y))

jax.distributed.shutdown()
