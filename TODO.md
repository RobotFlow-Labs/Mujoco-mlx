# MJX → MLX Port TODO

Port MuJoCo XLA (MJX) from JAX to Apple MLX. This creates the first MLX-native physics engine.

## Scope

MJX is a pure-Python GPU physics engine for MuJoCo. It uses JAX for GPU acceleration.
We replace JAX with MLX to run on Apple Silicon Metal GPUs.

**Source:** `mjx/mujoco/mjx/_src/` — 19,393 lines across ~30 core files
**Target:** `mjx/mujoco/mjx_mlx/_src/` — same API, MLX backend

## Translation Rules

| JAX | MLX |
|-----|-----|
| `import jax` | `import mlx.core as mx` |
| `import jax.numpy as jnp` | `import mlx.core as mx` |
| `jnp.array(...)` | `mx.array(...)` |
| `jnp.zeros(...)` | `mx.zeros(...)` |
| `jnp.ones(...)` | `mx.ones(...)` |
| `jnp.concatenate(...)` | `mx.concatenate(...)` |
| `jnp.stack(...)` | `mx.stack(...)` |
| `jnp.where(...)` | `mx.where(...)` |
| `jnp.dot(...)` | `mx.matmul(...)` or `mx.inner(...)` |
| `jnp.einsum(...)` | Manual implementation or `mx.einsum` if available |
| `jnp.linalg.norm(...)` | `mx.sqrt(mx.sum(mx.square(...)))` |
| `jnp.float32` | `mx.float32` |
| `jax.jit(fn)` | `mx.compile(fn)` |
| `jax.vmap(fn)` | Loop or manual batching (MLX has limited vmap) |
| `jax.lax.scan(...)` | Python loop with `mx.eval()` |
| `jax.lax.cond(...)` | `mx.where(...)` or Python if |
| `jax.lax.switch(...)` | Python if/elif chain |
| `jax.lax.fori_loop(...)` | Python for loop |
| `jax.lax.while_loop(...)` | Python while loop |
| `jax.lax.dynamic_slice(...)` | Array slicing |
| `jax.tree_util.tree_map(...)` | Manual or helper function |
| `@jax.custom_vjp` | Not needed (MLX has autograd) |
| `jax.random.PRNGKey(...)` | `mx.random.key(...)` |
| `jax.random.split(...)` | Manual key management |
| `functools.partial(jax.jit, ...)` | `mx.compile(...)` |
| `chex` assertions | Plain Python assertions |
| `flax.struct.dataclass` | `@dataclass` or custom |

## Key Differences to Handle

1. **vmap**: MLX doesn't have full vmap. Batch dimensions must be explicit.
2. **scan**: MLX doesn't have lax.scan. Use Python loops.
3. **Custom VJP**: MLX autograd handles most cases. Remove custom_vjp decorators.
4. **Tree utilities**: Replace jax.tree_util with manual traversal or a simple helper.
5. **Dynamic shapes**: MLX is less flexible with dynamic shapes. May need padding.
6. **In-place ops**: MLX arrays are immutable like JAX. No issue.
7. **Device placement**: MLX auto-places on GPU. Remove jax.devices() calls.

## Phase 0: Foundation
- [x] Fork repo, set remotes
- [x] Push to RobotFlow-Labs/Mujoco-mlx
- [ ] Create `mjx/mujoco/mjx_mlx/` package structure
- [ ] Port `_src/types.py` (data structures — no compute, just dataclasses)
- [ ] Port `_src/math.py` (linear algebra primitives)
- [ ] Port `_src/dataclasses.py` (MJX dataclass utilities)
- [ ] Port `_src/io.py` (model loading — bridges C MuJoCo to Python)
- [ ] Validate: types + math + io import and pass basic tests

## Phase 1: Core Dynamics
- [ ] Port `_src/support.py` (helper functions used everywhere)
- [ ] Port `_src/smooth.py` (smooth dynamics — kinematics, mass matrix)
- [ ] Port `_src/passive.py` (passive forces — gravity, springs)
- [ ] Port `_src/constraint.py` (constraint computation)
- [ ] Port `_src/solver.py` (constraint solver — CG, Newton)
- [ ] Port `_src/forward.py` (forward dynamics — the main loop)
- [ ] Port `_src/inverse.py` (inverse dynamics)
- [ ] Port `_src/derivative.py` (finite-difference derivatives)
- [ ] Validate: forward step on cartpole model matches C MuJoCo

## Phase 2: Collision
- [ ] Port `_src/collision_types.py` (collision data structures)
- [ ] Port `_src/collision_primitive.py` (sphere, capsule, box)
- [ ] Port `_src/collision_convex.py` (convex mesh collision)
- [ ] Port `_src/collision_sdf.py` (signed distance fields)
- [ ] Port `_src/collision_driver.py` (collision dispatch)
- [ ] Port `_src/bvh.py` (bounding volume hierarchy)
- [ ] Validate: collision detection matches C MuJoCo on test scenes

## Phase 3: Sensors and Rendering
- [ ] Port `_src/sensor.py` (sensor simulation)
- [ ] Port `_src/ray.py` (ray casting)
- [ ] Port `_src/mesh.py` (mesh utilities)
- [ ] Port `_src/render.py` (basic rendering)
- [ ] Port `_src/render_util.py` (rendering helpers)
- [ ] Port `_src/scan.py` (parallel scan operations)
- [ ] Validate: sensor outputs match C MuJoCo

## Phase 4: Tests and Benchmarks
- [ ] Port all `*_test.py` files from JAX assertions to MLX
- [ ] Run numerical comparison: MJX-MLX vs MJX-JAX vs C MuJoCo
- [ ] Benchmark: steps/second on Apple Silicon vs JAX on GPU
- [ ] Integration tests: cartpole, humanoid, ant

## Phase 5: Integration with IsaacLab-MLX
- [ ] Wire MJX-MLX as physics backend for IsaacLab mac-sim
- [ ] Replace analytical dynamics with MJX-MLX forward step
- [ ] Validate: IsaacLab cartpole task runs on MJX-MLX physics
- [ ] Performance comparison: analytical mac-sim vs MJX-MLX

## File-by-File Porting Checklist

| File | Lines | Phase | Status |
|------|-------|-------|--------|
| `types.py` | 685 | 0 | TODO |
| `math.py` | 430 | 0 | TODO |
| `dataclasses.py` | 280 | 0 | TODO |
| `io.py` | 890 | 0 | TODO |
| `support.py` | 1,200 | 1 | TODO |
| `smooth.py` | 2,100 | 1 | TODO |
| `passive.py` | 350 | 1 | TODO |
| `constraint.py` | 1,800 | 1 | TODO |
| `solver.py` | 1,500 | 1 | TODO |
| `forward.py` | 850 | 1 | TODO |
| `inverse.py` | 400 | 1 | TODO |
| `derivative.py` | 300 | 1 | TODO |
| `collision_types.py` | 200 | 2 | TODO |
| `collision_primitive.py` | 1,500 | 2 | TODO |
| `collision_convex.py` | 1,200 | 2 | TODO |
| `collision_sdf.py` | 800 | 2 | TODO |
| `collision_driver.py` | 600 | 2 | TODO |
| `bvh.py` | 500 | 2 | TODO |
| `sensor.py` | 900 | 3 | TODO |
| `ray.py` | 600 | 3 | TODO |
| `mesh.py` | 500 | 3 | TODO |
| `render.py` | 400 | 3 | TODO |
| `render_util.py` | 300 | 3 | TODO |
| `scan.py` | 200 | 3 | TODO |
| **TOTAL** | **~18,000** | | |
