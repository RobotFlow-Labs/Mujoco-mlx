# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Collision functions for shapes represented as signed distance functions (SDF).

A signed distance function at a given point in space is the shortest distance to
a surface. This enables to define a geometry implicitly and exactly.

See https://iquilezles.org/articles/distfunctions/ for a list of analytic SDFs.
"""

import functools
from typing import Callable
from typing import Tuple

import mlx.core as mx
from mujoco.mjx_mlx._src import math
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.collision_types import Collision
from mujoco.mjx_mlx._src.collision_types import GeomInfo
from mujoco.mjx_mlx._src.dataclasses import PyTreeNode
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DataMLX
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import ModelMLX
# pylint: enable=g-importing-member

# the SDF function takes position in, and returns a distance or objective
SDFFn = Callable[[mx.array], mx.array]

def collider(ncon: int):
  """Wraps collision functions for use by collision_driver."""

  def wrapper(func):
    def collide(m: Model, d: Data, _, geom: mx.array) -> Collision:
      g1, g2 = geom.T
      info1 = GeomInfo(d.geom_xpos[g1], d.geom_xmat[g1], m.geom_size[g1])
      info2 = GeomInfo(d.geom_xpos[g2], d.geom_xmat[g2], m.geom_size[g2])
      dist, pos, frame = _vmap(func)(info1, info2)
      if ncon > 1:
        return _tree_map(mx.concatenate, (dist, pos, frame))
      return dist, pos, frame

    collide.ncon = ncon
    return collide

  return wrapper

def _plane(pos: mx.array, size: mx.array) -> mx.array:
  del size
  return pos[2]

def _sphere(pos: mx.array, size: mx.array):
  return math.norm(pos) - size[0]

def _capsule(pos: mx.array, size: mx.array):
  pa = -size[1] * mx.array([0, 0, 1])
  pb = size[1] * mx.array([0, 0, 1])
  ab = pb - pa
  ap = pos - pa
  denom = ab.dot(ab)
  denom = mx.where(mx.abs(denom) < 1e-12, 1e-12 * math.sign(denom), denom)
  t = ab.dot(ap) / denom
  t = mx.clip(t, 0, 1)
  c = pa + t * ab
  return math.norm(pos - c) - size[0]

def _ellipsoid(pos: mx.array, size: mx.array) -> mx.array:
  k0 = math.norm(pos / size)
  k1 = math.norm(pos / (size * size))
  return k0 * (k0 - 1.0) / (k1 + (k1 == 0.0) * 1e-12)

@jax.custom_jvp
def _cylinder(pos: mx.array, size: mx.array) -> mx.array:
  a0 = mx.sqrt(pos[0] * pos[0] + pos[1] * pos[1]) - size[0]
  a1 = mx.abs(pos[2]) - size[1]
  b0 = mx.maximum(a0, 0)
  b1 = mx.maximum(a1, 0)
  return mx.minimum(mx.maximum(a0, a1), 0) + mx.sqrt(b0 * b0 + b1 * b1)

def _cylinder_grad(x: mx.array, size: mx.array) -> mx.array:
  """Gradient of the cylinder SDF wrt query point and singularities removed."""
  c = mx.sqrt(x[0] * x[0] + x[1] * x[1])
  e = mx.abs(x[2])
  a = mx.array([c - size[0], e - size[1]])
  b = mx.array([mx.maximum(a[0], 0), mx.maximum(a[1], 0)])
  j = mx.argmax(a)
  bnorm = mx.sqrt(b[0] * b[0] + b[1] * b[1])
  bnorm += mx.allclose(bnorm, 0) * 1e-12
  grada = mx.array([
      x[0] / (c + mx.allclose(c, 0) * 1e-12),
      x[1] / (c + mx.allclose(c, 0) * 1e-12),
      x[2] / (e + mx.allclose(e, 0) * 1e-12),
  ])
  gradm = mx.array([[grada[0], grada[1], 0], [0, 0, grada[2]]])
  gradb = grada * b[mx.array([0, 0, 1])] / bnorm
  return mx.where(a[j] < 0, gradm[j], gradb)

@_cylinder.defjvp
def cylinder_jvp(primals, tangents):
  x, y = primals
  x_dot, _ = tangents
  primal_out = _cylinder(x, y)
  tangent_out = mx.dot(_cylinder_grad(x, y), x_dot)
  return primal_out, tangent_out

def _from_to(
    f: SDFFn,
    from_pos: mx.array,
    from_mat: mx.array,
    to_pos: mx.array,
    to_mat: mx.array,
) -> SDFFn:
  relmat = math.matmul_unroll(to_mat.T, from_mat)
  relpos = to_mat.T @ (from_pos - to_pos)
  return lambda p: f(relmat @ p + relpos)

def _intersect(d1: SDFFn, d2: SDFFn) -> SDFFn:
  return lambda p: mx.maximum(d1(p), d2(p))

def _clearance(d1: SDFFn, d2: SDFFn) -> SDFFn:
  return lambda p: (d1(p) + d2(p) + mx.abs(_intersect(d1, d2)(p))).squeeze()

class GradientState(PyTreeNode):
  dist: mx.array
  x: mx.array

def _gradient_step(objective: SDFFn, state: GradientState) -> GradientState:
  """Performs a step of gradient descent."""
  # TODO: find better parameters
  amin = 1e-4  # minimum value for line search factor scaling the gradient
  amax = 2.0  # maximum value for line search factor scaling the gradient
  nlinesearch = 10  # line search points
  grad = jax.grad(objective)(state.x)
  alpha = mx.geomspace(amin, amax, nlinesearch).reshape(nlinesearch, -1)
  candidates = state.x - alpha * grad.reshape(-1, 3)
  values = _vmap(objective)(candidates)
  idx = mx.argmin(values)
  return state.replace(x=candidates[idx], dist=values[idx])

def _gradient_descent(
    objective: SDFFn,
    x: mx.array,
    niter: int,
) -> Tuple[mx.array, mx.array]:
  """Performs gradient descent with backtracking line search."""
  state = GradientState(
      dist=1e10,
      x=x,
  )

  state, _ = jax.lax.scan(
      lambda s, _: (_gradient_step(objective, s), None), state, (), length=niter
  )
  return state.dist, state.x

def _optim(
    d1,
    d2,
    info1: GeomInfo,
    info2: GeomInfo,
    x0: mx.array,
) -> Collision:
  """Optimizes the clearance function."""
  d1 = functools.partial(d1, size=info1.size)
  # evaluate d1 in d2 frame
  d1 = _from_to(d1, info2.pos, info2.mat, info1.pos, info1.mat)
  d2 = functools.partial(d2, size=info2.size)
  x0 = info2.mat.T @ (x0 - info2.pos)
  fn = _clearance(d1, d2)
  _, pos = _gradient_descent(fn, x0, 10)
  dist = d1(pos) + d2(pos)
  n = jax.grad(d1)(pos) - jax.grad(d2)(pos)
  pos = info2.mat @ pos + info2.pos  # d2 to global frame
  n = info2.mat @ n
  return dist, pos, math.make_frame(n)

@collider(ncon=1)
def sphere_ellipsoid(s: GeomInfo, e: GeomInfo) -> Collision:
  """Calculates contact between a sphere and an ellipsoid."""
  x0 = 0.5 * (s.pos + e.pos)
  return _optim(_sphere, _ellipsoid, s, e, x0)

@collider(ncon=1)
def sphere_cylinder(s: GeomInfo, c: GeomInfo) -> Collision:
  """Calculates contact between a sphere and a cylinder."""
  # TODO: implement analytical version.
  x0 = 0.5 * (s.pos + c.pos)
  return _optim(_sphere, _cylinder, s, c, x0)

@collider(ncon=1)
def capsule_ellipsoid(c: GeomInfo, e: GeomInfo) -> Collision:
  """ "Calculates contact between a capsule and an ellipsoid."""
  x0 = 0.5 * (c.pos + e.pos)
  return _optim(_capsule, _ellipsoid, c, e, x0)

@collider(ncon=2)
def capsule_cylinder(ca: GeomInfo, cy: GeomInfo) -> Collision:
  """Calculates contact between a capsule and a cylinder."""
  # TODO: improve robustness
  # Near sharp corners, the SDF might give the penetration depth with respect
  # to a surface that is not in collision. Possible solutions is to find the
  # contact points analytically or to change the SDF depending on the relative
  # pose of the bodies.
  mid = 0.5 * (ca.pos + cy.pos)
  vec = ca.mat[:, 2] * ca.size[1]
  x0 = mx.array([mid - vec, mid + vec])
  optim_ = functools.partial(_optim, _capsule, _cylinder, ca, cy)
  return _vmap(optim_)(x0)

@collider(ncon=1)
def ellipsoid_ellipsoid(e1: GeomInfo, e2: GeomInfo) -> Collision:
  """Calculates contact between two ellipsoids."""
  x0 = 0.5 * (e1.pos + e2.pos)
  return _optim(_ellipsoid, _ellipsoid, e1, e2, x0)

@collider(ncon=1)
def ellipsoid_cylinder(e: GeomInfo, c: GeomInfo) -> Collision:
  """Calculates contact between and ellipsoid and a cylinder."""
  x0 = 0.5 * (e.pos + c.pos)
  return _optim(_ellipsoid, _cylinder, e, c, x0)

@collider(ncon=4)
def cylinder_cylinder(c1: GeomInfo, c2: GeomInfo) -> Collision:
  """Calculates contact between a cylinder and a cylinder."""
  # TODO: improve robustness
  # Near sharp corners, the SDF might give the penetration depth with respect
  # to a surface that is not in collision. Possible solutions is to find the
  # contact points analytically or to change the SDF depending on the relative
  # pose of the bodies.
  basis = math.make_frame(c2.pos - c1.pos)
  mid = 0.5 * (c1.pos + c2.pos)
  r = mx.maximum(c1.size[0], c2.size[0])
  x0 = mx.array([
      mid + r * basis[1],
      mid + r * basis[2],
      mid - r * basis[1],
      mid - r * basis[2],
  ])
  optim_ = functools.partial(_optim, _cylinder, _cylinder, c1, c2)
  return _vmap(optim_)(x0)
