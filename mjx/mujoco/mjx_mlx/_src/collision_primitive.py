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
"""Collision primitives – MLX port."""

from typing import Tuple

import mlx.core as mx
from mujoco.mjx_mlx._src import math
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.collision_types import Collision
from mujoco.mjx_mlx._src.collision_types import GeomInfo
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import Model
# pylint: enable=g-importing-member


def _concat_tree(*collisions):
  """Concatenate collision tuples (dist, pos, frame) along axis 0."""
  dists = mx.concatenate([c[0] for c in collisions])
  poss = mx.concatenate([c[1] for c in collisions])
  frames = mx.concatenate([c[2] for c in collisions])
  return dists, poss, frames


def collider(ncon: int):
  """Wraps collision functions for use by collision_driver."""

  def wrapper(func):
    def collide(m: Model, d: Data, _, geom: mx.array) -> Collision:
      g1, g2 = geom.T[0], geom.T[1]
      # Build per-pair info and call func in a loop (replaces jax.vmap)
      n_pairs = g1.shape[0] if hasattr(g1, 'shape') and len(g1.shape) > 0 else 1
      results = []
      for i in range(n_pairs):
        gi1 = int(g1[i]) if n_pairs > 1 else int(g1)
        gi2 = int(g2[i]) if n_pairs > 1 else int(g2)
        info1 = GeomInfo(d.geom_xpos[gi1], d.geom_xmat[gi1], m.geom_size[gi1])
        info2 = GeomInfo(d.geom_xpos[gi2], d.geom_xmat[gi2], m.geom_size[gi2])
        dist_i, pos_i, frame_i = func(info1, info2)
        # ensure batch dims
        if len(dist_i.shape) == 0:
          dist_i = mx.expand_dims(dist_i, axis=0)
        if len(pos_i.shape) == 1:
          pos_i = mx.expand_dims(pos_i, axis=0)
        if len(frame_i.shape) == 2:
          frame_i = mx.expand_dims(frame_i, axis=0)
        results.append((dist_i, pos_i, frame_i))
      if ncon > 1:
        return _concat_tree(*results)
      # ncon == 1: stack across pairs
      dist = mx.concatenate([r[0] for r in results])
      pos = mx.concatenate([r[1] for r in results])
      frame = mx.concatenate([r[2] for r in results])
      return dist, pos, frame

    collide.ncon = ncon
    return collide

  return wrapper


def _plane_sphere(
    plane_normal: mx.array,
    plane_pos: mx.array,
    sphere_pos: mx.array,
    sphere_radius: mx.array,
) -> Tuple[mx.array, mx.array]:
  """Returns the distance and contact point between a plane and sphere."""
  dist = mx.sum((sphere_pos - plane_pos) * plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@collider(ncon=1)
def plane_sphere(plane: GeomInfo, sphere: GeomInfo) -> Collision:
  """Calculates contact between a plane and a sphere."""
  n = plane.mat[:, 2]
  dist, pos = _plane_sphere(n, plane.pos, sphere.pos, sphere.size[0])
  return dist, pos, math.make_frame(n)


@collider(ncon=2)
def plane_capsule(plane: GeomInfo, cap: GeomInfo) -> Collision:
  """Calculates two contacts between a capsule and a plane."""
  n, axis = plane.mat[:, 2], cap.mat[:, 2]
  # align contact frames with capsule axis
  b, b_norm = math.normalize_with_norm(axis - n * mx.sum(n * axis))
  y, z = mx.array([0.0, 1.0, 0.0]), mx.array([0.0, 0.0, 1.0])
  b = mx.where(b_norm < 0.5, mx.where((-0.5 < n[1]) & (n[1] < 0.5), y, z), b)
  frame = mx.array([mx.stack([n, b, math._cross(n, b)])])
  segment = axis * cap.size[1]
  collisions = []
  for offset in [segment, -segment]:
    dist, pos = _plane_sphere(n, plane.pos, cap.pos + offset, cap.size[0])
    dist = mx.expand_dims(dist, axis=0)
    pos = mx.expand_dims(pos, axis=0)
    collisions.append((dist, pos, frame))
  return _concat_tree(*collisions)


@collider(ncon=1)
def plane_ellipsoid(plane: GeomInfo, ellipsoid: GeomInfo) -> Collision:
  """Calculates one contact between an ellipsoid and a plane."""
  n = plane.mat[:, 2]
  size = ellipsoid.size
  sphere_support = -math.normalize((ellipsoid.mat.T @ n) * size)
  pos = ellipsoid.pos + ellipsoid.mat @ (sphere_support * size)
  dist = mx.sum(n * (pos - plane.pos))
  pos = pos - n * dist * 0.5
  return dist, pos, math.make_frame(n)


@collider(ncon=3)
def plane_cylinder(plane: GeomInfo, cylinder: GeomInfo) -> Collision:
  """Calculates three contacts between a cylinder and a plane."""
  n = plane.mat[:, 2]
  axis = cylinder.mat[:, 2]

  # make sure axis points towards plane
  prjaxis = mx.sum(n * axis)
  sign = -math.sign(prjaxis)
  axis, prjaxis = axis * sign, prjaxis * sign

  # compute normal distance to cylinder center
  dist0 = mx.sum((cylinder.pos - plane.pos) * n)

  # remove component of -normal along axis, compute length
  vec = axis * prjaxis - n
  len_ = math.norm(vec)

  vec = mx.where(
      len_ < 1e-12,
      # disk parallel to plane: pick x-axis of cylinder, scale by radius
      cylinder.mat[:, 0] * cylinder.size[0],
      # general configuration: normalize vector, scale by radius
      math.safe_div(vec, len_) * cylinder.size[0],
  )

  # project vector on normal
  prjvec = mx.sum(vec * n)

  # scale axis by half-length
  axis = axis * cylinder.size[1]
  prjaxis = prjaxis * cylinder.size[1]

  # compute sideways vector: vec1
  prjvec1 = -prjvec * 0.5
  vec1 = math.normalize(math._cross(vec, axis)) * cylinder.size[0]
  vec1 = vec1 * mx.sqrt(mx.array(3.0)) * 0.5

  # disk parallel to plane
  d1 = dist0 + prjaxis + prjvec
  d2 = dist0 + prjaxis + prjvec1
  dist = mx.array([d1, d2, d2])
  pos = (
      cylinder.pos
      + axis
      + mx.stack([
          vec - n * d1 * 0.5,
          vec1 + vec * -0.5 - n * d2 * 0.5,
          -vec1 + vec * -0.5 - n * d2 * 0.5,
      ])
  )

  # cylinder parallel to plane
  cond = mx.abs(prjaxis) < 1e-3
  d3 = dist0 - prjaxis + prjvec
  # dist[1] = d3 if cond
  dist_list = [dist[0], mx.where(cond, d3, dist[1]), dist[2]]
  dist = mx.stack(dist_list)
  # pos[1] if cond
  new_pos1 = cylinder.pos + vec - axis - n * d3 * 0.5
  pos_rows = [pos[0], mx.where(cond, new_pos1, pos[1]), pos[2]]
  pos = mx.stack(pos_rows)

  frame = mx.stack([math.make_frame(n)] * 3, axis=0)
  return dist, pos, frame


def _sphere_sphere(
    pos1: mx.array, radius1: mx.array, pos2: mx.array, radius2: mx.array
) -> Tuple[mx.array, mx.array, mx.array]:
  """Returns the penetration, contact point, and normal between two spheres."""
  n, dist = math.normalize_with_norm(pos2 - pos1)
  n = mx.where(dist == 0.0, mx.array([1.0, 0.0, 0.0]), n)
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + dist * 0.5)
  return dist, pos, n


@collider(ncon=1)
def sphere_sphere(s1: GeomInfo, s2: GeomInfo) -> Collision:
  """Calculates contact between two spheres."""
  dist, pos, n = _sphere_sphere(s1.pos, s1.size[0], s2.pos, s2.size[0])
  return dist, pos, math.make_frame(n)


@collider(ncon=1)
def sphere_capsule(sphere: GeomInfo, cap: GeomInfo) -> Collision:
  """Calculates one contact between a sphere and a capsule."""
  axis, length = cap.mat[:, 2], cap.size[1]
  segment = axis * length
  pt = math.closest_segment_point(
      cap.pos - segment, cap.pos + segment, sphere.pos
  )
  dist, pos, n = _sphere_sphere(sphere.pos, sphere.size[0], pt, cap.size[0])
  return dist, pos, math.make_frame(n)


@collider(ncon=1)
def capsule_capsule(cap1: GeomInfo, cap2: GeomInfo) -> Collision:
  """Calculates one contact between two capsules."""
  axis1, length1 = cap1.mat[:, 2], cap1.size[1]
  axis2, length2 = cap2.mat[:, 2], cap2.size[1]
  seg1, seg2 = axis1 * length1, axis2 * length2
  pt1, pt2 = math.closest_segment_to_segment_points(
      cap1.pos - seg1,
      cap1.pos + seg1,
      cap2.pos - seg2,
      cap2.pos + seg2,
  )
  radius1, radius2 = cap1.size[0], cap2.size[0]
  dist, pos, n = _sphere_sphere(pt1, radius1, pt2, radius2)
  return dist, pos, math.make_frame(n)
