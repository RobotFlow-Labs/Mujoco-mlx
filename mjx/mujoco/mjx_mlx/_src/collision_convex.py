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
"""Convex collisions – MLX port."""

import functools
from typing import Callable, Tuple, Union

import mlx.core as mx
from mujoco.mjx_mlx._src import math
from mujoco.mjx_mlx._src import mesh
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.collision_types import Collision
from mujoco.mjx_mlx._src.collision_types import ConvexInfo
from mujoco.mjx_mlx._src.collision_types import FunctionKey
from mujoco.mjx_mlx._src.collision_types import GeomInfo
from mujoco.mjx_mlx._src.collision_types import HFieldInfo
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DataMLX
from mujoco.mjx_mlx._src.types import GeomType
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import ModelMLX
# pylint: enable=g-importing-member
from mujoco.mjx_mlx._src.dataclasses import tree_map as _tree_map

_GeomInfo = Union[GeomInfo, ConvexInfo]

# ---------------------------------------------------------------------------
# Helpers: vmap replacements for MLX (explicit loops + stack)
# ---------------------------------------------------------------------------


def _vmap_cross_with_fixed(vecs: mx.array, fixed: mx.array) -> mx.array:
  """vmap(cross, in_axes=[0, None])(vecs, fixed) replacement."""
  return mx.stack([math._cross(vecs[i], fixed) for i in range(vecs.shape[0])])


def _vmap_dot(a: mx.array, b: mx.array) -> mx.array:
  """vmap(dot) replacement: row-wise dot products."""
  return mx.sum(a * b, axis=-1)


# ---------------------------------------------------------------------------
# collider wrapper (replaces jax.vmap based wrapper)
# ---------------------------------------------------------------------------


def collider(ncon: int):
  """Wraps collision functions for use by collision_driver."""

  def wrapper(collision_fn):
    def collide(
        m: Model, d: Data, key: FunctionKey, geom: mx.array
    ) -> Collision:
      if not isinstance(m._impl, ModelMLX) or not isinstance(d._impl, DataMLX):
        raise ValueError('collider requires MLX backend implementation.')

      g1, g2 = geom.T[0], geom.T[1]
      n_pairs = g1.shape[0] if hasattr(g1, 'shape') and len(g1.shape) > 0 else 1

      fn = collision_fn

      # Build per-pair infos and call fn in a loop (replaces jax.vmap)
      results = []
      for idx in range(n_pairs):
        gi1 = int(g1[idx]) if n_pairs > 1 else int(g1)
        gi2 = int(g2[idx]) if n_pairs > 1 else int(g2)

        infos = [
            GeomInfo(d.geom_xpos[gi1], d.geom_xmat[gi1], m.geom_size[gi1]),
            GeomInfo(d.geom_xpos[gi2], d.geom_xmat[gi2], m.geom_size[gi2]),
        ]

        cur_fn = fn
        for side in [0, 1]:
          if key.types[side] == GeomType.BOX:
            infos[side] = mesh.box(infos[side])
          elif key.types[side] == GeomType.MESH:
            c = infos[side]
            cm = m._impl.mesh_convex[key.data_ids[side]]
            infos[side] = ConvexInfo(**{
                'pos': c.pos, 'mat': c.mat, 'size': c.size,
                'vert': cm.vert, 'face': cm.face,
                'face_normal': cm.face_normal,
                'edge': cm.edge, 'edge_face_normal': cm.edge_face_normal,
            })
          elif key.types[side] == GeomType.HFIELD:
            hfield_info = mesh.hfield(m, key.data_ids[side])
            infos[side] = hfield_info.replace(
                pos=infos[side].pos, mat=infos[side].mat
            )
            cur_fn = functools.partial(cur_fn, subgrid_size=key.subgrid_size)

        dist_i, pos_i, frame_i = cur_fn(*infos)

        # ensure batch dims
        if len(dist_i.shape) == 0:
          dist_i = mx.expand_dims(dist_i, axis=0)
        if len(pos_i.shape) == 1:
          pos_i = mx.expand_dims(pos_i, axis=0)
        if len(frame_i.shape) == 2:
          frame_i = mx.expand_dims(frame_i, axis=0)
        results.append((dist_i, pos_i, frame_i))

      if ncon > 1:
        dist = mx.concatenate([r[0] for r in results])
        pos = mx.concatenate([r[1] for r in results])
        frame = mx.concatenate([r[2] for r in results])
        return dist, pos, frame

      dist = mx.concatenate([r[0] for r in results])
      pos = mx.concatenate([r[1] for r in results])
      frame = mx.concatenate([r[2] for r in results])
      return dist, pos, frame

    collide.ncon = ncon
    return collide

  return wrapper


# ---------------------------------------------------------------------------
# Utility geometry functions
# ---------------------------------------------------------------------------


def _closest_segment_point_plane(
    a: mx.array, b: mx.array, p0: mx.array, plane_normal: mx.array
) -> mx.array:
  """Gets the closest point between a line segment and a plane."""
  n = plane_normal
  d = mx.sum(p0 * n)
  denom = mx.sum(n * (b - a))
  t = (d - mx.sum(n * a)) / (denom + 1e-6 * (denom == 0.0))
  t = mx.clip(t, 0, 1)
  segment_point = a + t * (b - a)
  return segment_point


def _manifold_points(
    poly: mx.array, poly_mask: mx.array, poly_norm: mx.array
) -> mx.array:
  """Chooses four points on the polygon with approximately maximal area."""
  dist_mask = mx.where(poly_mask, mx.zeros(poly_mask.shape), mx.full(poly_mask.shape, -1e6))
  a_idx = mx.argmax(dist_mask)
  a = poly[a_idx]
  # choose point b furthest from a
  b_idx = mx.argmax(mx.sum((a - poly) ** 2, axis=1) + dist_mask)
  b = poly[b_idx]
  # choose point c furthest along the axis orthogonal to (a-b)
  ab = math._cross(poly_norm, a - b)
  ap = a - poly
  c_idx = mx.argmax(mx.abs(ap @ ab) + dist_mask)
  c = poly[c_idx]
  # choose point d furthest from the other two triangle edges
  ac = math._cross(poly_norm, a - c)
  bc = math._cross(poly_norm, b - c)
  bp = b - poly
  dist_bp = mx.abs(bp @ bc) + dist_mask
  dist_ap = mx.abs(ap @ ac) + dist_mask
  d_idx = mx.argmax(dist_bp + dist_ap) % poly.shape[0]
  return mx.array([a_idx, b_idx, c_idx, d_idx])


# ---------------------------------------------------------------------------
# plane_convex
# ---------------------------------------------------------------------------


@collider(ncon=4)
def plane_convex(plane: GeomInfo, convex: ConvexInfo) -> Collision:
  """Calculates contacts between a plane and a convex object."""
  vert = convex.vert

  # get points in the convex frame
  plane_pos = convex.mat.T @ (plane.pos - convex.pos)
  n = convex.mat.T @ plane.mat[:, 2]
  support = (plane_pos - vert) @ n
  # search for manifold points within a 1mm skin depth
  idx = _manifold_points(vert, support > mx.maximum(mx.array(0.0), mx.max(support) - 1e-3), n)
  pos = vert[idx]

  # convert to world frame
  pos = convex.pos + pos @ convex.mat.T
  n = plane.mat[:, 2]

  frame = mx.stack([math.make_frame(n)] * 4, axis=0)
  unique = mx.sum(mx.tril(mx.array([[int(idx[i] == idx[j]) for j in range(4)] for i in range(4)])), axis=1) == 1
  dist = mx.where(unique, -support[idx], mx.array(1.0))
  pos = pos - 0.5 * dist[:, None] * n
  return dist, pos, frame


# ---------------------------------------------------------------------------
# sphere_convex
# ---------------------------------------------------------------------------


def _sphere_convex(sphere: GeomInfo, convex: ConvexInfo) -> Collision:
  """Calculates contact between a sphere and a convex mesh."""
  faces = convex.face
  normals = convex.face_normal

  # Put sphere in convex frame.
  sphere_pos = convex.mat.T @ (sphere.pos - convex.pos)

  # Get support from face normals.
  def _get_support_single(face, normal):
    pos = sphere_pos - normal * sphere.size[0]
    return mx.sum((pos - face[0]) * normal)

  support = mx.stack([_get_support_single(faces[i], normals[i]) for i in range(faces.shape[0])])
  has_separating_axis = mx.any(support >= 0)

  # Pick the face with the best separating axis.
  best_idx = int(mx.argmax(support))
  face = faces[best_idx]
  face_normal = normals[best_idx]

  # Get closest point between the polygon face and the sphere center point.
  pt = _project_pt_onto_plane(sphere_pos, face[0], face_normal)
  edge_p0 = mx.roll(face, 1, axis=0)
  edge_p1 = face
  side_normals = _vmap_cross_with_fixed(edge_p1 - edge_p0, face_normal)

  # edge_dist for each edge
  edge_dist = mx.stack([mx.sum((pt - edge_p0[i]) * side_normals[i]) for i in range(edge_p0.shape[0])])
  pt_on_face = mx.all(edge_dist <= 0)

  # If the point is outside side planes, project onto the closest side plane
  degenerate_edge = mx.all(side_normals == 0, axis=1)
  behind = edge_dist < 0.0
  edge_dist_mod = mx.where(degenerate_edge | behind, mx.array(1e12), edge_dist)
  idx = int(mx.argmin(edge_dist_mod))
  edge_pt = math.closest_segment_point(edge_p0[idx], edge_p1[idx], pt)
  pt = mx.where(pt_on_face, pt, edge_pt)

  # Get the normal, dist, and contact position.
  pt_normal, d = math.normalize_with_norm(pt - sphere_pos)
  inside = mx.sum(pt * pt_normal) > 0
  sign = mx.where(inside, mx.array(-1.0), mx.array(1.0))
  n = mx.where(pt_on_face | (d < 1e-6), -face_normal, sign * pt_normal)
  d = d * sign

  spt = sphere_pos + n * sphere.size[0]
  dist = mx.where(has_separating_axis, mx.array(1.0), d - sphere.size[0])
  pos = (pt + spt) * 0.5

  # Go back to world frame.
  n = convex.mat @ n
  pos = convex.mat @ pos + convex.pos

  return dist, pos, n


@collider(ncon=1)
def sphere_convex(sphere: GeomInfo, convex: ConvexInfo) -> Collision:
  """Calculates contact between a sphere and a convex mesh."""
  dist, pos, n = _sphere_convex(sphere, convex)
  return dist, pos, math.make_frame(n)


# ---------------------------------------------------------------------------
# capsule_convex
# ---------------------------------------------------------------------------


def _capsule_convex(cap: GeomInfo, convex: ConvexInfo) -> Collision:
  """Calculates contacts between a capsule and a convex object."""
  faces = convex.face
  normals = convex.face_normal

  # Put capsule in convex frame.
  cap_pos = convex.mat.T @ (cap.pos - convex.pos)
  axis, length = cap.mat[:, 2], cap.size[1]
  axis = convex.mat.T @ axis
  seg = axis * length
  cap_pts = mx.stack([cap_pos - seg, cap_pos + seg])

  # Get support from face normals.
  def _get_support_cap(face, normal):
    pts = cap_pts - normal * cap.size[0]
    sup = mx.stack([mx.sum((pts[j] - face[0]) * normal) for j in range(pts.shape[0])])
    return mx.min(sup)

  support = mx.stack([_get_support_cap(faces[i], normals[i]) for i in range(faces.shape[0])])
  has_support = mx.all(support < 0)

  # Pick the face with minimal penetration.
  best_idx = int(mx.argmax(support))
  face = faces[best_idx]
  normal = normals[best_idx]

  # Clip the segment against side planes.
  edge_p0 = mx.roll(face, 1, axis=0)
  edge_p1 = face
  side_planes = _vmap_cross_with_fixed(edge_p1 - edge_p0, normal)

  cap_pts_clipped, mask = _clip_edge_to_planes(
      cap_pts[0], cap_pts[1], edge_p0, side_planes
  )
  cap_pts_clipped = cap_pts_clipped - normal * cap.size[0]
  face_pts = mx.stack([
      _project_pt_onto_plane(cap_pts_clipped[i], face[0], normal)
      for i in range(cap_pts_clipped.shape[0])
  ])

  pos = (cap_pts_clipped + face_pts) * 0.5
  contact_normal = -mx.stack([normal] * 2, axis=0)
  face_penetration = mx.where(
      mask & has_support,
      mx.sum((face_pts - cap_pts_clipped) * normal, axis=-1),
      mx.array(-1.0),
  )

  # Pick a potential shallow edge contact.
  def get_edge_axis(edge_pair):
    edge_closest_pt, cap_closest_pt = math.closest_segment_to_segment_points(
        edge_pair[0], edge_pair[1], cap_pts[0], cap_pts[1]
    )
    edge_dir = edge_closest_pt - cap_closest_pt
    degenerate_edge_dir = mx.sum(mx.square(edge_dir)) < 1e-6
    edge_axis, edge_dist = math.normalize_with_norm(edge_dir)
    return edge_dist, edge_axis, degenerate_edge_dir, edge_closest_pt, cap_closest_pt

  edge_verts = convex.vert[convex.edge]  # (n_edges, 2, 3)
  edge_face_normal = convex.edge_face_normal

  # Loop over edges
  edge_results = [get_edge_axis(edge_verts[i]) for i in range(edge_verts.shape[0])]
  e_dists = mx.stack([r[0] for r in edge_results])
  e_axes = mx.stack([r[1] for r in edge_results])
  e_degens = mx.stack([r[2] for r in edge_results])
  e_edge_pts = mx.stack([r[3] for r in edge_results])
  e_cap_pts = mx.stack([r[4] for r in edge_results])

  e_idx = int(mx.argmin(mx.abs(e_dists)))
  edge_dist = e_dists[e_idx]
  edge_axis = e_axes[e_idx]
  degenerate_edge_dir = e_degens[e_idx]
  edge_closest_pt = e_edge_pts[e_idx]
  cap_closest_pt = e_cap_pts[e_idx]

  edge_face_normals = edge_face_normal[e_idx]
  edge_voronoi_front = mx.all((edge_face_normals @ edge_axis) < 0)
  shallow = ~degenerate_edge_dir & edge_voronoi_front
  edge_penetration = mx.where(shallow, cap.size[0] - edge_dist, mx.array(-1.0))

  # Determine edge contact position.
  edge_pos = (
      edge_closest_pt + (cap_closest_pt + edge_axis * cap.size[0])
  ) * 0.5
  edge_dir_parallel_to_face = (
      mx.abs(mx.sum(edge_axis * normal)) > 0.99
  ) & ~degenerate_edge_dir
  min_face_penetration = mx.min(face_penetration)
  has_edge_contact = (
      (edge_penetration > 0)
      & mx.where(
          min_face_penetration > 0,
          edge_penetration < min_face_penetration,
          mx.array(True),
      )
      & ~edge_dir_parallel_to_face
      & edge_voronoi_front
  )

  # Get the contact info.
  pos = mx.where(has_edge_contact, mx.stack([edge_pos, pos[1]]), pos)
  n = mx.where(has_edge_contact, mx.stack([edge_axis, contact_normal[1]]), contact_normal)

  # Go back to world frame.
  pos = convex.pos + pos @ convex.mat.T
  n = n @ convex.mat.T

  dist = -mx.where(
      has_edge_contact, mx.array([edge_penetration, -1.0]), face_penetration
  )
  return dist, pos, n


@collider(ncon=2)
def capsule_convex(cap: GeomInfo, convex: ConvexInfo) -> Collision:
  """Calculates contacts between a capsule and a convex object."""
  dist, pos, n = _capsule_convex(cap, convex)
  frame = mx.stack([math.make_frame(n[i]) for i in range(n.shape[0])])
  return dist, pos, frame


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def _project_pt_onto_plane(
    pt: mx.array, plane_pt: mx.array, plane_normal: mx.array
) -> mx.array:
  """Projects a point onto a plane along the plane normal."""
  dist = mx.sum((pt - plane_pt) * plane_normal)
  return pt - dist * plane_normal


def _project_poly_onto_plane(
    poly: mx.array, plane_pt: mx.array, plane_normal: mx.array
) -> mx.array:
  """Projects a polygon onto a plane using the plane normal."""
  nrm = math.normalize(plane_normal)
  return mx.stack([_project_pt_onto_plane(poly[i], plane_pt, nrm) for i in range(poly.shape[0])])


def _project_poly_onto_poly_plane(
    poly1: mx.array, norm1: mx.array, poly2: mx.array, norm2: mx.array
) -> mx.array:
  """Projects poly1 onto the poly2 plane along poly1's normal."""
  d = mx.sum(poly2[0] * norm2)
  denom = mx.sum(norm1 * norm2)
  t = (d - poly1 @ norm2) / (denom + 1e-6 * (denom == 0.0))
  new_poly = poly1 + t.reshape(-1, 1) * norm1
  return new_poly


def _point_in_front_of_plane(
    plane_pt: mx.array, plane_normal: mx.array, pt: mx.array
) -> mx.array:
  """Checks if a point is strictly in front of a plane."""
  return mx.sum((pt - plane_pt) * plane_normal) > 1e-6


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


def _clip_edge_to_planes(
    edge_p0: mx.array,
    edge_p1: mx.array,
    plane_pts: mx.array,
    plane_normals: mx.array,
) -> Tuple[mx.array, mx.array]:
  """Clips an edge against side planes."""
  p0, p1 = edge_p0, edge_p1
  p0_in_front = _vmap_dot(p0 - plane_pts, plane_normals) > 1e-6
  p1_in_front = _vmap_dot(p1 - plane_pts, plane_normals) > 1e-6

  # Get candidate clipped points
  candidate_clipped_ps = mx.stack([
      _closest_segment_point_plane(p0, p1, plane_pts[i], plane_normals[i])
      for i in range(plane_pts.shape[0])
  ])

  def clip_edge_point(p0_, p1_, p0_in_front_, clipped_ps):
    new_edge_ps = mx.stack([
        mx.where(p0_in_front_[i], clipped_ps[i], p0_)
        for i in range(p0_in_front_.shape[0])
    ])
    dists = new_edge_ps @ (p1_ - p0_)
    new_edge_p = new_edge_ps[mx.argmax(dists)]
    return new_edge_p

  new_p0 = clip_edge_point(p0, p1, p0_in_front, candidate_clipped_ps)
  new_p1 = clip_edge_point(p1, p0, p1_in_front, candidate_clipped_ps)
  clipped_pts = mx.stack([new_p0, new_p1])

  both_in_front = p0_in_front & p1_in_front
  mask = ~mx.any(both_in_front)
  new_ps = mx.where(mask, clipped_pts, mx.stack([p0, p1]))
  mask = mx.where(mx.sum((p0 - p1) * (new_ps[0] - new_ps[1])) < 0, mx.array(False), mask)
  return new_ps, mx.array([mask, mask])


def _clip(
    clipping_poly: mx.array,
    subject_poly: mx.array,
    clipping_normal: mx.array,
    subject_normal: mx.array,
) -> Tuple[mx.array, mx.array]:
  """Clips a subject polygon against a clipping polygon."""
  # Get clipping edge points, edge planes, and edge normals.
  clipping_p0 = mx.roll(clipping_poly, 1, axis=0)
  clipping_plane_pts = clipping_p0
  clipping_p1 = clipping_poly
  clipping_plane_normals = _vmap_cross_with_fixed(
      clipping_p1 - clipping_p0, clipping_normal
  )

  # Get subject edge points, edge planes, and edge normals.
  subject_edge_p0 = mx.roll(subject_poly, 1, axis=0)
  subject_plane_pts = subject_edge_p0
  subject_edge_p1 = subject_poly
  subject_plane_normals = _vmap_cross_with_fixed(
      subject_edge_p1 - subject_edge_p0, subject_normal
  )

  # Clip all edges of the subject poly against clipping side planes.
  clipped_edges0_list = []
  masks0_list = []
  for i in range(subject_edge_p0.shape[0]):
    ce, cm = _clip_edge_to_planes(
        subject_edge_p0[i], subject_edge_p1[i],
        clipping_plane_pts, clipping_plane_normals,
    )
    clipped_edges0_list.append(ce)
    masks0_list.append(cm)
  clipped_edges0 = mx.stack(clipped_edges0_list)
  masks0 = mx.stack(masks0_list)

  # Project the clipping poly onto the subject plane.
  clipping_p0_s = _project_poly_onto_poly_plane(
      clipping_p0, clipping_normal, subject_poly, subject_normal
  )
  clipping_p1_s = _project_poly_onto_poly_plane(
      clipping_p1, clipping_normal, subject_poly, subject_normal
  )

  # Clip all edges of the clipping poly against subject planes.
  clipped_edges1_list = []
  masks1_list = []
  for i in range(clipping_p0_s.shape[0]):
    ce, cm = _clip_edge_to_planes(
        clipping_p0_s[i], clipping_p1_s[i],
        subject_plane_pts, subject_plane_normals,
    )
    clipped_edges1_list.append(ce)
    masks1_list.append(cm)
  clipped_edges1 = mx.stack(clipped_edges1_list)
  masks1 = mx.stack(masks1_list)

  # Merge the points and reshape.
  clipped_edges = mx.concatenate([clipped_edges0, clipped_edges1])
  masks = mx.concatenate([masks0, masks1])
  clipped_points = clipped_edges.reshape((-1, 3))
  mask = masks.reshape(-1)

  return clipped_points, mask


def _create_contact_manifold(
    clipping_poly: mx.array,
    subject_poly: mx.array,
    clipping_norm: mx.array,
    subject_norm: mx.array,
    sep_axis: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
  """Creates a contact manifold between two convex polygons."""
  poly_incident, mask = _clip(
      clipping_poly, subject_poly, clipping_norm, subject_norm
  )
  poly_ref = _project_poly_onto_plane(
      poly_incident, clipping_poly[0], clipping_norm
  )
  behind_clipping_plane = mx.stack([
      _point_in_front_of_plane(clipping_poly[0], -clipping_norm, poly_incident[i])
      for i in range(poly_incident.shape[0])
  ])
  mask = mask & behind_clipping_plane

  # Choose four contact points.
  best = _manifold_points(poly_ref, mask, clipping_norm)
  contact_pts = poly_ref[best]
  mask_pts = mask[best]
  penetration_dir = poly_incident[best] - contact_pts
  penetration = penetration_dir @ (-clipping_norm)

  dist = mx.where(mask_pts, -penetration, mx.ones_like(penetration))
  pos = contact_pts
  normal = -mx.stack([sep_axis] * 4, axis=0)
  return dist, pos, normal


# ---------------------------------------------------------------------------
# box_box
# ---------------------------------------------------------------------------


def _box_box_impl(
    faces_a: mx.array,
    faces_b: mx.array,
    vertices_a: mx.array,
    vertices_b: mx.array,
    normals_a: mx.array,
    normals_b: mx.array,
    unique_edges_a: mx.array,
    unique_edges_b: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
  """Runs the Separating Axis Test for two boxes."""
  edge_dir_a, edge_dir_b = unique_edges_a, unique_edges_b
  # tile / repeat
  n_a, n_b = edge_dir_a.shape[0], edge_dir_b.shape[0]
  edge_dir_a_r = mx.concatenate([edge_dir_a] * n_b, axis=0)
  edge_dir_b_r = mx.stack([edge_dir_b[i] for i in range(n_b) for _ in range(n_a)])
  edge_axes = mx.stack([math._cross(edge_dir_a_r[i], edge_dir_b_r[i]) for i in range(edge_dir_a_r.shape[0])])
  degenerate_edge_axes = mx.sum(edge_axes ** 2, axis=1) < 1e-6
  edge_axes = mx.stack([math.normalize(edge_axes[i], axis=0) for i in range(edge_axes.shape[0])])
  n_face_axes = normals_a.shape[0] + normals_b.shape[0]
  degenerate_axes = mx.concatenate(
      [mx.array([False] * n_face_axes), degenerate_edge_axes]
  )

  axes = mx.concatenate([normals_a, normals_b, edge_axes])

  # for each separating axis, get the support
  def _get_support_single(axis, is_degenerate):
    support_a = vertices_a @ axis
    support_b = vertices_b @ axis
    dist1 = mx.max(support_a) - mx.min(support_b)
    dist2 = mx.max(support_b) - mx.min(support_a)
    sign = mx.where(dist1 > dist2, mx.array(-1.0), mx.array(1.0))
    dist = mx.minimum(dist1, dist2)
    dist = mx.where(~is_degenerate, dist, mx.array(1e6))
    return dist, sign

  supports = [_get_support_single(axes[i], degenerate_axes[i]) for i in range(axes.shape[0])]
  support = mx.stack([s[0] for s in supports])
  sign_arr = mx.stack([s[1] for s in supports])

  # get the best face axis
  best_face_idx = int(mx.argmin(support[:n_face_axes]))
  best_face_axis = axes[best_face_idx]

  # choose the best separating axis
  best_idx = int(mx.argmin(support))
  best_sign = sign_arr[best_idx]
  best_axis = axes[best_idx]
  is_edge_contact = best_idx >= n_face_axes
  is_edge_contact = is_edge_contact & (mx.abs(mx.sum(best_face_axis * best_axis)) < 0.99)

  # get the (reference) face most aligned with the separating axis
  dist_a = normals_a @ best_axis
  dist_b = normals_b @ best_axis
  a_max = int(mx.argmax(dist_a))
  b_max = int(mx.argmax(dist_b))
  a_min = int(mx.argmin(dist_a))
  b_min = int(mx.argmin(dist_b))

  ref_face = mx.where(best_sign > 0, faces_a[a_max], faces_b[b_max])
  ref_face_norm = mx.where(best_sign > 0, normals_a[a_max], normals_b[b_max])
  incident_face = mx.where(best_sign > 0, faces_b[b_min], faces_a[a_min])
  incident_face_norm = mx.where(
      best_sign > 0, normals_b[b_min], normals_a[a_min]
  )

  dist, pos, normal = _create_contact_manifold(
      ref_face,
      incident_face,
      ref_face_norm,
      incident_face_norm,
      -best_sign * best_axis,
  )

  # For edge contacts, use the clipped face point
  idx = int(mx.argmin(dist))
  dist = mx.where(
      is_edge_contact,
      mx.array([dist[idx], 1.0, 1.0, 1.0]),
      dist,
  )
  pos = mx.where(is_edge_contact, mx.stack([pos[idx]] * 4), pos)

  return dist, pos, normal


def _box_box(b1: ConvexInfo, b2: ConvexInfo) -> Collision:
  """Calculates contacts between two boxes."""
  faces1 = b1.face
  faces2 = b2.face

  to_local_pos = b2.mat.T @ (b1.pos - b2.pos)
  to_local_mat = b2.mat.T @ b1.mat

  faces1 = to_local_pos + faces1 @ to_local_mat.T
  normals1 = b1.face_normal @ to_local_mat.T
  normals2 = b2.face_normal

  vertices1 = to_local_pos + b1.vert @ to_local_mat.T
  vertices2 = b2.vert

  dist, pos, normal = _box_box_impl(
      faces1,
      faces2,
      vertices1,
      vertices2,
      normals1,
      normals2,
      to_local_mat.T,
      mx.eye(3),
  )

  # Go back to world frame.
  pos = b2.pos + pos @ b2.mat.T
  n = normal @ b2.mat.T
  dist = mx.where(mx.isinf(dist), mx.array(float('inf')), dist)

  return dist, pos, n


# ---------------------------------------------------------------------------
# Gauss-map SAT for general convex-convex
# ---------------------------------------------------------------------------


def _arcs_intersect(
    a: mx.array, b: mx.array, c: mx.array, d: mx.array
) -> mx.array:
  """Tests if arcs AB and CD on the unit sphere intersect."""
  ba, dc = math._cross(b, a), math._cross(d, c)
  cba, dba = mx.sum(c * ba), mx.sum(d * ba)
  adc, bdc = mx.sum(a * dc), mx.sum(b * dc)
  return (cba * dba < 0) & (adc * bdc < 0) & (cba * bdc > 0)


def _sat_gaussmap(
    centroid_a: mx.array,
    faces_a: mx.array,
    faces_b: mx.array,
    vertices_a: mx.array,
    vertices_b: mx.array,
    normals_a: mx.array,
    normals_b: mx.array,
    edges_a: mx.array,
    edges_b: mx.array,
    edge_face_normals_a: mx.array,
    edge_face_normals_b: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
  """Runs the Separating Axis Test for a pair of hulls via gauss maps."""
  # Handle face separating axes.
  axes = mx.concatenate([normals_a, -normals_b])

  def _get_support_face(axis):
    support_a = vertices_a @ axis
    support_b = vertices_b @ axis
    dist = mx.max(support_a) - mx.min(support_b)
    separating = dist < 0
    dist = mx.where(dist < 0, mx.array(1e6), dist)
    return dist, separating

  face_results = [_get_support_face(axes[i]) for i in range(axes.shape[0])]
  support = mx.stack([r[0] for r in face_results])
  separating = mx.stack([r[1] for r in face_results])
  is_face_separating = mx.any(separating)

  # choose the best separating axis
  best_idx = int(mx.argmin(support))
  best_axis = axes[best_idx]

  # get the (reference) face most aligned with the separating axis
  dist_a = normals_a @ best_axis
  dist_b = normals_b @ (-best_axis)
  face_a_idx = int(mx.argmax(dist_a))
  face_b_idx = int(mx.argmax(dist_b))

  cond = best_idx < normals_a.shape[0]
  ref_face = mx.where(cond, faces_a[face_a_idx], faces_b[face_b_idx])
  incident_face = mx.where(cond, faces_b[face_b_idx], faces_a[face_a_idx])
  ref_face_norm = mx.where(cond, normals_a[face_a_idx], normals_b[face_b_idx])
  incident_face_norm = mx.where(
      cond, normals_b[face_b_idx], normals_a[face_a_idx]
  )

  dist, pos, normal = _create_contact_manifold(
      ref_face,
      incident_face,
      ref_face_norm,
      incident_face_norm,
      -best_axis,
  )
  dist = mx.where(is_face_separating, mx.array(1.0), dist)

  # Handle edge separating axes by checking all edge pairs.
  n_ea, n_eb = edges_a.shape[0], edges_b.shape[0]
  # Build index arrays
  a_idx_list = list(range(n_ea)) * n_eb
  b_idx_list = [i for i in range(n_eb) for _ in range(n_ea)]

  normal_a_1 = mx.stack([edge_face_normals_a[a_idx_list[i], 0] for i in range(len(a_idx_list))])
  normal_a_2 = mx.stack([edge_face_normals_a[a_idx_list[i], 1] for i in range(len(a_idx_list))])
  normal_b_1 = mx.stack([edge_face_normals_b[b_idx_list[i], 0] for i in range(len(b_idx_list))])
  normal_b_2 = mx.stack([edge_face_normals_b[b_idx_list[i], 1] for i in range(len(b_idx_list))])
  is_minkowski_face = mx.stack([
      _arcs_intersect(normal_a_1[i], normal_a_2[i], -normal_b_1[i], -normal_b_2[i])
      for i in range(len(a_idx_list))
  ])

  # get distances
  edge_a_dirs = mx.stack([math.normalize(edges_a[i, 0] - edges_a[i, 1]) for i in range(n_ea)])
  edge_b_dirs = mx.stack([math.normalize(edges_b[i, 0] - edges_b[i, 1]) for i in range(n_eb)])
  edge_a_dir = edge_a_dirs[mx.array(a_idx_list)]
  edge_b_dir = edge_b_dirs[mx.array(b_idx_list)]
  edges_a_sel = edges_a[mx.array(a_idx_list)]
  edges_b_sel = edges_b[mx.array(b_idx_list)]
  edge_a_pt = edges_a_sel[:, 0]
  edge_a_pt_2 = edges_a_sel[:, 1]
  edge_b_pt = edges_b_sel[:, 0]
  edge_b_pt_2 = edges_b_sel[:, 1]

  def get_normals(a_dir, a_pt, b_dir):
    edge_axis = math._cross(a_dir, b_dir)
    degenerate_edge_axis = mx.sum(edge_axis ** 2) < 1e-6
    edge_axis = math.normalize(edge_axis)
    sign = mx.where(mx.sum(edge_axis * (a_pt - centroid_a)) > 0.0, mx.array(1.0), mx.array(-1.0))
    return edge_axis * sign, degenerate_edge_axis

  edge_normals_res = [get_normals(edge_a_dir[i], edge_a_pt[i], edge_b_dir[i]) for i in range(len(a_idx_list))]
  edge_axes = mx.stack([r[0] for r in edge_normals_res])
  degenerate_edge_axes = mx.stack([r[1] for r in edge_normals_res])
  edge_dist = _vmap_dot(edge_axes, edge_b_pt - edge_a_pt)
  # handle degenerate axis
  edge_dist = mx.where(degenerate_edge_axes, mx.array(-float('inf')), edge_dist)
  # ensure edges create minkowski face
  edge_dist = mx.where(is_minkowski_face, edge_dist, mx.array(-float('inf')))

  best_edge_idx = int(mx.argmax(edge_dist))
  best_edge_dist = edge_dist[best_edge_idx]
  is_edge_contact = mx.where(
      mx.max(dist) < 0.0,
      best_edge_dist > mx.max(dist) - 1e-6,
      (best_edge_dist < 0) & ~mx.isinf(best_edge_dist),
  )
  is_edge_contact = is_edge_contact & ~is_face_separating
  normal = mx.where(is_edge_contact, edge_axes[best_edge_idx], normal)
  dist = mx.where(
      is_edge_contact,
      mx.array([best_edge_dist, 1.0, 1.0, 1.0]),
      dist,
  )
  a_closest, b_closest = math.closest_segment_to_segment_points(
      edge_a_pt[best_edge_idx],
      edge_a_pt_2[best_edge_idx],
      edge_b_pt[best_edge_idx],
      edge_b_pt_2[best_edge_idx],
  )
  pos = mx.where(
      is_edge_contact, mx.stack([0.5 * (a_closest + b_closest)] * 4), pos
  )

  return dist, pos, normal


def _convex_convex(c1: ConvexInfo, c2: ConvexInfo) -> Collision:
  """Calculates contacts between two convex meshes."""
  # pad face vertices so that we can broadcast between geom1 and geom2
  nvert1, nvert2 = c1.face.shape[1], c2.face.shape[1]
  if nvert1 < nvert2:
    pad_width = nvert2 - nvert1
    # edge-pad: repeat last vertex
    last = c1.face[:, -1:, :]
    face = mx.concatenate([c1.face] + [last] * pad_width, axis=1)
    c1 = c1.replace(face=face)
  elif nvert2 < nvert1:
    pad_width = nvert1 - nvert2
    last = c2.face[:, -1:, :]
    face = mx.concatenate([c2.face] + [last] * pad_width, axis=1)
    c2 = c2.replace(face=face)

  # ensure that the first object has fewer verts
  swapped = c1.vert.shape[0] > c2.vert.shape[0]
  if swapped:
    c1, c2 = c2, c1

  faces1 = c1.face
  faces2 = c2.face

  # convert to c2 frame
  to_local_pos = c2.mat.T @ (c1.pos - c2.pos)
  to_local_mat = c2.mat.T @ c1.mat

  faces1 = to_local_pos + faces1 @ to_local_mat.T
  normals1 = c1.face_normal @ to_local_mat.T
  normals2 = c2.face_normal

  vertices1 = to_local_pos + c1.vert @ to_local_mat.T
  vertices2 = c2.vert

  edges1 = c1.vert[c1.edge]  # use original vert indices
  edges1 = to_local_pos + edges1.reshape(-1, 3) @ to_local_mat.T
  edges1 = edges1.reshape(-1, 2, 3)
  edges2 = c2.vert[c2.edge]
  edges2 = edges2.reshape(-1, 2, 3)

  edge_face_normals1 = c1.edge_face_normal @ to_local_mat.T
  edge_face_normals2 = c2.edge_face_normal

  dist, pos, normal = _sat_gaussmap(
      to_local_pos,
      faces1,
      faces2,
      vertices1,
      vertices2,
      normals1,
      normals2,
      edges1,
      edges2,
      edge_face_normals1,
      edge_face_normals2,
  )

  # Go back to world frame.
  pos = c2.pos + pos @ c2.mat.T
  n = normal @ c2.mat.T
  n = -n if swapped else n
  dist = mx.where(mx.isinf(dist), mx.array(float('inf')), dist)

  return dist, pos, n


@collider(ncon=4)
def box_box(b1: ConvexInfo, b2: ConvexInfo) -> Collision:
  """Calculates contacts between two boxes."""
  dist, pos, n = _box_box(b1, b2)
  frame = mx.stack([math.make_frame(n[i]) for i in range(n.shape[0])])
  return dist, pos, frame


@collider(ncon=4)
def convex_convex(c1: ConvexInfo, c2: ConvexInfo) -> Collision:
  """Calculates contacts between two convex objects."""
  dist, pos, n = _convex_convex(c1, c2)
  frame = mx.stack([math.make_frame(n[i]) for i in range(n.shape[0])])
  return dist, pos, frame


# ---------------------------------------------------------------------------
# hfield collisions
# ---------------------------------------------------------------------------


def _hfield_collision(
    collider_fn: Callable[[_GeomInfo, _GeomInfo], Collision],
    h: HFieldInfo,
    obj: _GeomInfo,
    obj_rbound: mx.array,
    subgrid_size: Tuple[int, int],
) -> Collision:
  """Collides an object with prisms in a height field."""
  # put obj in hfield frame
  obj_pos = h.mat.T @ (obj.pos - h.pos)
  obj_mat = h.mat.T @ obj.mat

  xmin = obj_pos[0] - obj_rbound
  ymin = obj_pos[1] - obj_rbound
  cmin = mx.floor((xmin + h.size[0]) / (2 * h.size[0]) * (h.ncol - 1))
  cmin = cmin.astype(mx.int32)
  rmin = mx.floor((ymin + h.size[1]) / (2 * h.size[1]) * (h.nrow - 1))
  rmin = rmin.astype(mx.int32)

  dx = 2.0 * h.size[0] / (h.ncol - 1)
  dy = 2.0 * h.size[1] / (h.nrow - 1)

  bvert = mx.array([0.0, 0.0, -h.size[3]])
  bmask = mx.array([True, True, False])

  prism1_list = []
  prism2_list = []
  for r_off in range(subgrid_size[1]):
    for c_off in range(subgrid_size[0]):
      ri = mx.clip(rmin + r_off, 0, h.nrow - 2)
      ci = mx.clip(cmin + c_off, 0, h.ncol - 2)
      ri_int, ci_int = int(ri), int(ci)

      p1 = mx.array([
          dx * ci - h.size[0],
          dy * ri - h.size[1],
          h.data[ci_int, ri_int] * h.size[2],
      ])
      p2 = mx.array([
          dx * (ci + 1) - h.size[0],
          dy * (ri + 1) - h.size[1],
          h.data[ci_int + 1, ri_int + 1] * h.size[2],
      ])
      p3 = mx.array([
          dx * ci - h.size[0],
          dy * (ri + 1) - h.size[1],
          h.data[ci_int, ri_int + 1] * h.size[2],
      ])
      top = mx.stack([p1, p2, p3])
      bottom = mx.stack([p1, p3, p2]) * bmask + bvert
      vert = mx.concatenate([bottom, top])
      prism1_list.append(mesh.hfield_prism(vert))

      p3_new = p2
      p2_new = mx.array([
          dx * (ci + 1) - h.size[0],
          dy * ri - h.size[1],
          h.data[ci_int + 1, ri_int] * h.size[2],
      ])
      top = mx.stack([p1, p2_new, p3_new])
      bottom = mx.stack([p1, p3_new, p2_new]) * bmask + bvert
      vert = mx.concatenate([bottom, top])
      prism2_list.append(mesh.hfield_prism(vert))

  n_prisms = 2 * len(prism1_list)
  # Collide obj against all prisms
  obj_local = obj.replace(pos=obj_pos, mat=obj_mat)
  all_dists = []
  all_pos = []
  all_n = []
  for prism in prism1_list + prism2_list:
    d_i, p_i, n_i = collider_fn(obj_local, prism)
    if len(d_i.shape) == 0:
      d_i = mx.expand_dims(d_i, 0)
    if len(p_i.shape) == 1:
      p_i = mx.expand_dims(p_i, 0)
    if len(n_i.shape) == 1:
      n_i = mx.expand_dims(n_i, 0)
    all_dists.append(d_i)
    all_pos.append(p_i)
    all_n.append(n_i)

  dist = mx.concatenate(all_dists).flatten()
  pos = mx.concatenate(all_pos).reshape((-1, 3))
  n = mx.concatenate(all_n).reshape((-1, 3))
  n = n * -1  # flip the normal since we flipped args in the call to collider_fn

  # Check that we're in the half-space of the hfield norm.
  # Gather top face normals from all prisms
  all_prisms = prism1_list + prism2_list
  n_repeats = dist.shape[0] // n_prisms
  top_norms = []
  for prism in all_prisms:
    for _ in range(n_repeats):
      top_norms.append(prism.face_normal[1])
  top_norm = mx.stack(top_norms)
  cond = mx.stack([mx.sum(n[i] * h.mat[2]) for i in range(n.shape[0])]) < 1e-6
  n = mx.where(cond[:, None], top_norm, n)

  return dist, pos, n


def _hfield_post_process(h, dist, pos, n):
  """Common post-processing for hfield collisions."""
  n_mean = mx.mean(n, axis=0)
  mask = dist < mx.minimum(mx.array(0.0), mx.min(dist) + 1e-3)
  idx = _manifold_points(pos, mask, n_mean)
  dist, pos, n = dist[idx], pos[idx], n[idx]

  # zero out non-unique contacts
  unique = mx.sum(mx.tril(mx.array(
      [[int(idx[i] == idx[j]) for j in range(4)] for i in range(4)]
  )), axis=1) == 1
  dist = mx.where(unique, dist, mx.array(1.0))

  # back to world frame
  pos = mx.stack([h.mat @ pos[i] + h.pos for i in range(pos.shape[0])])
  n = mx.stack([h.mat @ n[i] for i in range(n.shape[0])])

  return dist, pos, n


@collider(ncon=4)
def hfield_sphere(
    h: HFieldInfo, s: GeomInfo, subgrid_size: Tuple[int, int]
) -> Collision:
  """Calculates contacts between a hfield and a sphere."""
  rbound = mx.max(s.size)
  dist, pos, n = _hfield_collision(_sphere_convex, h, s, rbound, subgrid_size)
  dist, pos, n = _hfield_post_process(h, dist, pos, n)
  return dist, pos, mx.stack([math.make_frame(n[i]) for i in range(n.shape[0])])


@collider(ncon=4)
def hfield_capsule(
    h: HFieldInfo, c: GeomInfo, subgrid_size: Tuple[int, int]
) -> Collision:
  """Calculates contacts between a hfield and a capsule."""
  rbound = c.size[0] + c.size[1]
  dist, pos, n = _hfield_collision(_capsule_convex, h, c, rbound, subgrid_size)
  dist, pos, n = _hfield_post_process(h, dist, pos, n)
  return dist, pos, mx.stack([math.make_frame(n[i]) for i in range(n.shape[0])])


@collider(ncon=4)
def hfield_convex(
    h: HFieldInfo, c: ConvexInfo, subgrid_size: Tuple[int, int]
) -> Collision:
  """Calculates contacts between a hfield and a convex."""
  rbound = mx.max(c.size)
  dist, pos, n = _hfield_collision(_convex_convex, h, c, rbound, subgrid_size)
  dist, pos, n = _hfield_post_process(h, dist, pos, n)
  return dist, pos, mx.stack([math.make_frame(n[i]) for i in range(n.shape[0])])
