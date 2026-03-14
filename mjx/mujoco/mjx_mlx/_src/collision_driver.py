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
"""Runs collision checking for all geoms in a Model – MLX port.

To do this, collision_driver builds a collision function table, and then runs
the collision functions serially on the parameters in the table.

For example, if a Model has three geoms:

geom   |   type
---------------
1      | sphere
2      | capsule
3      | sphere

collision_driver organizes it into these functions and runs them:

function       | geom pair
--------------------------
sphere_sphere  | (1, 3)
sphere_capsule | (1, 2), (2, 3)


Besides collision function, function tables are keyed on mesh id and condim,
in order to guarantee static shapes for contacts and jacobians.
"""

import dataclasses as _dc
import itertools
from typing import Dict, Iterator, List, Tuple, Union

import mlx.core as mx
import mujoco
from mujoco.mjx_mlx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.collision_convex import box_box
from mujoco.mjx_mlx._src.collision_convex import capsule_convex
from mujoco.mjx_mlx._src.collision_convex import convex_convex
from mujoco.mjx_mlx._src.collision_convex import hfield_capsule
from mujoco.mjx_mlx._src.collision_convex import hfield_convex
from mujoco.mjx_mlx._src.collision_convex import hfield_sphere
from mujoco.mjx_mlx._src.collision_convex import plane_convex
from mujoco.mjx_mlx._src.collision_convex import sphere_convex
from mujoco.mjx_mlx._src.collision_primitive import capsule_capsule
from mujoco.mjx_mlx._src.collision_primitive import plane_capsule
from mujoco.mjx_mlx._src.collision_primitive import plane_cylinder
from mujoco.mjx_mlx._src.collision_primitive import plane_ellipsoid
from mujoco.mjx_mlx._src.collision_primitive import plane_sphere
from mujoco.mjx_mlx._src.collision_primitive import sphere_capsule
from mujoco.mjx_mlx._src.collision_primitive import sphere_sphere
# NOTE: SDF collisions not yet ported to MLX; will raise if encountered.
# from mujoco.mjx_mlx._src.collision_sdf import ...
from mujoco.mjx_mlx._src.collision_types import FunctionKey
from mujoco.mjx_mlx._src.types import Contact
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DataMLX
from mujoco.mjx_mlx._src.types import DisableBit
from mujoco.mjx_mlx._src.types import GeomType
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import ModelMLX
from mujoco.mjx_mlx._src.types import OptionMLX
# pylint: enable=g-importing-member
from mujoco.mjx_mlx._src.dataclasses import tree_map as _tree_map
from mujoco.mjx_mlx._src import math as _math
import numpy as np


# pair-wise collision functions
_COLLISION_FUNC = {
    (GeomType.PLANE, GeomType.SPHERE): plane_sphere,
    (GeomType.PLANE, GeomType.CAPSULE): plane_capsule,
    (GeomType.PLANE, GeomType.BOX): plane_convex,
    (GeomType.PLANE, GeomType.ELLIPSOID): plane_ellipsoid,
    (GeomType.PLANE, GeomType.CYLINDER): plane_cylinder,
    (GeomType.PLANE, GeomType.MESH): plane_convex,
    (GeomType.HFIELD, GeomType.SPHERE): hfield_sphere,
    (GeomType.HFIELD, GeomType.CAPSULE): hfield_capsule,
    (GeomType.HFIELD, GeomType.BOX): hfield_convex,
    (GeomType.HFIELD, GeomType.MESH): hfield_convex,
    (GeomType.SPHERE, GeomType.SPHERE): sphere_sphere,
    (GeomType.SPHERE, GeomType.CAPSULE): sphere_capsule,
    # SDF-based collisions not yet ported:
    # (GeomType.SPHERE, GeomType.CYLINDER): sphere_cylinder,
    # (GeomType.SPHERE, GeomType.ELLIPSOID): sphere_ellipsoid,
    (GeomType.SPHERE, GeomType.BOX): sphere_convex,
    (GeomType.SPHERE, GeomType.MESH): sphere_convex,
    (GeomType.CAPSULE, GeomType.CAPSULE): capsule_capsule,
    (GeomType.CAPSULE, GeomType.BOX): capsule_convex,
    # (GeomType.CAPSULE, GeomType.ELLIPSOID): capsule_ellipsoid,
    # (GeomType.CAPSULE, GeomType.CYLINDER): capsule_cylinder,
    (GeomType.CAPSULE, GeomType.MESH): capsule_convex,
    # (GeomType.ELLIPSOID, GeomType.ELLIPSOID): ellipsoid_ellipsoid,
    # (GeomType.ELLIPSOID, GeomType.CYLINDER): ellipsoid_cylinder,
    # (GeomType.CYLINDER, GeomType.CYLINDER): cylinder_cylinder,
    (GeomType.BOX, GeomType.BOX): box_box,
    (GeomType.BOX, GeomType.MESH): convex_convex,
    (GeomType.MESH, GeomType.MESH): convex_convex,
}

# Maximum constraint dimension for collision functions.
_MAX_NCON = 8

# geoms for which we ignore broadphase
_GEOM_NO_BROADPHASE = {GeomType.HFIELD, GeomType.PLANE}


def has_collision_fn(t1: GeomType, t2: GeomType) -> bool:
  """Returns True if a collision function exists for a pair of geom types."""
  return (t1, t2) in _COLLISION_FUNC


def geom_pairs(
    m: Union[Model, mujoco.MjModel],
) -> Iterator[Tuple[int, int, int]]:
  """Yields geom pairs to check for collisions.

  Args:
    m: a MuJoCo or MJX model

  Yields:
    geom1, geom2, and pair index if defined in <pair> (else -1)
  """
  pairs = set()

  for i in range(m.npair):
    g1, g2 = m.pair_geom1[i], m.pair_geom2[i]
    # order pairs by geom_type for correct function mapping
    if m.geom_type[g1] > m.geom_type[g2]:
      g1, g2 = g2, g1
    pairs.add((g1, g2))
    yield g1, g2, i

  exclude_signature = set(m.exclude_signature)
  geom_con = m.geom_contype | m.geom_conaffinity
  filterparent = not (m.opt.disableflags & DisableBit.FILTERPARENT)
  b_start = m.body_geomadr
  b_end = b_start + m.body_geomnum

  for b1 in range(m.nbody):
    if not geom_con[b_start[b1] : b_end[b1]].any():
      continue
    w1 = m.body_weldid[b1]
    w1_p = m.body_weldid[m.body_parentid[w1]]

    for b2 in range(b1, m.nbody):
      if not geom_con[b_start[b2] : b_end[b2]].any():
        continue
      signature = (b1 << 16) + (b2)
      if signature in exclude_signature:
        continue
      w2 = m.body_weldid[b2]
      # ignore self-collisions
      if w1 == w2:
        continue
      w2_p = m.body_weldid[m.body_parentid[w2]]
      # ignore parent-child collisions
      if filterparent and w1 != 0 and w2 != 0 and (w1 == w2_p or w2 == w1_p):
        continue
      g1_range = [g for g in range(b_start[b1], b_end[b1]) if geom_con[g]]
      g2_range = [g for g in range(b_start[b2], b_end[b2]) if geom_con[g]]

      for g1, g2 in itertools.product(g1_range, g2_range):
        t1, t2 = m.geom_type[g1], m.geom_type[g2]
        # order pairs by geom_type for correct function mapping
        if t1 > t2:
          g1, g2, t1, t2 = g2, g1, t2, t1
        # ignore plane<>plane and plane<>hfield
        if (t1, t2) == (GeomType.PLANE, GeomType.PLANE):
          continue
        if (t1, t2) == (GeomType.PLANE, GeomType.HFIELD):
          continue
        # geoms must match contype and conaffinity on some bit
        mask = m.geom_contype[g1] & m.geom_conaffinity[g2]
        mask |= m.geom_contype[g2] & m.geom_conaffinity[g1]
        if not mask:
          continue

        if (g1, g2) not in pairs:
          pairs.add((g1, g2))
          yield g1, g2, -1


def _geom_groups(
    m: Union[Model, mujoco.MjModel],
) -> Dict[FunctionKey, List[Tuple[int, int, int]]]:
  """Returns geom pairs grouped by collision function."""
  groups = {}

  for g1, g2, ip in geom_pairs(m):
    types = m.geom_type[g1], m.geom_type[g2]
    data_ids = m.geom_dataid[g1], m.geom_dataid[g2]
    if ip > -1:
      condim = m.pair_dim[ip]
    elif m.geom_priority[g1] > m.geom_priority[g2]:
      condim = m.geom_condim[g1]
    elif m.geom_priority[g1] < m.geom_priority[g2]:
      condim = m.geom_condim[g2]
    else:
      condim = max(m.geom_condim[g1], m.geom_condim[g2])

    key = FunctionKey(types, data_ids, condim)

    if types[0] == mujoco.mjtGeom.mjGEOM_HFIELD:
      geom_rbound_hfield = (
          m._impl.geom_rbound_hfield if isinstance(m, Model) else m.geom_rbound
      )
      nrow, ncol = m.hfield_nrow[data_ids[0]], m.hfield_ncol[data_ids[0]]
      xsize, ysize = m.hfield_size[data_ids[0]][:2]
      xtick, ytick = (2 * xsize) / (ncol - 1), (2 * ysize) / (nrow - 1)
      xbound = int(np.ceil(2 * geom_rbound_hfield[g2] / xtick)) + 1
      xbound = min(xbound, ncol)
      ybound = int(np.ceil(2 * geom_rbound_hfield[g2] / ytick)) + 1
      ybound = min(ybound, nrow)
      key = FunctionKey(types, data_ids, condim, (xbound, ybound))

    groups.setdefault(key, []).append((g1, g2, ip))

  return groups


def _contact_groups(m: Model, d: Data) -> Dict[FunctionKey, Contact]:
  """Returns contact groups to check for collisions."""
  groups = {}
  eps = mujoco.mjMINVAL

  for key, geom_ids in _geom_groups(m).items():
    geom = np.array(geom_ids)
    geom1, geom2, ip = geom.T
    geom1_nopair = geom1[ip == -1]
    geom2_nopair = geom2[ip == -1]
    ip_pair = ip[ip != -1]
    params = []

    if ip_pair.size > 0:
      params.append((
          mx.array(m.pair_margin[ip_pair] - m.pair_gap[ip_pair]),
          mx.clip(mx.array(m.pair_friction[ip_pair]), a_min=eps),
          mx.array(m.pair_solref[ip_pair]),
          mx.array(m.pair_solreffriction[ip_pair]),
          mx.array(m.pair_solimp[ip_pair]),
      ))
    if geom1_nopair.size > 0 and geom2_nopair.size > 0:
      margin = mx.array(m.geom_margin[geom1_nopair] + m.geom_margin[geom2_nopair])
      gap = mx.array(m.geom_gap[geom1_nopair] + m.geom_gap[geom2_nopair])
      solmix1 = mx.array(m.geom_solmix[geom1_nopair])
      solmix2 = mx.array(m.geom_solmix[geom2_nopair])
      mix = solmix1 / (solmix1 + solmix2)
      mix = mx.where((solmix1 < eps) & (solmix2 < eps), mx.array(0.5), mix)
      mix = mx.where((solmix1 < eps) & (solmix2 >= eps), mx.array(0.0), mix)
      mix = mx.where((solmix1 >= eps) & (solmix2 < eps), mx.array(1.0), mix)
      mix = mx.expand_dims(mix, axis=-1)
      # friction: max
      friction = mx.maximum(
          mx.array(m.geom_friction[geom1_nopair]),
          mx.array(m.geom_friction[geom2_nopair]),
      )
      solref1 = mx.array(m.geom_solref[geom1_nopair])
      solref2 = mx.array(m.geom_solref[geom2_nopair])
      # reference standard: mix
      solref_standard = mix * solref1 + (1 - mix) * solref2
      # reference direct: min
      solref_direct = mx.minimum(solref1, solref2)
      is_standard = (solref1[:, [0, 0]] > 0) & (solref2[:, [0, 0]] > 0)
      solref = mx.where(is_standard, solref_standard, solref_direct)
      solreffriction = mx.zeros(geom1_nopair.shape + (mujoco.mjNREF,))
      # impedance: mix
      solimp = (
          mix * mx.array(m.geom_solimp[geom1_nopair])
          + (1 - mix) * mx.array(m.geom_solimp[geom2_nopair])
      )

      pri = m.geom_priority[geom1_nopair] != m.geom_priority[geom2_nopair]
      if pri.any():
        gp1 = m.geom_priority[geom1_nopair]
        gp2 = m.geom_priority[geom2_nopair]
        gp = np.where(gp1 > gp2, geom1_nopair, geom2_nopair)[pri]
        friction_np = np.array(friction)
        friction_np[pri] = np.array(m.geom_friction[gp])
        friction = mx.array(friction_np)
        solref_np = np.array(solref)
        solref_np[pri] = np.array(m.geom_solref[gp])
        solref = mx.array(solref_np)
        solimp_np = np.array(solimp)
        solimp_np[pri] = np.array(m.geom_solimp[gp])
        solimp = mx.array(solimp_np)

      # unpack 5d friction:
      friction = friction[:, [0, 0, 1, 2, 2]]
      params.append((margin - gap, friction, solref, solreffriction, solimp))

    params_concat = [mx.concatenate(p) for p in zip(*params)]
    includemargin, friction, solref, solreffriction, solimp = params_concat

    groups[key] = Contact(
        dist=None,
        pos=None,
        frame=None,
        includemargin=includemargin,
        friction=friction,
        solref=solref,
        solreffriction=solreffriction,
        solimp=solimp,
        dim=d._impl.contact.dim,
        geom1=mx.array(geom[:, 0]),
        geom2=mx.array(geom[:, 1]),
        geom=mx.array(geom[:, :2]),
        efc_address=d._impl.contact.efc_address,
    )

  return groups


def _numeric(m: Union[Model, mujoco.MjModel], name: str) -> int:
  id_ = support.name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
  return int(m.numeric_data[id_]) if id_ >= 0 else -1


def make_condim(
    m: Union[Model, mujoco.MjModel],
) -> np.ndarray:
  """Returns the dims of the contacts for a Model."""
  if isinstance(m, mujoco.MjModel):
    sdf_initpoints = m.opt.sdf_initpoints
  elif isinstance(m.opt._impl, OptionMLX):
    sdf_initpoints = m.opt._impl.sdf_initpoints
  else:
    raise ValueError(
        'make_condim requires mujoco.MjModel or mjx_mlx.Model with MLX'
        ' backend implementation.'
    )

  if m.opt.disableflags & DisableBit.CONTACT:
    return np.empty(0, dtype=int)

  group_counts = {k: len(v) for k, v in _geom_groups(m).items()}

  max_geom_pairs = _numeric(m, 'max_geom_pairs')

  if max_geom_pairs > -1:
    for k in group_counts:
      if set(k.types) & _GEOM_NO_BROADPHASE:
        continue
      group_counts[k] = min(group_counts[k], max_geom_pairs)

  max_contact_points = _numeric(m, 'max_contact_points')

  condim_counts = {}
  for k, v in group_counts.items():
    func = _COLLISION_FUNC.get(k.types, None)
    if func is not None:
      ncon = func.ncon
    else:
      raise ValueError(
          f'Collision function not found for geom types {k.types[0]},'
          f' {k.types[1]}'
      )
    num_contacts = condim_counts.get(k.condim, 0) + ncon * v
    if max_contact_points > -1:
      num_contacts = min(max_contact_points, num_contacts)
    condim_counts[k.condim] = num_contacts

  dims = sum(([c] * condim_counts[c] for c in sorted(condim_counts)), [])

  return np.array(dims)


def collision(m: Model, d: Data) -> Data:
  """Collides geometries."""
  if not isinstance(m._impl, ModelMLX) or not isinstance(d._impl, DataMLX):
    raise ValueError('collision requires MLX backend implementation.')

  if d._impl.ncon == 0:
    return d

  max_geom_pairs = _numeric(m, 'max_geom_pairs')
  max_contact_points = _numeric(m, 'max_contact_points')

  # run collision functions on groups
  groups = _contact_groups(m, d)
  for key, contact in groups.items():
    # broad phase cull if requested
    if (
        max_geom_pairs > -1
        and contact.geom.shape[0] > max_geom_pairs
        and not set(key.types) & _GEOM_NO_BROADPHASE
    ):
      # compute distances between geom bounding spheres
      dists = []
      for i in range(contact.geom.shape[0]):
        gi1, gi2 = int(contact.geom[i, 0]), int(contact.geom[i, 1])
        p1, p2 = d.geom_xpos[gi1], d.geom_xpos[gi2]
        s1 = float(m.geom_rbound[gi1])
        s2 = float(m.geom_rbound[gi2])
        dists.append(float(_math.norm(p2 - p1)) - (s1 + s2))
      dists_arr = mx.array(dists)
      # select top-k closest pairs
      idx = mx.argsort(dists_arr)[:max_geom_pairs]
      contact = _tree_map(lambda x, idx=idx: x[idx], contact)

    # run the collision function specified by the grouping key
    func = _COLLISION_FUNC[key.types]
    ncon = func.ncon

    dist, pos, frame = func(m, d, key, contact.geom)
    if ncon > 1:
      # repeat contacts to match the number of collisions returned
      def _repeat_fn(x, r=ncon):
        pieces = []
        for i in range(x.shape[0]):
          row = mx.expand_dims(x[i], 0)
          pieces.append(mx.broadcast_to(row, (r,) + x.shape[1:]))
        return mx.concatenate(pieces)
      contact = _tree_map(_repeat_fn, contact)
    groups[key] = _dc.replace(contact, dist=dist, pos=pos, frame=frame)

  # collapse contacts together, ensuring they are grouped by condim
  condim_groups = {}
  for key, contact in groups.items():
    condim_groups.setdefault(key.condim, []).append(contact)

  # limit the number of contacts per condim group if requested
  if max_contact_points > -1:
    for key, contacts in condim_groups.items():
      contact = _concat_contacts(contacts)
      if contact.geom.shape[0] > max_contact_points:
        idx = mx.argsort(contact.dist)[:max_contact_points]
        contact = _tree_map(lambda x, idx=idx: x[idx], contact)
      condim_groups[key] = [contact]

  contacts = sum([condim_groups[k] for k in sorted(condim_groups)], [])
  contact = _concat_contacts(contacts)

  return d.tree_replace({'_impl.contact': contact})


def _concat_contacts(contacts):
  """Concatenate a list of Contact dataclasses."""
  if len(contacts) == 1:
    return contacts[0]
  fields = _dc.fields(contacts[0])
  kwargs = {}
  for f in fields:
    vals = [getattr(c, f.name) for c in contacts]
    if vals[0] is None:
      kwargs[f.name] = None
    elif isinstance(vals[0], mx.array):
      kwargs[f.name] = mx.concatenate(vals)
    elif isinstance(vals[0], np.ndarray):
      kwargs[f.name] = np.concatenate(vals)
    else:
      kwargs[f.name] = vals[0]  # scalar / non-array
  return type(contacts[0])(**kwargs)
