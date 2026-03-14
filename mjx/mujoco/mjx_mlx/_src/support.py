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
"""Engine support functions -- MLX port.

Ported from JAX (mujoco.mjx._src.support) to Apple MLX.

Translation notes:
  - jax.Array -> mx.array
  - jnp.XXX -> mx.XXX
  - jax.vmap(fn)(args) -> mx.stack([fn(a) for a in args]) or explicit loops
  - jax.lax.scan -> Python for loop
  - jp.dot(a, b) -> mx.sum(a * b) for 1D, a @ b for matmul
  - jp.cross -> math._cross (manual)
  - scan.body_tree -> _body_tree_scan (inline Python loop)
"""

from collections.abc import Iterable, Sequence
from typing import Optional, Tuple, Union

import mlx.core as mx
import mujoco
from mujoco.mjx_mlx._src import math as mjx_math
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.types import ConeType
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import JacobianType
from mujoco.mjx_mlx._src.types import JointType
from mujoco.mjx_mlx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np


# ---------------------------------------------------------------------------
# Internal: body tree traversal helpers
# ---------------------------------------------------------------------------

def _body_tree_depths(m: Model) -> np.ndarray:
  """Compute depth of each body in the kinematic tree."""
  depths = np.zeros(m.nbody, dtype=np.int32)
  for i in range(1, m.nbody):
    depths[i] = depths[int(m.body_parentid[i])] + 1
  return depths


def _body_tree_order(m: Model, reverse: bool = False):
  """Return body ids grouped by depth, optionally reversed."""
  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0
  levels = []
  for d in range(max_depth + 1):
    ids = np.where(depths == d)[0]
    levels.append(ids)
  if reverse:
    levels = list(reversed(levels))
  return levels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_sparse(m: Union[mujoco.MjModel, Model]) -> bool:
  """Return True if this model should create sparse mass matrices.

  Args:
    m: a MuJoCo or MJX model

  Returns:
    True if provided model should create sparse mass matrices

  For the MLX backend we default to dense (Apple Silicon GPUs are
  typically fast with dense ops).  Sparse is used only when nv >= 60
  or the user explicitly requests it.
  """
  if m.opt.jacobian == JacobianType.AUTO:
    return m.nv >= 60
  return m.opt.jacobian == JacobianType.SPARSE


def make_m(
    m: Model, a: mx.array, b: mx.array, d: Optional[mx.array] = None
) -> mx.array:
  """Computes M = a @ b.T + diag(d)."""

  ij = []
  for i in range(m.nv):
    j = i
    while j > -1:
      ij.append((i, j))
      j = int(m.dof_parentid[j])

  i_idx, j_idx = (mx.array(x) for x in zip(*ij))

  if not is_sparse(m):
    qm = a @ b.T
    if d is not None:
      qm = qm + mx.diag(d)
    mask = mx.zeros((m.nv, m.nv), dtype=mx.bool_)
    # MLX doesn't have .at[].set(), so build mask via scatter
    mask_np = np.zeros((m.nv, m.nv), dtype=bool)
    for ii, jj in ij:
      mask_np[ii, jj] = True
    mask = mx.array(mask_np)
    qm = qm * mask
    qm = qm + mx.tril(qm, -1).T
    return qm

  # Sparse path: compute element-wise dot products
  a_i = a[i_idx]
  b_j = b[j_idx]
  # vmap(dot) -> stack of dot products
  qm = mx.sum(a_i * b_j, axis=-1)

  # add diagonal
  if d is not None:
    qm_np = np.array(qm)
    d_np = np.array(d)
    madr = np.array(m.dof_Madr)
    for k in range(len(madr)):
      qm_np[madr[k]] += d_np[k]
    qm = mx.array(qm_np)

  return qm


def full_m(m: Model, d: Data) -> mx.array:
  """Reconstitute dense mass matrix from qM."""

  if not is_sparse(m):
    return (d._impl or d).qM

  ij = []
  for i in range(m.nv):
    j = i
    while j > -1:
      ij.append((i, j))
      j = int(m.dof_parentid[j])

  i_arr, j_arr = zip(*ij)

  mat_np = np.zeros((m.nv, m.nv))
  qM_np = np.array((d._impl or d).qM)
  for k, (ii, jj) in enumerate(ij):
    mat_np[ii, jj] = qM_np[k]
  mat = mx.array(mat_np)

  # also set upper triangular
  mat = mat + mx.tril(mat, -1).T

  return mat


def mul_m(m: Model, d: Data, vec: mx.array) -> mx.array:
  """Multiply vector by inertia matrix."""

  if not is_sparse(m):
    return (d._impl or d).qM @ vec

  diag_mul = (d._impl or d).qM[mx.array(m.dof_Madr)] * vec

  is_, js, madr_ijs = [], [], []
  for i in range(m.nv):
    madr_ij, j = int(m.dof_Madr[i]), i

    while True:
      madr_ij, j = madr_ij + 1, int(m.dof_parentid[j])
      if j == -1:
        break
      is_, js, madr_ijs = is_ + [i], js + [j], madr_ijs + [madr_ij]

  if not is_:
    return diag_mul

  i_arr = mx.array(is_, dtype=mx.int32)
  j_arr = mx.array(js, dtype=mx.int32)
  madr_arr = mx.array(madr_ijs, dtype=mx.int32)

  # Build output with scatter-add approach via numpy
  out_np = np.array(diag_mul)
  qM_np = np.array((d._impl or d).qM)
  vec_np = np.array(vec)

  for k in range(len(is_)):
    out_np[is_[k]] += qM_np[madr_ijs[k]] * vec_np[js[k]]
    out_np[js[k]] += qM_np[madr_ijs[k]] * vec_np[is_[k]]

  return mx.array(out_np)


def _body_tree_scan_mask(m: Model, body_id) -> mx.array:
  """Compute DOF mask for a given body_id using body tree reverse scan."""
  # Accumulate mask: mark body_id and all its descendants
  # reverse scan up tree: any child that is body_id propagates upward
  mask_np = np.zeros(m.nbody, dtype=np.float32)
  mask_np[int(body_id)] = 1.0
  # Walk upward from leaves: accumulate children to parents
  # Actually we need descendant mask: walk from leaves to root
  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0
  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] == depth and mask_np[b] > 0:
        mask_np[int(m.body_parentid[b])] += mask_np[b]

  # Now mask[b] > 0 for body_id and all its ancestors
  # But we want body_id and all its descendants
  # Re-do: forward pass from root
  desc_mask = np.zeros(m.nbody, dtype=np.float32)
  desc_mask[int(body_id)] = 1.0
  for depth in range(max_depth + 1):
    for b in range(m.nbody):
      if depths[b] == depth and int(m.body_parentid[b]) >= 0:
        if desc_mask[int(m.body_parentid[b])] > 0:
          desc_mask[b] = 1.0

  # Map to dof mask
  dof_mask = np.array([desc_mask[int(m.dof_bodyid[dof])] > 0 for dof in range(m.nv)])
  return mx.array(dof_mask.astype(np.float32))


def jac(
    m: Model, d: Data, point: mx.array, body_id
) -> Tuple[mx.array, mx.array]:
  """Compute pair of (NV, 3) Jacobians of global point attached to body."""
  # Build mask: dofs that affect body_id (body_id and all ancestors)
  # Use reverse tree scan: accumulate from body_id upward
  body_id_int = int(body_id) if not isinstance(body_id, int) else body_id

  # Mark body_id, then propagate to all descendants (reverse scan)
  mask_np = np.zeros(m.nbody, dtype=np.float32)
  mask_np[body_id_int] = 1.0
  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0
  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] == depth:
        mask_np[int(m.body_parentid[b])] += mask_np[b]

  dof_mask = mx.array(
      np.array([mask_np[int(m.dof_bodyid[dof])] > 0 for dof in range(m.nv)],
               dtype=np.float32)
  )

  offset = point - d.subtree_com[mx.array(m.body_rootid)[body_id_int]]

  # jacp[i] = cdof[i, 3:] + cross(cdof[i, :3], offset)
  # jacr[i] = cdof[i, :3]
  jacp_list = []
  jacr_list = []
  for i in range(m.nv):
    cdof_i = d.cdof[i]
    jp_i = cdof_i[3:] + mjx_math._cross(cdof_i[:3], offset)
    jacp_list.append(jp_i * dof_mask[i])
    jacr_list.append(cdof_i[:3] * dof_mask[i])

  jacp = mx.stack(jacp_list) if jacp_list else mx.zeros((0, 3))
  jacr = mx.stack(jacr_list) if jacr_list else mx.zeros((0, 3))

  return jacp, jacr


def jac_dot(
    m: Model, d: Data, point: mx.array, body_id
) -> Tuple[mx.array, mx.array]:
  """Compute pair of (NV, 3) Jacobian time-derivatives of global point."""
  body_id_int = int(body_id) if not isinstance(body_id, int) else body_id

  # Build mask (same as jac)
  mask_np = np.zeros(m.nbody, dtype=np.float32)
  mask_np[body_id_int] = 1.0
  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0
  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] == depth:
        mask_np[int(m.body_parentid[b])] += mask_np[b]

  dof_mask_np = np.array(
      [mask_np[int(m.dof_bodyid[dof])] > 0 for dof in range(m.nv)],
      dtype=np.float32,
  )
  dof_mask = mx.array(dof_mask_np)

  offset = point - d.subtree_com[mx.array(m.body_rootid)[body_id_int]]
  pvel_lin = d.cvel[body_id_int][3:] - mjx_math._cross(
      offset, d.cvel[body_id_int][:3]
  )

  cdof = d.cdof
  cdof_dot = d.cdof_dot

  # check for quaternion dofs
  jnt_type = m.jnt_type[m.dof_jntid]
  dof_adr = m.jnt_dofadr[m.dof_jntid]
  is_quat = (jnt_type == JointType.BALL) | (
      (jnt_type == JointType.FREE) & (np.arange(m.nv) >= dof_adr + 3)
  )

  # compute cdof_dot for quaternion (use current body cvel)
  cdof_dot_list = []
  for i in range(m.nv):
    if is_quat[i]:
      cdof_dot_i = mjx_math.motion_cross(d.cvel[int(m.dof_bodyid[i])], cdof[i])
    else:
      cdof_dot_i = cdof_dot[i]
    cdof_dot_list.append(cdof_dot_i)

  if cdof_dot_list:
    cdof_dot_final = mx.stack(cdof_dot_list)
  else:
    cdof_dot_final = mx.zeros((0, 6))

  jacp_list = []
  jacr_list = []
  for i in range(m.nv):
    a = cdof_dot_final[i]
    b = cdof[i]
    jp_i = a[3:] + mjx_math._cross(a[:3], offset) + mjx_math._cross(
        b[:3], pvel_lin
    )
    jacp_list.append(jp_i * dof_mask[i])
    jacr_list.append(cdof_dot_final[i, :3] * dof_mask[i])

  jacp = mx.stack(jacp_list) if jacp_list else mx.zeros((0, 3))
  jacr = mx.stack(jacr_list) if jacr_list else mx.zeros((0, 3))

  return jacp, jacr


def apply_ft(
    m: Model,
    d: Data,
    force: mx.array,
    torque: mx.array,
    point: mx.array,
    body_id,
) -> mx.array:
  """Apply Cartesian force and torque."""
  jacp, jacr = jac(m, d, point, body_id)
  return jacp @ force + jacr @ torque


def xfrc_accumulate(m: Model, d: Data) -> mx.array:
  """Accumulate xfrc_applied into a qfrc."""
  qfrc_list = []
  for i in range(m.nbody):
    qfrc_i = apply_ft(
        m, d, d.xfrc_applied[i, :3], d.xfrc_applied[i, 3:], d.xipos[i], i
    )
    qfrc_list.append(qfrc_i)
  if qfrc_list:
    return mx.sum(mx.stack(qfrc_list), axis=0)
  return mx.zeros((m.nv,))


def local_to_global(
    world_pos: mx.array,
    world_quat: mx.array,
    local_pos: mx.array,
    local_quat: mx.array,
) -> Tuple[mx.array, mx.array]:
  """Converts local position/orientation to world frame."""
  pos = world_pos + mjx_math.rotate(local_pos, world_quat)
  mat = mjx_math.quat_to_mat(mjx_math.quat_mul(world_quat, local_quat))
  return pos, mat


def _getnum(m: Union[Model, mujoco.MjModel], obj: mujoco._enums.mjtObj) -> int:
  """Gets the number of objects for the given object type."""
  return {
      mujoco.mjtObj.mjOBJ_BODY: m.nbody,
      mujoco.mjtObj.mjOBJ_JOINT: m.njnt,
      mujoco.mjtObj.mjOBJ_GEOM: m.ngeom,
      mujoco.mjtObj.mjOBJ_SITE: m.nsite,
      mujoco.mjtObj.mjOBJ_CAMERA: m.ncam,
      mujoco.mjtObj.mjOBJ_MESH: m.nmesh,
      mujoco.mjtObj.mjOBJ_HFIELD: m.nhfield,
      mujoco.mjtObj.mjOBJ_PAIR: m.npair,
      mujoco.mjtObj.mjOBJ_EQUALITY: m.neq,
      mujoco.mjtObj.mjOBJ_TENDON: m.ntendon,
      mujoco.mjtObj.mjOBJ_ACTUATOR: m.nu,
      mujoco.mjtObj.mjOBJ_SENSOR: m.nsensor,
      mujoco.mjtObj.mjOBJ_NUMERIC: m.nnumeric,
      mujoco.mjtObj.mjOBJ_TUPLE: m.ntuple,
      mujoco.mjtObj.mjOBJ_KEY: m.nkey,
  }.get(obj, 0)


def _getadr(
    m: Union[Model, mujoco.MjModel], obj: mujoco._enums.mjtObj
) -> np.ndarray:
  """Gets the name addresses for the given object type."""
  return {
      mujoco.mjtObj.mjOBJ_BODY: m.name_bodyadr,
      mujoco.mjtObj.mjOBJ_JOINT: m.name_jntadr,
      mujoco.mjtObj.mjOBJ_GEOM: m.name_geomadr,
      mujoco.mjtObj.mjOBJ_SITE: m.name_siteadr,
      mujoco.mjtObj.mjOBJ_CAMERA: m.name_camadr,
      mujoco.mjtObj.mjOBJ_MESH: m.name_meshadr,
      mujoco.mjtObj.mjOBJ_HFIELD: m.name_hfieldadr,
      mujoco.mjtObj.mjOBJ_PAIR: m.name_pairadr,
      mujoco.mjtObj.mjOBJ_EQUALITY: m.name_eqadr,
      mujoco.mjtObj.mjOBJ_TENDON: m.name_tendonadr,
      mujoco.mjtObj.mjOBJ_ACTUATOR: m.name_actuatoradr,
      mujoco.mjtObj.mjOBJ_SENSOR: m.name_sensoradr,
      mujoco.mjtObj.mjOBJ_NUMERIC: m.name_numericadr,
      mujoco.mjtObj.mjOBJ_TUPLE: m.name_tupleadr,
      mujoco.mjtObj.mjOBJ_KEY: m.name_keyadr,
  }[obj]


def id2name(
    m: Union[Model, mujoco.MjModel], typ: mujoco._enums.mjtObj, i: int
) -> Optional[str]:
  """Gets the name of an object with the specified mjtObj type and id."""
  num = _getnum(m, typ)
  if i < 0 or i >= num:
    return None

  adr = _getadr(m, typ)
  name = m.names[adr[i] :].decode('utf-8').split('\x00', 1)[0]
  return name or None


def name2id(
    m: Union[Model, mujoco.MjModel], typ: mujoco._enums.mjtObj, name: str
) -> int:
  """Gets the id of an object with the specified mjtObj type and name."""
  num = _getnum(m, typ)
  adr = _getadr(m, typ)

  names_map = {
      m.names[adr[i] :].decode('utf-8').split('\x00', 1)[0]: i
      for i in range(num)
  }

  return names_map.get(name, -1)


# ---------------------------------------------------------------------------
# Decode pyramid / contact force
# ---------------------------------------------------------------------------


def _decode_pyramid(
    pyramid: mx.array, mu: mx.array, condim: int
) -> mx.array:
  """Converts pyramid representation to contact force."""
  force = mx.zeros((6,))
  if condim == 1:
    force_np = np.zeros(6)
    force_np[0] = float(pyramid[0])
    return mx.array(force_np)

  force_np = np.zeros(6)
  # force_normal = sum(pyramid0_i + pyramid1_i)
  force_np[0] = float(mx.sum(pyramid[0 : 2 * (condim - 1)]))

  # force_tangent_i = (pyramid0_i - pyramid1_i) * mu_i
  for i_dim in range(condim - 1):
    force_np[i_dim + 1] = float(
        (pyramid[2 * i_dim] - pyramid[2 * i_dim + 1]) * mu[i_dim]
    )

  return mx.array(force_np)


def contact_force(
    m: Model, d: Data, contact_id: int, to_world_frame: bool = False
) -> mx.array:
  """Extract 6D force:torque for one contact, in contact frame by default."""
  efc_address = (d._impl or d).contact.efc_address[contact_id]
  condim = (d._impl or d).contact.dim[contact_id]
  if m.opt.cone == ConeType.PYRAMIDAL:
    force = _decode_pyramid(
        (d._impl or d).efc_force[efc_address:],
        (d._impl or d).contact.friction[contact_id],
        condim,
    )
  elif m.opt.cone == ConeType.ELLIPTIC:
    force = (d._impl or d).efc_force[efc_address : efc_address + condim]
    force = mx.concatenate([force, mx.zeros((6 - condim,))])
  else:
    raise ValueError(f'Unknown cone type: {m.opt.cone}')

  if to_world_frame:
    force = (force.reshape((-1, 3)) @ (d._impl or d).contact.frame[contact_id]).reshape(-1)

  return force * (efc_address >= 0)


def contact_force_dim(
    m: Model, d: Data, dim: int
) -> Tuple[mx.array, np.ndarray]:
  """Extract 6D force:torque for contacts with dimension dim."""
  idx_dim = ((d._impl or d).contact.efc_address >= 0) & ((d._impl or d).contact.dim == dim)

  if m.opt.cone == ConeType.PYRAMIDAL:
    efc_width = 1 if dim == 1 else 2 * (dim - 1)
    efc_address = (
        (d._impl or d).contact.efc_address[idx_dim, None]
        + np.arange(efc_width)[None]
    )
    efc_force = (d._impl or d).efc_force[efc_address]
    # vmap _decode_pyramid -> loop
    forces = []
    n = efc_force.shape[0]
    for i in range(n):
      forces.append(
          _decode_pyramid(efc_force[i], (d._impl or d).contact.friction[idx_dim][i], dim)
      )
    force = mx.stack(forces) if forces else mx.zeros((0, 6))
  elif m.opt.cone == ConeType.ELLIPTIC:
    efc_address = (
        (d._impl or d).contact.efc_address[idx_dim, None]
        + np.arange(dim)[None]
    )
    force = (d._impl or d).efc_force[efc_address]
    pad = mx.zeros((force.shape[0], 6 - dim))
    force = mx.concatenate([force, pad], axis=1)
  else:
    raise ValueError(f'Unknown cone type: {m.opt.cone}.')
  return force, np.where(idx_dim)[0]


# ---------------------------------------------------------------------------
# Tendon wrap helpers
# ---------------------------------------------------------------------------


def _length_circle(
    p0: mx.array, p1: mx.array, ind: mx.array, rad: mx.array
) -> mx.array:
  """Compute length of circle."""
  p0n = mjx_math.normalize(p0).reshape(-1)
  p1n = mjx_math.normalize(p1).reshape(-1)

  angle = mx.arccos(mx.clip(mx.sum(p0n * p1n), -1, 1))

  cross_val = p0[1] * p1[0] - p0[0] * p1[1]
  flip = ((cross_val > 0) & (ind != 0)) | ((cross_val < 0) & (ind == 0))
  angle = mx.where(flip, 2 * mx.array(3.141592653589793) - angle, angle)

  return rad * angle


def _is_intersect(
    p1: mx.array, p2: mx.array, p3: mx.array, p4: mx.array
) -> mx.array:
  """Check for intersection between two lines defined by their endpoints."""
  det = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])

  a = mjx_math.safe_div(
      (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0]),
      det,
  )
  b = mjx_math.safe_div(
      (p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0]),
      det,
  )

  return mx.where(
      mx.abs(det) < mujoco.mjMINVAL,
      mx.array(0),
      (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1),
  )


def wrap_circle(
    d: mx.array, sd: mx.array, sidesite: mx.array, rad: mx.array
) -> Tuple[mx.array, mx.array]:
  """Compute circle wrap arc length and end points."""
  sqlen0 = d[0] ** 2 + d[1] ** 2
  sqlen1 = d[2] ** 2 + d[3] ** 2
  sqrad = rad * rad
  dif = mx.array([d[2] - d[0], d[3] - d[1]])
  dd = dif[0] ** 2 + dif[1] ** 2
  a = mx.clip(
      -(dif[0] * d[0] + dif[1] * d[1]) / mx.maximum(mujoco.mjMINVAL, dd),
      0,
      1,
  )
  seg = mx.array([a * dif[0] + d[0], a * dif[1] + d[1]])

  point_inside0 = sqlen0 < sqrad
  point_inside1 = sqlen1 < sqrad
  circle_too_small = rad < mujoco.mjMINVAL
  points_too_close = dd < mujoco.mjMINVAL

  intersect_and_side = (seg[0] ** 2 + seg[1] ** 2 > sqrad) & (
      mx.where(sidesite, mx.array(0), mx.array(1)) | (mx.sum(sd * seg) >= 0)
  )

  def _sol(sgn):
    sqrt0 = mx.sqrt(mx.maximum(mujoco.mjMINVAL, sqlen0 - sqrad))
    sqrt1 = mx.sqrt(mx.maximum(mujoco.mjMINVAL, sqlen1 - sqrad))

    d00 = (d[0] * sqrad + sgn * rad * d[1] * sqrt0) / mx.maximum(
        mujoco.mjMINVAL, sqlen0
    )
    d01 = (d[1] * sqrad - sgn * rad * d[0] * sqrt0) / mx.maximum(
        mujoco.mjMINVAL, sqlen0
    )
    d10 = (d[2] * sqrad - sgn * rad * d[3] * sqrt1) / mx.maximum(
        mujoco.mjMINVAL, sqlen1
    )
    d11 = (d[3] * sqrad + sgn * rad * d[2] * sqrt1) / mx.maximum(
        mujoco.mjMINVAL, sqlen1
    )

    sol = mx.array([[d00, d01], [d10, d11]])

    tmp0 = sol[0] + sol[1]
    tmp0 = mjx_math.normalize(tmp0).reshape(-1)
    good0 = mx.sum(tmp0 * sd)

    tmp1 = (sol[0] - sol[1]).reshape(-1)
    good1 = -mx.sum(tmp1 * tmp1)

    good = mx.where(sidesite, good0, good1)

    intersect = _is_intersect(d[:2], sol[0], d[2:], sol[1])
    good = mx.where(intersect, mx.array(-10000.0), good)

    return sol, good

  # Evaluate both solutions
  sol_pos, good_pos = _sol(mx.array(1.0))
  sol_neg, good_neg = _sol(mx.array(-1.0))

  # Select the better solution
  use_neg = good_neg > good_pos
  sol = mx.where(use_neg, sol_neg, sol_pos)
  i = mx.where(use_neg, mx.array(1), mx.array(0))
  pnt = sol.reshape(-1)

  # check for intersection
  intersect = _is_intersect(d[:2], pnt[:2], d[2:], pnt[2:])

  # compute curve length
  wlen = _length_circle(sol[0], sol[1], i, rad)

  # check cases
  invalid = (
      point_inside0
      | point_inside1
      | circle_too_small
      | points_too_close
      | intersect_and_side
      | intersect
  )

  wlen = mx.where(invalid, mx.array(-1.0), wlen)
  pnt = mx.where(invalid, mx.zeros(4), pnt)

  return wlen, pnt


def wrap_inside(
    end: mx.array,
    radius: mx.array,
    maxiter: int,
    tolerance: float,
    z_init: float,
) -> Tuple[mx.array, mx.array]:
  """Compute 2D inside wrap point."""
  mjMINVAL = mujoco.mjMINVAL

  len0 = mjx_math.norm(end[:2])
  len1 = mjx_math.norm(end[2:])
  dif = mx.array([end[2] - end[0], end[3] - end[1]])
  dd = dif[0] * dif[0] + dif[1] * dif[1]

  no_wrap0 = (
      (len0 <= radius)
      | (len1 <= radius)
      | (radius < mjMINVAL)
      | (len0 < mjMINVAL)
      | (len1 < mjMINVAL)
  )

  a = -1 * (dif[0] * end[0] + dif[1] * end[1]) / mx.maximum(mjMINVAL, dd)
  tmp = end[:2] + a * dif
  no_wrap1 = (dd > mjMINVAL) & (a > 0) & (a < 1) & (mjx_math.norm(tmp) <= radius)

  pnt_avg = 0.5 * mx.array([end[0] + end[2], end[1] + end[3]])
  pnt_avg = radius * mjx_math.normalize(pnt_avg)

  A = radius / mx.maximum(mjMINVAL, len0)
  B = radius / mx.maximum(mjMINVAL, len1)
  cosG = (len0 * len0 + len1 * len1 - dd) / mx.maximum(mjMINVAL, 2 * len0 * len1)

  no_wrap2 = cosG < -1 + mjMINVAL
  early_return0 = cosG > 1 - mjMINVAL

  G = mx.arccos(cosG)

  z = mx.array([z_init])
  f = mx.arcsin(A * z) + mx.arcsin(B * z) - 2 * mx.arcsin(z) + G

  early_return1 = f > 0

  # Newton iteration (Python loop replacing jax.lax.scan)
  status = mx.array([False])
  for _ in range(maxiter):
    converged = mx.abs(f) <= tolerance
    df = (
        A / mx.maximum(mjMINVAL, mx.sqrt(1 - z * z * A * A))
        + B / mx.maximum(mjMINVAL, mx.sqrt(1 - z * z * B * B))
        - 2 / mx.maximum(mjMINVAL, mx.sqrt(1 - z * z))
    )
    status0 = df > -mjMINVAL
    z_next = z - (1 - converged) * mjx_math.safe_div(f, df)
    status1 = z_next > z
    f_next = mx.arcsin(A * z_next) + mx.arcsin(B * z_next) - 2 * mx.arcsin(z_next) + G
    status2 = f_next > tolerance
    status = status | status0 | status1 | status2
    z = z_next
    f = f_next

  early_return2 = status

  sign = end[0] * end[3] - end[1] * end[2] > 0
  vec = mx.where(sign, end[:2], end[2:])
  vec = mjx_math.normalize(vec)
  ang = mx.arcsin(z) - mx.where(sign, mx.arcsin(A * z), mx.arcsin(B * z))
  pnt_sol = radius * mx.array([
      mx.cos(ang) * vec[0] - mx.sin(ang) * vec[1],
      mx.sin(ang) * vec[0] + mx.cos(ang) * vec[1],
  ]).reshape(-1)

  no_wrap = no_wrap0 | no_wrap1 | no_wrap2
  early_return = early_return0 | early_return1 | early_return2
  status_out = -1 * no_wrap * mx.ones(1)

  pnt = mx.where(early_return, pnt_avg, pnt_sol)
  pnt = mx.where(no_wrap, mx.zeros(2), pnt)

  return status_out, mx.concatenate([pnt, pnt])


def wrap(
    x0: mx.array,
    x1: mx.array,
    xpos: mx.array,
    xmat: mx.array,
    size: mx.array,
    side: mx.array,
    sidesite: mx.array,
    is_sphere: mx.array,
    is_wrap_inside: bool,
    wrap_inside_maxiter: int,
    wrap_inside_tolerance: float,
    wrap_inside_z_init: float,
) -> Tuple[mx.array, mx.array, mx.array]:
  """Wrap tendon around sphere or cylinder."""
  p0 = xmat.T @ (x0 - xpos)
  p1 = xmat.T @ (x1 - xpos)

  close_to_origin = (mjx_math._norm_raw(p0) < mujoco.mjMINVAL) | (
      mjx_math._norm_raw(p1) < mujoco.mjMINVAL
  )

  axis0 = mjx_math.normalize(p0)

  normal = mjx_math._cross(p0, p1)
  normal, nrm = mjx_math.normalize_with_norm(normal)

  axis_alt = mx.ones(3)
  # set the element with max |axis0| to 0
  idx = int(mx.argmax(mx.abs(axis0)))
  axis_alt_np = np.ones(3)
  axis_alt_np[idx] = 0.0
  axis_alt = mx.array(axis_alt_np)
  normal_alt = mjx_math.normalize(mjx_math._cross(axis0, axis_alt))
  normal = mx.where(nrm < mujoco.mjMINVAL, normal_alt, normal)

  axis1 = mjx_math.normalize(mjx_math._cross(normal, axis0))

  axis0 = mx.where(is_sphere, axis0, mx.array([1.0, 0.0, 0.0]))
  axis1 = mx.where(is_sphere, axis1, mx.array([0.0, 1.0, 0.0]))

  d_vec = mx.array([
      mx.sum(p0 * axis0),
      mx.sum(p0 * axis1),
      mx.sum(p1 * axis0),
      mx.sum(p1 * axis1),
  ])

  s = xmat.T @ (side - xpos)
  sd = mx.array([mx.sum(s * axis0), mx.sum(s * axis1)])
  sd = mjx_math.normalize(sd) * size

  if is_wrap_inside:
    wlen, pnt = wrap_inside(
        d_vec, size, wrap_inside_maxiter, wrap_inside_tolerance, wrap_inside_z_init
    )
  else:
    wlen, pnt = wrap_circle(d_vec, sd, sidesite, size)

  no_wrap = wlen < 0

  res0 = axis0 * pnt[0] + axis1 * pnt[1]
  res1 = axis0 * pnt[2] + axis1 * pnt[3]
  res = mx.concatenate([res0, res1])

  # cylinder correction
  l0 = mx.sqrt(
      (p0[0] - res[0]) ** 2 + (p0[1] - res[1]) ** 2
  )
  l1 = mx.sqrt(
      (p1[0] - res[3]) ** 2 + (p1[1] - res[4]) ** 2
  )
  r2 = p0[2] + (p1[2] - p0[2]) * mjx_math.safe_div(l0, l0 + wlen + l1)
  r5 = p0[2] + (p1[2] - p0[2]) * mjx_math.safe_div(l0 + wlen, l0 + wlen + l1)
  height = mx.abs(r5 - r2)

  wlen = mx.where(is_sphere, wlen, mx.sqrt(wlen * wlen + height * height))

  # For cylinder, update res[2] and res[5]
  res_np = np.array(res)
  r2_val = float(r2)
  r5_val = float(r5)
  res_cyl = res_np.copy()
  res_cyl[2] = r2_val
  res_cyl[5] = r5_val
  res = mx.where(is_sphere, res, mx.array(res_cyl))

  wpnt0 = xmat @ res[:3] + xpos
  wpnt1 = xmat @ res[3:] + xpos

  invalid = close_to_origin | no_wrap

  wlen = mx.where(invalid, mx.array(-1.0), wlen)
  wpnt0 = mx.where(invalid, mx.zeros(3), wpnt0)
  wpnt1 = mx.where(invalid, mx.zeros(3), wpnt1)

  return wlen, wpnt0, wpnt1


# ---------------------------------------------------------------------------
# Muscle helpers
# ---------------------------------------------------------------------------


def muscle_gain_length(
    length: mx.array, lmin: mx.array, lmax: mx.array
) -> mx.array:
  """Normalized muscle length-gain curve."""
  a = 0.5 * (lmin + 1)
  b = 0.5 * (1 + lmax)

  out0 = 0.5 * mx.square(
      (length - lmin) / mx.maximum(mujoco.mjMINVAL, a - lmin)
  )
  out1 = 1 - 0.5 * mx.square(
      (1 - length) / mx.maximum(mujoco.mjMINVAL, 1 - a)
  )
  out2 = 1 - 0.5 * mx.square(
      (length - 1) / mx.maximum(mujoco.mjMINVAL, b - 1)
  )
  out3 = 0.5 * mx.square(
      (lmax - length) / mx.maximum(mujoco.mjMINVAL, lmax - b)
  )

  out = mx.where(length <= b, out2, out3)
  out = mx.where(length <= 1, out1, out)
  out = mx.where(length <= a, out0, out)
  out = mx.where((lmin <= length) & (length <= lmax), out, 0.0)

  return out


def muscle_gain(
    length: mx.array,
    vel: mx.array,
    lengthrange: mx.array,
    acc0: mx.array,
    prm: mx.array,
) -> mx.array:
  """Muscle active force."""
  lrange = prm[:2]
  force, scale, lmin, lmax, vmax, _, fvmax = prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8]

  force = mx.where(force < 0, scale / mx.maximum(mujoco.mjMINVAL, acc0), force)

  L0 = (lengthrange[1] - lengthrange[0]) / mx.maximum(
      mujoco.mjMINVAL, lrange[1] - lrange[0]
  )

  L = lrange[0] + (length - lengthrange[0]) / mx.maximum(mujoco.mjMINVAL, L0)
  V = vel / mx.maximum(mujoco.mjMINVAL, L0 * vmax)

  FL = muscle_gain_length(L, lmin, lmax)

  y = fvmax - 1
  FV = mx.where(
      V <= y,
      fvmax - mx.square(y - V) / mx.maximum(mujoco.mjMINVAL, y),
      fvmax,
  )
  FV = mx.where(V <= 0, mx.square(V + 1), FV)
  FV = mx.where(V <= -1, mx.array(0.0), FV)

  return -force * FL * FV


def muscle_bias(
    length: mx.array, lengthrange: mx.array, acc0: mx.array, prm: mx.array
) -> mx.array:
  """Muscle passive force."""
  lrange = prm[:2]
  force, scale, _, lmax, _, fpmax = prm[2], prm[3], prm[4], prm[5], prm[6], prm[7]

  force = mx.where(force < 0, scale / mx.maximum(mujoco.mjMINVAL, acc0), force)

  L0 = (lengthrange[1] - lengthrange[0]) / mx.maximum(
      mujoco.mjMINVAL, lrange[1] - lrange[0]
  )

  L = lrange[0] + (length - lengthrange[0]) / mx.maximum(mujoco.mjMINVAL, L0)

  b = 0.5 * (1 + lmax)

  out1 = (
      -force
      * fpmax
      * 0.5
      * mx.square((L - 1) / mx.maximum(mujoco.mjMINVAL, b - 1))
  )
  out2 = -force * fpmax * (0.5 + (L - b) / mx.maximum(mujoco.mjMINVAL, b - 1))

  out = mx.where(L <= b, out1, out2)
  out = mx.where(L <= 1, 0.0, out)

  return out


def muscle_dynamics_timescale(
    dctrl: mx.array,
    tau_act: mx.array,
    tau_deact: mx.array,
    smoothing_width: mx.array,
) -> mx.array:
  """Muscle time constant with optional smoothing."""
  tau_hard = mx.where(dctrl > 0, tau_act, tau_deact)

  def _sigmoid(x):
    sol = x * x * x * (3 * x * (2 * x - 5) + 10)
    sol = mx.where(x <= 0, mx.array(0.0), sol)
    sol = mx.where(x >= 1, mx.array(1.0), sol)
    return sol

  tau_smooth = tau_deact + (tau_act - tau_deact) * _sigmoid(
      mjx_math.safe_div(dctrl, smoothing_width) + 0.5
  )

  return mx.where(smoothing_width < mujoco.mjMINVAL, tau_hard, tau_smooth)


def muscle_dynamics(
    ctrl: mx.array, act: mx.array, prm: mx.array
) -> mx.array:
  """Muscle activation dynamics."""
  ctrlclamp = mx.clip(ctrl, 0, 1)
  actclamp = mx.clip(act, 0, 1)

  tau_act = prm[0] * (0.5 + 1.5 * actclamp)
  tau_deact = prm[1] / (0.5 + 1.5 * actclamp)
  smoothing_width = prm[2]
  dctrl = ctrlclamp - act

  tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smoothing_width)

  return dctrl / mx.maximum(mujoco.mjMINVAL, tau)
