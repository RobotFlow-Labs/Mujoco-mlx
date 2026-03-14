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
"""Core smooth dynamics functions -- MLX port.

Ported from JAX (mujoco.mjx._src.smooth) to Apple MLX.

Translation notes:
  - scan.body_tree -> inline Python loops over kinematic tree
  - scan.flat -> inline Python loops over bodies/actuators
  - jax.vmap -> explicit loops with mx.stack
  - jax.Array -> mx.array
  - jp.XXX -> mx.XXX
  - jax.scipy.linalg -> scipy.linalg (for factor_m/solve_m)
"""

import dataclasses

import mlx.core as mx
import mujoco
from mujoco.mjx_mlx._src import math as mjx_math
from mujoco.mjx_mlx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.types import CamLightType
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DisableBit
from mujoco.mjx_mlx._src.types import EqType
from mujoco.mjx_mlx._src.types import JointType
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import ObjType
from mujoco.mjx_mlx._src.types import TrnType
from mujoco.mjx_mlx._src.types import WrapType
# pylint: enable=g-importing-member
import numpy as np
import scipy.linalg as sp_linalg


# ---------------------------------------------------------------------------
# Internal: tree traversal helpers
# ---------------------------------------------------------------------------

def _body_tree_depths(m: Model) -> np.ndarray:
  """Compute depth of each body in the kinematic tree."""
  depths = np.zeros(m.nbody, dtype=np.int32)
  for i in range(1, m.nbody):
    depths[i] = depths[m.body_parentid[i]] + 1
  return depths


def _body_joints(m: Model, body_id: int):
  """Return joint indices for a body."""
  adr = m.body_jntadr[body_id]
  num = m.body_jntnum[body_id]
  return list(range(adr, adr + num))


def _body_dofs(m: Model, body_id: int):
  """Return dof indices for a body."""
  adr = m.body_dofadr[body_id]
  num = m.body_dofnum[body_id]
  return list(range(adr, adr + num))


# ---------------------------------------------------------------------------
# Convenience: replace fields on MLXNode dataclasses
# ---------------------------------------------------------------------------

def _replace(obj, **kwargs):
  """dataclasses.replace wrapper."""
  return dataclasses.replace(obj, **kwargs)


def _tree_replace(obj, params: dict):
  """Replace nested fields using dot/underscore-separated keys.

  Supports paths like '_impl.cinert' which means obj._impl.cinert.
  """
  new = obj
  for k, v in params.items():
    parts = k.split('.')
    if len(parts) == 1:
      new = _replace(new, **{parts[0]: v})
    elif len(parts) == 2:
      sub = getattr(new, parts[0])
      sub = _replace(sub, **{parts[1]: v})
      new = _replace(new, **{parts[0]: sub})
    elif len(parts) == 3:
      sub1 = getattr(new, parts[0])
      sub2 = getattr(sub1, parts[1])
      sub2 = _replace(sub2, **{parts[2]: v})
      sub1 = _replace(sub1, **{parts[1]: sub2})
      new = _replace(new, **{parts[0]: sub1})
    else:
      raise ValueError(f'Unsupported depth in tree_replace key: {k}')
  return new


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------


def kinematics(m: Model, d: Data) -> Data:
  """Converts position/velocity from generalized coordinates to maximal."""

  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0

  # Per-body results
  xpos_list = [None] * m.nbody
  xquat_list = [None] * m.nbody
  xmat_list = [None] * m.nbody
  xanchor_list = [[] for _ in range(m.nbody)]
  xaxis_list = [[] for _ in range(m.nbody)]
  qpos_arr = np.array(d.qpos)  # mutable copy for normalization

  for depth in range(max_depth + 1):
    for body_id in range(m.nbody):
      if depths[body_id] != depth:
        continue

      pos = mx.array(m.body_pos[body_id])
      quat = mx.array(m.body_quat[body_id])

      if body_id > 0:
        parent_id = m.body_parentid[body_id]
        parent_pos = xpos_list[parent_id]
        parent_quat = xquat_list[parent_id]
        pos = parent_pos + mjx_math.rotate(pos, parent_quat)
        quat = mjx_math.quat_mul(parent_quat, quat)

      jnts = _body_joints(m, body_id)
      qpos = mx.array(qpos_arr)

      anchors = []
      axes = []

      for j in jnts:
        jnt_typ = m.jnt_type[j]
        jnt_pos_j = mx.array(m.jnt_pos[j])
        jnt_axis_j = mx.array(m.jnt_axis[j])
        qpos_i = m.jnt_qposadr[j]

        if jnt_typ == JointType.FREE:
          anchor = qpos[qpos_i : qpos_i + 3]
          axis = mx.array([0.0, 0.0, 1.0])
        else:
          anchor = mjx_math.rotate(jnt_pos_j, quat) + pos
          axis = mjx_math.rotate(jnt_axis_j, quat)
        anchors.append(anchor)
        axes.append(axis)

        if jnt_typ == JointType.FREE:
          pos = qpos[qpos_i : qpos_i + 3]
          q_raw = qpos[qpos_i + 3 : qpos_i + 7]
          quat = mjx_math.normalize(q_raw)
          q_norm = np.array(quat)
          qpos_arr[qpos_i + 3 : qpos_i + 7] = q_norm
        elif jnt_typ == JointType.BALL:
          qloc = mjx_math.normalize(qpos[qpos_i : qpos_i + 4])
          q_norm = np.array(qloc)
          qpos_arr[qpos_i : qpos_i + 4] = q_norm
          quat = mjx_math.quat_mul(quat, qloc)
          pos = anchor - mjx_math.rotate(jnt_pos_j, quat)
        elif jnt_typ == JointType.HINGE:
          angle = qpos[qpos_i] - mx.array(m.qpos0[qpos_i])
          qloc = mjx_math.axis_angle_to_quat(jnt_axis_j, angle)
          quat = mjx_math.quat_mul(quat, qloc)
          pos = anchor - mjx_math.rotate(jnt_pos_j, quat)
        elif jnt_typ == JointType.SLIDE:
          pos = pos + axis * (qpos[qpos_i] - mx.array(m.qpos0[qpos_i]))
        else:
          raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

      mat = mjx_math.quat_to_mat(quat)

      xpos_list[body_id] = pos
      xquat_list[body_id] = quat
      xmat_list[body_id] = mat
      xanchor_list[body_id] = anchors
      xaxis_list[body_id] = axes

  xpos = mx.stack(xpos_list)
  xquat = mx.stack(xquat_list)
  xmat = mx.stack(xmat_list)

  # Flatten anchors and axes to joint-indexed arrays
  all_anchors = []
  all_axes = []
  for body_id in range(m.nbody):
    all_anchors.extend(xanchor_list[body_id])
    all_axes.extend(xaxis_list[body_id])

  if all_anchors:
    xanchor = mx.stack(all_anchors)
    xaxis = mx.stack(all_axes)
  else:
    xanchor = mx.zeros((0, 3))
    xaxis = mx.zeros((0, 3))

  qpos_final = mx.array(qpos_arr)

  # Handle mocap bodies
  if m.nmocap:
    xpos_np = np.array(xpos)
    xquat_np = np.array(xquat)
    xmat_np = np.array(xmat)
    for i in range(m.nbody):
      if m.body_mocapid[i] >= 0:
        mid = m.body_mocapid[i]
        xpos_np[i] = np.array(d.mocap_pos[mid])
        mq = mjx_math.normalize(d.mocap_quat[mid])
        xquat_np[i] = np.array(mq)
        xmat_np[i] = np.array(mjx_math.quat_to_mat(mq))
    xpos = mx.array(xpos_np)
    xquat = mx.array(xquat_np)
    xmat = mx.array(xmat_np)

  # Compute xipos, ximat
  xipos_list = []
  ximat_list = []
  for i in range(m.nbody):
    p, mat = support.local_to_global(
        xpos[i], xquat[i],
        mx.array(m.body_ipos[i]), mx.array(m.body_iquat[i]),
    )
    xipos_list.append(p)
    ximat_list.append(mat)
  xipos = mx.stack(xipos_list)
  ximat = mx.stack(ximat_list)

  d = _replace(d, qpos=qpos_final, xanchor=xanchor, xaxis=xaxis, xpos=xpos)
  d = _replace(d, xquat=xquat, xmat=xmat, xipos=xipos, ximat=ximat)

  # Geom positions
  if m.ngeom:
    gxpos_list = []
    gxmat_list = []
    for i in range(m.ngeom):
      bid = m.geom_bodyid[i]
      p, mat = support.local_to_global(
          xpos[bid], xquat[bid],
          mx.array(m.geom_pos[i]), mx.array(m.geom_quat[i]),
      )
      gxpos_list.append(p)
      gxmat_list.append(mat)
    d = _replace(
        d, geom_xpos=mx.stack(gxpos_list), geom_xmat=mx.stack(gxmat_list)
    )

  # Site positions
  if m.nsite:
    sxpos_list = []
    sxmat_list = []
    for i in range(m.nsite):
      bid = m.site_bodyid[i]
      p, mat = support.local_to_global(
          xpos[bid], xquat[bid],
          mx.array(m.site_pos[i]), mx.array(m.site_quat[i]),
      )
      sxpos_list.append(p)
      sxmat_list.append(mat)
    d = _replace(
        d, site_xpos=mx.stack(sxpos_list), site_xmat=mx.stack(sxmat_list)
    )

  return d


# ---------------------------------------------------------------------------
# com_pos
# ---------------------------------------------------------------------------


def com_pos(m: Model, d: Data) -> Data:
  """Maps inertias and motion dofs to global frame centered at subtree-CoM."""

  # Compute subtree center of mass (reverse tree scan)
  pos_arr = np.array(d.xipos) * np.array(m.body_mass)[:, None]
  mass_arr = np.array(m.body_mass).copy()

  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0

  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] == depth:
        parent = m.body_parentid[b]
        pos_arr[parent] += pos_arr[b]
        mass_arr[parent] += mass_arr[b]

  subtree_com_np = np.zeros_like(pos_arr)
  xipos_np = np.array(d.xipos)
  for i in range(m.nbody):
    if mass_arr[i] < mujoco.mjMINVAL:
      subtree_com_np[i] = xipos_np[i]
    else:
      subtree_com_np[i] = pos_arr[i] / max(mass_arr[i], mujoco.mjMINVAL)

  subtree_com = mx.array(subtree_com_np)
  d = _replace(d, subtree_com=subtree_com)

  # Map inertias to frame centered at subtree_com
  cinert_list = []
  root_com = subtree_com[mx.array(m.body_rootid)]
  for i in range(m.nbody):
    inert = mx.array(m.body_inertia[i])
    ximat_i = d.ximat[i]
    off = d.xipos[i] - root_com[i]
    mass_i = mx.array(m.body_mass[i])

    # h = cross(off, -I) as 3x3 matrix
    h_cols = [mjx_math._cross(off, -mx.eye(3)[k]) for k in range(3)]
    h = mx.stack(h_cols, axis=1)  # 3x3

    inert_mat = mjx_math.matmul_unroll(ximat_i * inert, ximat_i.T)
    inert_mat = inert_mat + mjx_math.matmul_unroll(h, h.T) * mass_i

    # Extract upper triangular: (0,0),(1,1),(2,2),(0,1),(0,2),(1,2)
    inert_triu = mx.array([
        inert_mat[0, 0], inert_mat[1, 1], inert_mat[2, 2],
        inert_mat[0, 1], inert_mat[0, 2], inert_mat[1, 2],
    ])
    cinert_i = mx.concatenate([inert_triu, off * mass_i, mass_i[None]])
    cinert_list.append(cinert_i)

  cinert = mx.stack(cinert_list)
  d = _tree_replace(d, {'_impl.cinert': cinert})

  # Map motion dofs to global frame centered at subtree_com
  cdof_list = []
  for body_id in range(m.nbody):
    jnts = _body_joints(m, body_id)
    rc = root_com[body_id]

    for j in jnts:
      jnt_typ = m.jnt_type[j]
      offset = rc - d.xanchor[j]

      if jnt_typ == JointType.FREE:
        for k in range(3):
          cdof_np = np.zeros(6)
          cdof_np[3 + k] = 1.0
          cdof_list.append(mx.array(cdof_np))
        for k in range(3):
          a = d.xmat[body_id].T[k]
          cdof_k = mx.concatenate([a, mjx_math._cross(a, offset)])
          cdof_list.append(cdof_k)
      elif jnt_typ == JointType.BALL:
        for k in range(3):
          a = d.xmat[body_id].T[k]
          cdof_k = mx.concatenate([a, mjx_math._cross(a, offset)])
          cdof_list.append(cdof_k)
      elif jnt_typ == JointType.HINGE:
        a = d.xaxis[j]
        cdof_k = mx.concatenate([a, mjx_math._cross(a, offset)])
        cdof_list.append(cdof_k)
      elif jnt_typ == JointType.SLIDE:
        cdof_k = mx.concatenate([mx.zeros(3), d.xaxis[j]])
        cdof_list.append(cdof_k)
      else:
        raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

  if cdof_list:
    cdof = mx.stack(cdof_list)
  else:
    cdof = mx.zeros((m.nv, 6))

  d = _replace(d, cdof=cdof)
  return d


# ---------------------------------------------------------------------------
# camlight
# ---------------------------------------------------------------------------


def camlight(m: Model, d: Data) -> Data:
  """Computes camera and light positions and orientations."""
  if m.ncam == 0:
    return _replace(
        d, cam_xpos=mx.zeros((0, 3)), cam_xmat=mx.zeros((0, 3, 3))
    )

  is_target_cam = (m.cam_mode == CamLightType.TARGETBODY) | (
      m.cam_mode == CamLightType.TARGETBODYCOM
  )
  cam_mode = np.where(
      is_target_cam & (m.cam_targetbodyid < 0),
      CamLightType.FIXED,
      m.cam_mode,
  )

  cam_xpos_list = []
  cam_xmat_list = []
  for i in range(m.ncam):
    p, mat = support.local_to_global(
        d.xpos[m.cam_bodyid[i]], d.xquat[m.cam_bodyid[i]],
        mx.array(m.cam_pos[i]), mx.array(m.cam_quat[i]),
    )
    mode = cam_mode[i]
    if mode == CamLightType.TRACK:
      mat = mx.array(m.cam_mat0[i])
      p = d.xpos[m.cam_bodyid[i]] + mx.array(m.cam_pos0[i])
    elif mode == CamLightType.TRACKCOM:
      mat = mx.array(m.cam_mat0[i])
      p = d.subtree_com[m.cam_bodyid[i]] + mx.array(m.cam_poscom0[i])
    elif mode in (CamLightType.TARGETBODY, CamLightType.TARGETBODYCOM):
      target_pos = d.xpos[m.cam_targetbodyid[i]]
      if mode == CamLightType.TARGETBODYCOM:
        target_pos = d.subtree_com[m.cam_targetbodyid[i]]
      mat_3 = mjx_math.normalize(p - target_pos)
      mat_1 = mjx_math.normalize(
          mjx_math._cross(mx.array([0.0, 0.0, 1.0]), mat_3)
      )
      mat_2 = mjx_math.normalize(mjx_math._cross(mat_3, mat_1))
      mat = mx.stack([mat_1, mat_2, mat_3]).T
    cam_xpos_list.append(p)
    cam_xmat_list.append(mat)

  d = _replace(
      d,
      cam_xpos=mx.stack(cam_xpos_list),
      cam_xmat=mx.stack(cam_xmat_list),
  )
  return d


# ---------------------------------------------------------------------------
# crb (composite rigid body)
# ---------------------------------------------------------------------------


def crb(m: Model, d: Data) -> Data:
  """Runs composite rigid body inertia algorithm."""
  crb_body = np.array(d._impl.cinert).copy()

  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0

  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] == depth:
        parent = m.body_parentid[b]
        crb_body[parent] += crb_body[b]

  crb_body[0] = 0.0
  crb_body_mx = mx.array(crb_body)
  d = _tree_replace(d, {'_impl.crb': crb_body_mx})

  # Compute mass matrix
  crb_cdof_list = []
  for dof in range(m.nv):
    crb_cdof_list.append(
        mjx_math.inert_mul(crb_body_mx[m.dof_bodyid[dof]], d.cdof[dof])
    )
  crb_cdof = mx.stack(crb_cdof_list) if crb_cdof_list else mx.zeros((0, 6))

  qm = support.make_m(m, crb_cdof, d.cdof, mx.array(m.dof_armature))
  d = _tree_replace(d, {'_impl.qM': qm})
  return d


# ---------------------------------------------------------------------------
# factor_m
# ---------------------------------------------------------------------------


def factor_m(m: Model, d: Data) -> Data:
  """Gets factorization of inertia-like matrix M, assumed spd."""

  if not support.is_sparse(m):
    qM_np = np.array(d._impl.qM)
    qh = sp_linalg.cholesky(qM_np, lower=True)
    d = _tree_replace(d, {'_impl.qLD': mx.array(qh)})
    return d

  # Sparse factorization
  depth = []
  for i in range(m.nv):
    depth.append(
        depth[m.dof_parentid[i]] + 1 if m.dof_parentid[i] != -1 else 0
    )

  updates = {}
  madr_ds = []
  for i in range(m.nv):
    madr_d = madr_ij = m.dof_Madr[i]
    j = i
    while True:
      madr_ds.append(madr_d)
      madr_ij, j = madr_ij + 1, m.dof_parentid[j]
      if j == -1:
        break
      out_beg, out_end = tuple(m.dof_Madr[j : j + 2])
      updates.setdefault(depth[j], []).append(
          (out_beg, out_end, madr_d, madr_ij)
      )

  qld = np.array(d._impl.qM).astype(np.float64).copy()

  for _, upd_list in sorted(updates.items(), reverse=True):
    for b, e, madr_d, madr_ij in upd_list:
      for offset in range(e - b):
        row = madr_ij + offset
        out_idx = b + offset
        qld[out_idx] += -(qld[madr_ij] / qld[madr_d]) * qld[row]

  qld_diag = qld[m.dof_Madr].copy()
  for k in range(len(madr_ds)):
    if qld[madr_ds[k]] != 0:
      qld[k] /= qld[madr_ds[k]]
  qld[m.dof_Madr] = qld_diag

  d = _tree_replace(d, {
      '_impl.qLD': mx.array(qld.astype(np.float32)),
      '_impl.qLDiagInv': mx.array((1.0 / qld_diag).astype(np.float32)),
  })
  return d


# ---------------------------------------------------------------------------
# solve_m
# ---------------------------------------------------------------------------


def solve_m(m: Model, d: Data, x: mx.array) -> mx.array:
  """Computes sparse backsubstitution: x = inv(L'*D*L)*y."""

  if not support.is_sparse(m):
    qLD_np = np.array(d._impl.qLD)
    x_np = np.array(x)
    result = sp_linalg.cho_solve((qLD_np, True), x_np)
    return mx.array(result)

  depth = []
  for i in range(m.nv):
    depth.append(
        depth[m.dof_parentid[i]] + 1 if m.dof_parentid[i] != -1 else 0
    )

  updates_i, updates_j = {}, {}
  for i in range(m.nv):
    madr_ij, j = m.dof_Madr[i], i
    while True:
      madr_ij, j = madr_ij + 1, m.dof_parentid[j]
      if j == -1:
        break
      updates_i.setdefault(depth[i], []).append((i, madr_ij, j))
      updates_j.setdefault(depth[j], []).append((j, madr_ij, i))

  x_np = np.array(x).astype(np.float64).copy()
  qLD_np = np.array(d._impl.qLD).astype(np.float64)

  # x <- inv(L') * x
  for _, vals in sorted(updates_j.items(), reverse=True):
    for j, madr_ij, i in vals:
      x_np[j] += -qLD_np[madr_ij] * x_np[i]

  # x <- inv(D) * x
  qLDiagInv_np = np.array(d._impl.qLDiagInv).astype(np.float64)
  x_np *= qLDiagInv_np

  # x <- inv(L) * x
  for _, vals in sorted(updates_i.items()):
    for i, madr_ij, j in vals:
      x_np[i] += -qLD_np[madr_ij] * x_np[j]

  return mx.array(x_np.astype(np.float32))


# ---------------------------------------------------------------------------
# com_vel
# ---------------------------------------------------------------------------


def com_vel(m: Model, d: Data) -> Data:
  """Computes cvel, cdof_dot."""

  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0

  cvel_list = [None] * m.nbody
  cdof_dot_arr = [None] * m.nv

  for depth in range(max_depth + 1):
    for body_id in range(m.nbody):
      if depths[body_id] != depth:
        continue

      if body_id == 0 or m.body_parentid[body_id] == -1:
        cvel = mx.zeros(6)
      else:
        cvel = cvel_list[m.body_parentid[body_id]]
        if cvel is None:
          cvel = mx.zeros(6)

      jnts = _body_joints(m, body_id)

      for j in jnts:
        jnt_typ = m.jnt_type[j]
        dof_width = JointType(jnt_typ).dof_width()
        dof_adr = m.jnt_dofadr[j]

        if jnt_typ == JointType.FREE:
          # Translation dofs (first 3): zero cdof_dot
          for k in range(3):
            di = dof_adr + k
            cvel = cvel + d.cdof[di] * d.qvel[di]
            cdof_dot_arr[di] = mx.zeros(6)
          # Angular dofs (next 3): motion_cross
          for k in range(3):
            di = dof_adr + 3 + k
            cdof_dot_arr[di] = mjx_math.motion_cross(cvel, d.cdof[di])
            cvel = cvel + d.cdof[di] * d.qvel[di]
        else:
          for k in range(dof_width):
            di = dof_adr + k
            cdof_dot_arr[di] = mjx_math.motion_cross(cvel, d.cdof[di])
            cvel = cvel + d.cdof[di] * d.qvel[di]

      cvel_list[body_id] = cvel

  cvel = mx.stack(cvel_list) if cvel_list else mx.zeros((m.nbody, 6))

  if m.nv > 0 and all(x is not None for x in cdof_dot_arr):
    cdof_dot = mx.stack(cdof_dot_arr)
  else:
    cdof_dot = mx.zeros((m.nv, 6))

  d = _replace(d, cvel=cvel, cdof_dot=cdof_dot)
  return d


# ---------------------------------------------------------------------------
# subtree_vel
# ---------------------------------------------------------------------------


def subtree_vel(m: Model, d: Data) -> Data:
  """Subtree linear velocity and angular momentum."""

  body_vel_list = []
  subtree_linvel_np = np.zeros((m.nbody, 3))
  subtree_angmom_np = np.zeros((m.nbody, 3))

  for i in range(m.nbody):
    cvel_i = d.cvel[i]
    ang, lin = cvel_i[:3], cvel_i[3:]
    xipos_i = d.xipos[i]
    subtree_com_root_i = d.subtree_com[m.body_rootid[i]]
    mass_i = float(m.body_mass[i])
    inertia_i = mx.array(m.body_inertia[i])
    ximat_i = d.ximat[i]

    lin = lin - mjx_math._cross(xipos_i - subtree_com_root_i, ang)

    subtree_linvel_np[i] = np.array(lin) * mass_i
    subtree_angmom_np[i] = np.array(inertia_i * ximat_i @ ximat_i.T @ ang)
    body_vel_list.append(mx.concatenate([ang, lin]))

  # Sum body linear momentum up the tree (reverse scan)
  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0

  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] == depth:
        parent = m.body_parentid[b]
        subtree_linvel_np[parent] += subtree_linvel_np[b]

  body_subtreemass = np.array(m.body_subtreemass)
  for i in range(m.nbody):
    denom = max(mujoco.mjMINVAL, body_subtreemass[i])
    subtree_linvel_np[i] /= denom

  subtree_linvel_mx = mx.array(subtree_linvel_np.astype(np.float32))

  # Subtree angular momentum (reverse tree scan)
  angmom_out = subtree_angmom_np.copy()
  mom_parent_arr = np.zeros((m.nbody, 3))

  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] != depth or b == 0:
        continue
      parent = m.body_parentid[b]

      xipos_b = np.array(d.xipos[b])
      com_b = np.array(d.subtree_com[b])
      com_parent = np.array(d.subtree_com[parent])
      linvel_b = subtree_linvel_np[b]
      linvel_parent = subtree_linvel_np[parent]
      mass_b = float(m.body_mass[b])
      body_vel_b = np.array(body_vel_list[b])
      stm_b = float(m.body_subtreemass[b])

      dx = xipos_b - com_b
      dv = body_vel_b[3:] - linvel_b
      mom = np.cross(dx, dv * mass_b)

      dx_p = com_b - com_parent
      dv_p = linvel_b - linvel_parent
      mom_p = np.cross(dx_p, dv_p * stm_b)

      angmom_out[parent] += angmom_out[b] + mom + mom_parent_arr[b]
      mom_parent_arr[parent] += mom_p

  subtree_angmom_mx = mx.array(angmom_out.astype(np.float32))

  d = _tree_replace(d, {
      '_impl.subtree_linvel': subtree_linvel_mx,
      '_impl.subtree_angmom': subtree_angmom_mx,
  })
  return d


# ---------------------------------------------------------------------------
# rne (recursive Newton-Euler)
# ---------------------------------------------------------------------------


def rne(m: Model, d: Data, flg_acc: bool = False) -> Data:
  """Computes inverse dynamics using the recursive Newton-Euler algorithm."""

  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0

  cacc_list = [None] * m.nbody

  for depth in range(max_depth + 1):
    for body_id in range(m.nbody):
      if depths[body_id] != depth:
        continue

      if body_id == 0 or m.body_parentid[body_id] == -1:
        if m.opt.disableflags & DisableBit.GRAVITY:
          cacc = mx.zeros(6)
        else:
          cacc = mx.concatenate([mx.zeros(3), -mx.array(m.opt.gravity)])
      else:
        cacc = cacc_list[m.body_parentid[body_id]]

      dofs = _body_dofs(m, body_id)
      for di in dofs:
        cacc = cacc + d.cdof_dot[di] * d.qvel[di]
        if flg_acc:
          cacc = cacc + d.cdof[di] * d.qacc[di]

      cacc_list[body_id] = cacc

  # Compute local forces
  loc_cfrc_list = []
  for i in range(m.nbody):
    cinert_i = d._impl.cinert[i]
    cacc_i = cacc_list[i]
    cvel_i = d.cvel[i]
    frc = mjx_math.inert_mul(cinert_i, cacc_i)
    frc = frc + mjx_math.motion_cross_force(
        cvel_i, mjx_math.inert_mul(cinert_i, cvel_i)
    )
    loc_cfrc_list.append(frc)

  # Backward scan: accumulate body forces
  cfrc = np.array(mx.stack(loc_cfrc_list)).copy()
  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] == depth:
        parent = m.body_parentid[b]
        cfrc[parent] += cfrc[b]
  cfrc_mx = mx.array(cfrc)

  qfrc_bias_list = []
  for dof in range(m.nv):
    qfrc_bias_list.append(
        mx.sum(d.cdof[dof] * cfrc_mx[m.dof_bodyid[dof]])
    )
  qfrc_bias = (
      mx.stack(qfrc_bias_list) if qfrc_bias_list else mx.zeros((m.nv,))
  )

  d = _replace(d, qfrc_bias=qfrc_bias)
  return d


# ---------------------------------------------------------------------------
# rne_postconstraint
# ---------------------------------------------------------------------------


def rne_postconstraint(m: Model, d: Data) -> Data:
  """RNE with complete data: compute cacc, cfrc_ext, cfrc_int."""

  def _transform_force(frc, offset):
    force, torque = frc[:3], frc[3:]
    torque = torque - mjx_math._cross(offset, force)
    return mx.concatenate([torque, force])

  # cfrc_ext = perturb
  cfrc_ext_list = [mx.zeros(6)]  # world body
  for i in range(1, m.nbody):
    offset = d.subtree_com[m.body_rootid[i]] - d.xipos[i]
    cfrc_ext_list.append(_transform_force(d.xfrc_applied[i], offset))
  cfrc_ext = mx.stack(cfrc_ext_list)

  # Contact forces
  forces = []
  condim_idx = []
  dims_seen = (
      set(d._impl.contact.dim)
      if hasattr(d._impl.contact, 'dim') and d._impl.contact.dim is not None
      else set()
  )
  for dim in dims_seen:
    force, idx = support.contact_force_dim(m, d, dim)
    forces.append(force)
    condim_idx.append(idx)

  if forces:
    cfrc_ext_np = np.array(cfrc_ext)
    all_idx = np.concatenate(condim_idx)
    all_forces_mx = mx.concatenate(forces)

    for k in range(all_forces_mx.shape[0]):
      ci = all_idx[k]
      frame = d._impl.contact.frame[ci]
      pos = d._impl.contact.pos[ci]
      g1 = d._impl.contact.geom[ci, 0]
      g2 = d._impl.contact.geom[ci, 1]
      id1 = m.geom_bodyid[int(g1)]
      id2 = m.geom_bodyid[int(g2)]
      com1 = np.array(d.subtree_com[m.body_rootid[id1]])
      com2 = np.array(d.subtree_com[m.body_rootid[id2]])

      frc_world = np.array(
          all_forces_mx[k].reshape((-1, 3)) @ frame
      ).flatten()
      cfrc_com1 = np.array(_transform_force(
          mx.array(frc_world), mx.array(com1 - np.array(pos))
      ))
      cfrc_com2 = np.array(_transform_force(
          mx.array(frc_world), mx.array(com2 - np.array(pos))
      ))

      if id1 != 0:
        cfrc_ext_np[id1] -= cfrc_com1
      if id2 != 0:
        cfrc_ext_np[id2] += cfrc_com2

    cfrc_ext = mx.array(cfrc_ext_np)

  # TODO: equality constraint forces (connect, weld) -- deferred for initial port

  # Forward pass: compute cacc, cfrc_int
  depths = _body_tree_depths(m)
  max_depth = int(depths.max()) if m.nbody > 0 else 0

  cacc_list = [None] * m.nbody
  cfrc_int_list = [None] * m.nbody

  for depth in range(max_depth + 1):
    for body_id in range(m.nbody):
      if depths[body_id] != depth:
        continue

      if body_id == 0 or m.body_parentid[body_id] == -1:
        if m.opt.disableflags & DisableBit.GRAVITY:
          cacc0 = mx.zeros(6)
        else:
          cacc0 = mx.concatenate([mx.zeros(3), -mx.array(m.opt.gravity)])
        cacc_list[body_id] = cacc0
        cfrc_int_list[body_id] = mx.zeros(6)
        continue

      cacc_parent = cacc_list[m.body_parentid[body_id]]
      dof_adr = m.body_dofadr[body_id]
      dof_num = m.body_dofnum[body_id]

      cacc_vel = mx.zeros(6)
      cacc_acc = mx.zeros(6)
      for di in range(dof_adr, dof_adr + dof_num):
        cacc_vel = cacc_vel + d.cdof_dot[di] * d.qvel[di]
        cacc_acc = cacc_acc + d.cdof[di] * d.qacc[di]
      cacc = cacc_parent + cacc_vel + cacc_acc

      cinert_i = d._impl.cinert[body_id]
      cvel_i = d.cvel[body_id]
      cfrc_body = mjx_math.inert_mul(cinert_i, cacc)
      cfrc_corr = mjx_math.inert_mul(cinert_i, cvel_i)
      cfrc_body = cfrc_body + mjx_math.motion_cross_force(cvel_i, cfrc_corr)
      cfrc_int_i = cfrc_body - cfrc_ext[body_id]

      cacc_list[body_id] = cacc
      cfrc_int_list[body_id] = cfrc_int_i

  # Backward pass: accumulate cfrc_int
  cfrc_int_np = np.array(mx.stack(cfrc_int_list))
  for depth in range(max_depth, 0, -1):
    for b in range(m.nbody):
      if depths[b] == depth:
        parent = m.body_parentid[b]
        cfrc_int_np[parent] += cfrc_int_np[b]

  cacc = mx.stack(cacc_list)
  cfrc_int = mx.array(cfrc_int_np)

  d = _tree_replace(d, {
      '_impl.cacc': cacc,
      '_impl.cfrc_int': cfrc_int,
      '_impl.cfrc_ext': cfrc_ext,
  })
  return d


# ---------------------------------------------------------------------------
# tendon
# ---------------------------------------------------------------------------


def tendon(m: Model, d: Data) -> Data:
  """Computes tendon lengths and moments."""

  if not m.ntendon:
    return d

  # Joint tendons
  (wrap_id_jnt,) = np.nonzero(m.wrap_type == WrapType.JOINT)
  (tendon_id_jnt,) = np.nonzero(np.isin(m.tendon_adr, wrap_id_jnt))

  ntendon_jnt = tendon_id_jnt.size
  wrap_objid_jnt = m.wrap_objid[wrap_id_jnt]
  tendon_num_jnt = m.tendon_num[tendon_id_jnt]

  moment_jnt_np = np.array(m.wrap_prm[wrap_id_jnt])
  qpos_np = np.array(d.qpos)

  length_jnt_np = np.zeros(ntendon_jnt)
  offset = 0
  for t in range(ntendon_jnt):
    for w in range(tendon_num_jnt[t]):
      objid = wrap_objid_jnt[offset + w]
      length_jnt_np[t] += (
          moment_jnt_np[offset + w] * qpos_np[m.jnt_qposadr[objid]]
      )
    offset += tendon_num_jnt[t]
  length_jnt = mx.array(length_jnt_np)

  adr_moment_jnt = np.repeat(tendon_id_jnt, tendon_num_jnt)
  dofadr_moment_jnt = m.jnt_dofadr[wrap_objid_jnt]

  # Pulleys
  (wrap_id_pulley,) = np.nonzero(m.wrap_type == WrapType.PULLEY)
  divisor = np.ones(m.nwrap)
  for adr, num in zip(m.tendon_adr, m.tendon_num):
    for id_pulley in wrap_id_pulley:
      if adr <= id_pulley < adr + num:
        divisor[id_pulley : adr + num] = np.maximum(
            mujoco.mjMINVAL, m.wrap_prm[id_pulley]
        )

  # Spatial sites
  (wrap_id_site,) = np.nonzero(m.wrap_type == WrapType.SITE)
  (pair_id,) = np.nonzero(np.diff(wrap_id_site) == 1)
  wrap_id_site_pair = np.setdiff1d(
      wrap_id_site[pair_id], m.tendon_adr[1:] - 1
  )
  wrap_objid_site0 = m.wrap_objid[wrap_id_site_pair]
  wrap_objid_site1 = m.wrap_objid[wrap_id_site_pair + 1]

  lengths_site_list = []
  moments_site_list = []
  for k in range(wrap_id_site_pair.size):
    pnt0 = d.site_xpos[wrap_objid_site0[k]]
    pnt1 = d.site_xpos[wrap_objid_site1[k]]
    body0 = m.site_bodyid[wrap_objid_site0[k]]
    body1 = m.site_bodyid[wrap_objid_site1[k]]

    dif = pnt1 - pnt0
    length = mjx_math.norm(dif)
    vec = mx.where(
        length < mujoco.mjMINVAL,
        mx.array([1.0, 0.0, 0.0]),
        mjx_math.safe_div(dif, length),
    )

    jacp1, _ = support.jac(m, d, pnt0, body0)
    jacp2, _ = support.jac(m, d, pnt1, body1)
    jacdif = jacp2 - jacp1
    moment = mx.where(body0 != body1, jacdif @ vec, mx.zeros((m.nv,)))

    lengths_site_list.append(length)
    moments_site_list.append(moment)

  if lengths_site_list:
    lengths_site = mx.stack(lengths_site_list)
    moments_site = mx.stack(moments_site_list)
  else:
    lengths_site = mx.zeros((0,))
    moments_site = mx.zeros((0, m.nv))

  if wrap_id_site_pair.size:
    divisor_site_pair = divisor[wrap_id_site_pair]
    lengths_site = lengths_site / mx.array(divisor_site_pair)
    moments_site = moments_site / mx.array(divisor_site_pair)[:, None]

  tendon_nsite = np.array([
      sum((wrap_id_site_pair >= adr) & (wrap_id_site_pair < adr + num))
      for adr, num in zip(m.tendon_adr, m.tendon_num)
  ])
  tendon_has_site = tendon_nsite > 0
  (tendon_id_site,) = np.nonzero(tendon_has_site)
  tendon_nsite_filtered = tendon_nsite[tendon_has_site]
  tendon_with_site = tendon_nsite_filtered.size

  if tendon_with_site > 0 and lengths_site.shape[0] > 0:
    ten_site_id = np.repeat(
        np.arange(tendon_with_site), tendon_nsite_filtered
    )
    length_site_np = np.zeros(tendon_with_site)
    moment_site_np = np.zeros((tendon_with_site, m.nv))
    for k in range(len(ten_site_id)):
      tid = ten_site_id[k]
      length_site_np[tid] += float(lengths_site[k])
      moment_site_np[tid] += np.array(moments_site[k])
    length_site = mx.array(length_site_np)
    moment_site = mx.array(moment_site_np)
  else:
    length_site = mx.zeros((0,))
    moment_site = mx.zeros((0, m.nv))

  # TODO: geom wrap (sphere/cylinder) -- deferred for initial port

  # Assemble
  ten_length_np = np.zeros(m.ntendon)
  for k in range(ntendon_jnt):
    ten_length_np[tendon_id_jnt[k]] = float(length_jnt[k])
  for k in range(tendon_with_site):
    ten_length_np[tendon_id_site[k]] += float(length_site[k])

  ten_moment_np = np.zeros((m.ntendon, m.nv))
  for k in range(len(adr_moment_jnt)):
    ten_moment_np[adr_moment_jnt[k], dofadr_moment_jnt[k]] = float(
        moment_jnt_np[k]
    )
  for k in range(tendon_with_site):
    ten_moment_np[tendon_id_site[k]] += np.array(moment_site[k])

  d = _replace(d, ten_length=mx.array(ten_length_np))
  d = _tree_replace(d, {'_impl.ten_J': mx.array(ten_moment_np)})
  return d


# ---------------------------------------------------------------------------
# _site_dof_mask
# ---------------------------------------------------------------------------


def _site_dof_mask(m: Model) -> np.ndarray:
  """Creates a dof mask for site transmissions."""
  mask = np.ones((m.nu, m.nv))
  for i in np.nonzero(m.actuator_trnid[:, 1] != -1)[0]:
    id_, refid = m.actuator_trnid[i]
    b0 = m.body_weldid[m.site_bodyid[id_]]
    b1 = m.body_weldid[m.site_bodyid[refid]]
    dofadr0 = m.body_dofadr[b0] + m.body_dofnum[b0] - 1
    dofadr1 = m.body_dofadr[b1] + m.body_dofnum[b1] - 1

    while dofadr0 != dofadr1:
      if dofadr0 < dofadr1:
        dofadr1 = m.dof_parentid[dofadr1]
      else:
        dofadr0 = m.dof_parentid[dofadr0]
      if dofadr0 == -1 or dofadr1 == -1:
        break

    da = dofadr0 if dofadr0 == dofadr1 else -1
    while da >= 0:
      mask[i, da] = 0.0
      da = m.dof_parentid[da]

  return mask


# ---------------------------------------------------------------------------
# transmission
# ---------------------------------------------------------------------------


def transmission(m: Model, d: Data) -> Data:
  """Computes actuator/transmission lengths and moments."""

  if not m.nu:
    return d

  has_refsite = m.actuator_trnid[:, 1] != -1
  site_dof_mask = _site_dof_mask(m)

  site_quat_list = []
  for i in range(m.nsite):
    site_quat_list.append(
        mjx_math.quat_mul(
            mx.array(m.site_quat[i]), d.xquat[m.site_bodyid[i]]
        )
    )
  site_quat = (
      mx.stack(site_quat_list) if site_quat_list else mx.zeros((0, 4))
  )

  length_list = []
  moment_list = []

  for u in range(m.nu):
    trntype = m.actuator_trntype[u]
    trnid = m.actuator_trnid[u]
    gear = mx.array(m.actuator_gear[u])

    if trntype in (TrnType.JOINT, TrnType.JOINTINPARENT):
      j = trnid[0]
      jnt_typ = m.jnt_type[j]
      m_j = m.jnt_dofadr[j]

      if jnt_typ == JointType.FREE:
        length = mx.zeros(1)
        moment = gear.copy()
        if trntype == TrnType.JOINTINPARENT:
          qpos_start = m.jnt_qposadr[j]
          quat_neg = mjx_math.quat_inv(
              d.qpos[qpos_start + 3 : qpos_start + 7]
          )
          gearaxis = mjx_math.rotate(gear[3:], quat_neg)
          moment_np = np.array(moment)
          moment_np[3:] = np.array(gearaxis)
          moment = mx.array(moment_np)
        moment_full = np.zeros(m.nv)
        for k in range(6):
          moment_full[m_j + k] = float(moment[k])
        moment = mx.array(moment_full)
      elif jnt_typ == JointType.BALL:
        qpos_start = m.jnt_qposadr[j]
        q = d.qpos[qpos_start : qpos_start + 4]
        axis, angle = mjx_math.quat_to_axis_angle(q)
        gearaxis = gear[:3]
        if trntype == TrnType.JOINTINPARENT:
          quat_neg = mjx_math.quat_inv(q)
          gearaxis = mjx_math.rotate(gear[:3], quat_neg)
        length = mx.sum(axis * angle * gearaxis)[None]
        moment_full = np.zeros(m.nv)
        for k in range(3):
          moment_full[m_j + k] = float(gearaxis[k])
        moment = mx.array(moment_full)
      elif jnt_typ in (JointType.SLIDE, JointType.HINGE):
        qpos_j = d.qpos[m.jnt_qposadr[j]]
        length = qpos_j * gear[0]
        if not isinstance(length, mx.array) or length.ndim == 0:
          length = mx.array([float(length)])
        moment_full = np.zeros(m.nv)
        moment_full[m_j] = float(gear[0])
        moment = mx.array(moment_full)
      else:
        raise RuntimeError(
            f'unrecognized joint type: {JointType(jnt_typ)}'
        )

    elif trntype == TrnType.SITE:
      length = mx.zeros(1)
      id_ = trnid[0]
      refid = trnid[1]
      body_id_ = m.site_bodyid[id_]
      jacp, jacr = support.jac(m, d, d.site_xpos[id_], body_id_)
      frame_xmat = d.site_xmat[id_]

      if has_refsite[u]:
        body_refid = m.site_bodyid[refid]
        vecp = d.site_xmat[refid].T @ (
            d.site_xpos[id_] - d.site_xpos[refid]
        )
        vecr = mjx_math.quat_sub(site_quat[id_], site_quat[refid])
        length = length + mx.sum(mx.concatenate([vecp, vecr]) * gear)
        jacrefp, jacrefr = support.jac(
            m, d, d.site_xpos[refid], body_refid
        )
        jacp = jacp - jacrefp
        jacr = jacr - jacrefr
        frame_xmat = d.site_xmat[refid]

      jac_full = (
          mx.concatenate([jacp, jacr], axis=1)
          * mx.array(site_dof_mask[u])[:, None]
      )
      wrench = mx.concatenate([
          frame_xmat @ gear[:3], frame_xmat @ gear[3:]
      ])
      moment = jac_full @ wrench
      length = length[None] if length.ndim == 0 else length

    elif trntype == TrnType.TENDON:
      length = d.ten_length[trnid[0]] * gear[:1]
      moment = d._impl.ten_J[trnid[0]] * gear[0]
    else:
      raise RuntimeError(f'unrecognized trntype: {TrnType(trntype)}')

    length_list.append(length)
    moment_list.append(moment)

  length = mx.concatenate(length_list).reshape((m.nu,))
  moment = mx.stack(moment_list).reshape((m.nu, m.nv))

  d = _replace(d, actuator_length=length)
  d = _tree_replace(d, {'_impl.actuator_moment': moment})
  return d


# ---------------------------------------------------------------------------
# tendon_armature
# ---------------------------------------------------------------------------


def tendon_armature(m: Model, d: Data) -> Data:
  """Add tendon armature to qM."""

  if not m.ntendon:
    return d

  ten_J_np = np.array(d._impl.ten_J)
  armature_np = np.array(m.tendon_armature)
  JTAJ = ten_J_np.T @ (ten_J_np * armature_np[:, None])

  if support.is_sparse(m):
    ij = []
    for i in range(m.nv):
      j = i
      while j > -1:
        ij.append((i, j))
        j = m.dof_parentid[j]
    JTAJ_sparse = np.array([JTAJ[i, j] for i, j in ij])
    qM_new = mx.array(np.array(d._impl.qM) + JTAJ_sparse)
  else:
    qM_new = d._impl.qM + mx.array(JTAJ)

  d = _tree_replace(d, {'_impl.qM': qM_new})
  return d


# ---------------------------------------------------------------------------
# tendon_dot
# ---------------------------------------------------------------------------


def tendon_dot(m: Model, d: Data) -> mx.array:
  """Compute time derivative of dense tendon Jacobian."""

  ten_Jdot = np.zeros((m.ntendon, m.nv))

  if not m.ntendon:
    return mx.array(ten_Jdot)

  (wrap_id_pulley,) = np.nonzero(m.wrap_type == WrapType.PULLEY)
  divisor = np.ones(m.nwrap)
  for adr, num in zip(m.tendon_adr, m.tendon_num):
    for id_pulley in wrap_id_pulley:
      if adr <= id_pulley < adr + num:
        divisor[id_pulley : adr + num] = np.maximum(
            mujoco.mjMINVAL, m.wrap_prm[id_pulley]
        )

  (wrap_id_site,) = np.nonzero(m.wrap_type == WrapType.SITE)
  (pair_id,) = np.nonzero(np.diff(wrap_id_site) == 1)
  wrap_id_site_pair = np.setdiff1d(
      wrap_id_site[pair_id], m.tendon_adr[1:] - 1
  )
  wrap_objid_site0 = m.wrap_objid[wrap_id_site_pair]
  wrap_objid_site1 = m.wrap_objid[wrap_id_site_pair + 1]
  site_bodyid0 = m.site_bodyid[wrap_objid_site0]
  site_bodyid1 = m.site_bodyid[wrap_objid_site1]

  momentdots_list = []
  for k in range(wrap_id_site_pair.size):
    wpnt0 = d.site_xpos[wrap_objid_site0[k]]
    wpnt1 = d.site_xpos[wrap_objid_site1[k]]
    body0 = site_bodyid0[k]
    body1 = site_bodyid1[k]

    subtree_com0 = d.subtree_com[m.body_rootid[body0]]
    subtree_com1 = d.subtree_com[m.body_rootid[body1]]
    cvel0 = d.cvel[body0]
    cvel1 = d.cvel[body1]
    wvel0 = cvel0[3:] - mjx_math._cross(wpnt0 - subtree_com0, cvel0[:3])
    wvel1 = cvel1[3:] - mjx_math._cross(wpnt1 - subtree_com1, cvel1[:3])

    dpnt = wpnt1 - wpnt0
    norm_val = mjx_math.norm(dpnt)
    dpnt_n = mx.where(
        norm_val < mujoco.mjMINVAL,
        mx.array([1.0, 0.0, 0.0]),
        mjx_math.safe_div(dpnt, norm_val),
    )

    dvel = wvel1 - wvel0
    dot_val = mx.sum(dpnt_n * dvel)
    dvel = dvel + dpnt_n * (-dot_val)
    dvel = mx.where(
        norm_val > mujoco.mjMINVAL,
        mjx_math.safe_div(dvel, norm_val),
        mx.array(0.0),
    )

    jacp1_dot, _ = support.jac_dot(m, d, wpnt0, body0)
    jacp2_dot, _ = support.jac_dot(m, d, wpnt1, body1)
    jacdif_dot = jacp2_dot - jacp1_dot
    tmp0 = jacdif_dot @ dpnt_n

    jacp1, _ = support.jac(m, d, wpnt0, body0)
    jacp2, _ = support.jac(m, d, wpnt1, body1)
    jacdif = jacp2 - jacp1
    tmp1 = jacdif @ dvel

    md = mx.where(body0 != body1, tmp0 + tmp1, mx.zeros((m.nv,)))
    momentdots_list.append(md)

  if momentdots_list:
    momentdots = mx.stack(momentdots_list)
  else:
    momentdots = mx.zeros((0, m.nv))

  if wrap_id_site_pair.size:
    divisor_site_pair = divisor[wrap_id_site_pair]
    momentdots = momentdots / mx.array(divisor_site_pair)[:, None]

  tendon_nsite = np.array([
      sum((wrap_id_site_pair >= adr) & (wrap_id_site_pair < adr + num))
      for adr, num in zip(m.tendon_adr, m.tendon_num)
  ])
  tendon_has_site = tendon_nsite > 0
  (tendon_id_site,) = np.nonzero(tendon_has_site)
  tendon_nsite_filtered = tendon_nsite[tendon_has_site]
  tendon_with_site = tendon_nsite_filtered.size

  if tendon_with_site > 0 and momentdots.shape[0] > 0:
    ten_site_id = np.repeat(
        np.arange(tendon_with_site), tendon_nsite_filtered
    )
    momentdot_np = np.zeros((tendon_with_site, m.nv))
    for k in range(len(ten_site_id)):
      momentdot_np[ten_site_id[k]] += np.array(momentdots[k])
    for k in range(tendon_with_site):
      ten_Jdot[tendon_id_site[k]] = momentdot_np[k]

  return mx.array(ten_Jdot)


# ---------------------------------------------------------------------------
# tendon_bias
# ---------------------------------------------------------------------------


def tendon_bias(m: Model, d: Data) -> Data:
  """Add bias force due to tendon armature."""

  if not m.ntendon:
    return d

  ten_Jdot = tendon_dot(m, d)
  coef = mx.array(m.tendon_armature) * (ten_Jdot @ d.qvel)

  ten_J = d._impl.ten_J
  bias_add = mx.zeros((m.nv,))
  for i in range(m.ntendon):
    bias_add = bias_add + ten_J[i] * coef[i]

  d = _replace(d, qfrc_bias=d.qfrc_bias + bias_add)
  return d
