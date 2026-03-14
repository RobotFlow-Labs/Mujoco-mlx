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
"""Core non-smooth constraint functions (MLX port)."""

from typing import Optional, Tuple, Union

import mlx.core as mx
import mujoco
from mujoco.mjx_mlx._src import collision_driver
from mujoco.mjx_mlx._src import math
from mujoco.mjx_mlx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.dataclasses import PyTreeNode
from mujoco.mjx_mlx._src.dataclasses import tree_map as _tree_map
from mujoco.mjx_mlx._src.types import ConeType
from mujoco.mjx_mlx._src.types import ConstraintType
from mujoco.mjx_mlx._src.types import Contact
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DataMLX
from mujoco.mjx_mlx._src.types import DisableBit
from mujoco.mjx_mlx._src.types import EqType
from mujoco.mjx_mlx._src.types import JointType
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import ModelMLX
from mujoco.mjx_mlx._src.types import ObjType
from mujoco.mjx_mlx._src.types import OptionMLX
# pylint: enable=g-importing-member
import numpy as np

import dataclasses


# ---------------------------------------------------------------------------
# Helper: tree_map over _Efc (a PyTreeNode dataclass) by applying fn to every
# mx.array leaf.  For plain tuples/lists of _Efc we iterate manually.
# ---------------------------------------------------------------------------

def _efc_tree_map(fn, *efcs):
  """tree_map that works on _Efc PyTreeNode instances."""
  if len(efcs) == 1:
    return _tree_map(fn, efcs[0])
  return _tree_map(fn, efcs[0], *efcs[1:])


class _Efc(PyTreeNode):
  """Support data for creating constraint matrices."""

  J: mx.array
  pos_aref: mx.array
  pos_imp: mx.array
  invweight: mx.array
  solref: mx.array
  solimp: mx.array
  margin: mx.array
  frictionloss: mx.array


def _kbi(
    m: Model,
    solref: mx.array,
    solimp: mx.array,
    pos: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
  """Calculates stiffness, damping, and impedance of a constraint."""
  timeconst, dampratio = solref[0], solref[1]

  if not m.opt.disableflags & DisableBit.REFSAFE:
    timeconst = mx.maximum(timeconst, 2 * m.opt.timestep)

  dmin, dmax, width, mid, power = solimp[0], solimp[1], solimp[2], solimp[3], solimp[4]

  dmin = mx.clip(dmin, mujoco.mjMINIMP, mujoco.mjMAXIMP)
  dmax = mx.clip(dmax, mujoco.mjMINIMP, mujoco.mjMAXIMP)
  width = mx.maximum(mx.array(mujoco.mjMINVAL), width)
  mid = mx.clip(mid, mujoco.mjMINIMP, mujoco.mjMAXIMP)
  power = mx.maximum(mx.array(1.0), power)

  # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  k = 1 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
  b = 2 / (dmax * timeconst)
  k = mx.where(solref[0] <= 0, -solref[0] / (dmax * dmax), k)
  b = mx.where(solref[1] <= 0, -solref[1] / dmax, b)

  imp_x = mx.abs(pos) / width
  imp_a = (1.0 / mx.power(mid, power - 1)) * mx.power(imp_x, power)
  imp_b = 1 - (1.0 / mx.power(1 - mid, power - 1)) * mx.power(1 - imp_x, power)
  imp_y = mx.where(imp_x < mid, imp_a, imp_b)
  imp = dmin + imp_y * (dmax - dmin)
  imp = mx.clip(imp, dmin, dmax)
  imp = mx.where(imp_x > 1.0, dmax, imp)

  return k, b, imp  # corresponds to K, B, I of efc_KBIP


def _row(j: mx.array, *args) -> _Efc:
  """Creates an efc row, ensuring args all have same row count."""
  if len(j.shape) < 2:
    return _Efc(j, *args)  # if j isn't batched, ignore

  args = list(args)
  for i, arg in enumerate(args):
    # Ensure arg is mx.array before tiling
    if not isinstance(arg, mx.array):
      arg = mx.array(arg) if hasattr(arg, '__len__') or hasattr(arg, 'shape') else mx.array([arg])
      args[i] = arg
    if not arg.shape or arg.shape[0] != j.shape[0]:
      args[i] = mx.tile(arg, (j.shape[0],) + (1,) * max(0, len(arg.shape) - 1))
  return _Efc(j, *args)


def _efc_equality_connect(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for connect equality constraints."""

  eq_id = np.nonzero(m.eq_type == EqType.CONNECT)[0]
  if (m.opt.disableflags & DisableBit.EQUALITY) or eq_id.size == 0:
    return None

  # Vectorized over eq_id (replaces jax.vmap)
  def _single_row(is_site, obj1id, obj2id, body1id, body2id, data, solref, solimp, active):
    anchor1, anchor2 = data[0:3], data[3:6]

    pos1 = d.xmat[body1id] @ anchor1 + d.xpos[body1id]
    pos2 = d.xmat[body2id] @ anchor2 + d.xpos[body2id]

    if m.nsite:
      pos1 = mx.where(is_site, d.site_xpos[obj1id], pos1)
      pos2 = mx.where(is_site, d.site_xpos[obj2id], pos2)

    # error is difference in global positions
    pos = pos1 - pos2

    # compute Jacobian difference (opposite of contact: 0 - 1)
    jacp1, _ = support.jac(m, d, pos1, body1id)
    jacp2, _ = support.jac(m, d, pos2, body2id)
    j = (jacp1 - jacp2).T
    pos_imp = math.norm(pos)
    invweight = m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]
    zero = mx.zeros_like(pos)

    efc = _row(j, pos, pos_imp, invweight, solref, solimp, zero, zero)
    return _tree_map(lambda x: x * active, efc)

  is_site = m.eq_objtype == ObjType.SITE
  body1id = np.copy(m.eq_obj1id)
  body2id = np.copy(m.eq_obj2id)

  if m.nsite:
    body1id[is_site] = m.site_bodyid[m.eq_obj1id[is_site]]
    body2id[is_site] = m.site_bodyid[m.eq_obj2id[is_site]]

  # Slice by eq_id
  is_site_s = is_site[eq_id]
  obj1id_s = m.eq_obj1id[eq_id]
  obj2id_s = m.eq_obj2id[eq_id]
  body1id_s = body1id[eq_id]
  body2id_s = body2id[eq_id]
  data_s = m.eq_data[eq_id]
  solref_s = m.eq_solref[eq_id]
  solimp_s = m.eq_solimp[eq_id]
  active_s = d.eq_active[eq_id]

  results = [
      _single_row(
          is_site_s[i], int(obj1id_s[i]), int(obj2id_s[i]),
          int(body1id_s[i]), int(body2id_s[i]),
          data_s[i], solref_s[i], solimp_s[i], active_s[i],
      )
      for i in range(eq_id.size)
  ]
  if not results:
    return None
  # concatenate to drop row grouping
  return _tree_map(lambda *xs: mx.concatenate(xs, axis=0), *results)


def _efc_equality_weld(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for weld equality constraints."""

  eq_id = np.nonzero(m.eq_type == EqType.WELD)[0]
  if (m.opt.disableflags & DisableBit.EQUALITY) or eq_id.size == 0:
    return None

  def _single_row(is_site, obj1id, obj2id, body1id, body2id, data, solref, solimp, active):
    anchor1, anchor2 = data[0:3], data[3:6]
    relpose, torquescale = data[6:10], data[10]

    # error is difference in global position and orientation
    pos1 = d.xmat[body1id] @ anchor2 + d.xpos[body1id]
    pos2 = d.xmat[body2id] @ anchor1 + d.xpos[body2id]

    if m.nsite:
      pos1 = mx.where(is_site, d.site_xpos[obj1id], pos1)
      pos2 = mx.where(is_site, d.site_xpos[obj2id], pos2)

    cpos = pos1 - pos2

    # compute Jacobian difference (opposite of contact: 0 - 1)
    jacp1, jacr1 = support.jac(m, d, pos1, body1id)
    jacp2, jacr2 = support.jac(m, d, pos2, body2id)
    jacdifp = jacp1 - jacp2
    jacdifr = (jacr1 - jacr2) * torquescale

    # compute orientation error: neg(q1) * q0 * relpose (axis components only)
    quat = math.quat_mul(d.xquat[body1id], relpose)
    quat1 = math.quat_inv(d.xquat[body2id])

    if m.nsite:
      quat = mx.where(
          is_site, math.quat_mul(d.xquat[body1id], m.site_quat[obj1id]), quat
      )
      quat1 = mx.where(
          is_site,
          math.quat_inv(math.quat_mul(d.xquat[body2id], m.site_quat[obj2id])),
          quat1,
      )

    crot = math.quat_mul(quat1, quat)[1:]  # copy axis components

    pos = mx.concatenate([cpos, crot * torquescale])

    # correct rotation Jacobian: 0.5 * neg(q1) * (jac0-jac1) * q0 * relpose
    jac_fn = lambda j_row: math.quat_mul(math.quat_mul_axis(quat1, j_row), quat)[1:]
    # vmap over rows of jacdifr -> explicit loop
    jacdifr_corrected = mx.stack([jac_fn(jacdifr[k]) for k in range(jacdifr.shape[0])])
    jacdifr_corrected = 0.5 * jacdifr_corrected
    j = mx.concatenate([jacdifp.T, jacdifr_corrected.T])
    pos_imp = math.norm(pos)
    invweight = m.body_invweight0[body1id] + m.body_invweight0[body2id]
    invweight = mx.repeat(invweight, 3, axis=0)
    zero = mx.zeros_like(pos)

    efc = _row(j, pos, pos_imp, invweight, solref, solimp, zero, zero)
    return _tree_map(lambda x: x * active, efc)

  is_site = m.eq_objtype == ObjType.SITE
  body1id = np.copy(m.eq_obj1id)
  body2id = np.copy(m.eq_obj2id)

  if m.nsite:
    body1id[is_site] = m.site_bodyid[m.eq_obj1id[is_site]]
    body2id[is_site] = m.site_bodyid[m.eq_obj2id[is_site]]

  is_site_s = is_site[eq_id]
  obj1id_s = m.eq_obj1id[eq_id]
  obj2id_s = m.eq_obj2id[eq_id]
  body1id_s = body1id[eq_id]
  body2id_s = body2id[eq_id]
  data_s = m.eq_data[eq_id]
  solref_s = m.eq_solref[eq_id]
  solimp_s = m.eq_solimp[eq_id]
  active_s = d.eq_active[eq_id]

  results = [
      _single_row(
          is_site_s[i], int(obj1id_s[i]), int(obj2id_s[i]),
          int(body1id_s[i]), int(body2id_s[i]),
          data_s[i], solref_s[i], solimp_s[i], active_s[i],
      )
      for i in range(eq_id.size)
  ]
  if not results:
    return None
  # concatenate to drop row grouping
  return _tree_map(lambda *xs: mx.concatenate(xs, axis=0), *results)


def _efc_equality_joint(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for joint equality constraints."""

  eq_id = np.nonzero(m.eq_type == EqType.JOINT)[0]

  if (m.opt.disableflags & DisableBit.EQUALITY) or eq_id.size == 0:
    return None

  def _single_row(obj2id, data, solref, solimp, active, dofadr1, dofadr2, qposadr1, qposadr2):
    pos1, pos2 = d.qpos[qposadr1], d.qpos[qposadr2]
    ref1, ref2 = m.qpos0[qposadr1], m.qpos0[qposadr2]
    dif = (pos2 - ref2) * (obj2id > -1)
    dif_power = mx.power(dif, mx.arange(0, 5, dtype=mx.float32))
    pos = pos1 - ref1 - mx.sum(data[:5] * dif_power)
    deriv = mx.sum(data[1:5] * dif_power[:4] * mx.arange(1, 5, dtype=mx.float32)) * (obj2id > -1)

    j = mx.zeros((m.nv,))
    # .at[idx].set -> construct via scatter
    j = _set_at(j, dofadr2, -deriv)
    j = _set_at(j, dofadr1, mx.array(1.0))
    invweight = m.dof_invweight0[dofadr1]
    invweight = invweight + m.dof_invweight0[dofadr2] * (obj2id > -1)
    zero = mx.zeros_like(pos)

    efc = _row(j, pos, pos, invweight, solref, solimp, zero, zero)
    return _tree_map(lambda x: x * active, efc)

  # Slice args
  eq_obj1id_s = m.eq_obj1id[eq_id]
  eq_obj2id_s = m.eq_obj2id[eq_id]
  eq_data_s = m.eq_data[eq_id]
  eq_solref_s = m.eq_solref[eq_id]
  eq_solimp_s = m.eq_solimp[eq_id]
  eq_active_s = d.eq_active[eq_id]

  dofadr1 = int(m.jnt_dofadr[eq_obj1id_s])
  dofadr2 = int(m.jnt_dofadr[eq_obj2id_s])
  qposadr1 = int(m.jnt_qposadr[eq_obj1id_s])
  qposadr2 = int(m.jnt_qposadr[eq_obj2id_s])

  results = [
      _single_row(
          int(eq_obj2id_s[i]), eq_data_s[i], eq_solref_s[i], eq_solimp_s[i],
          eq_active_s[i], int(dofadr1[i]), int(dofadr2[i]),
          int(qposadr1[i]), int(qposadr2[i]),
      )
      for i in range(eq_id.size)
  ]
  if not results:
    return None
  # Stack without concatenation (each row is scalar-ish)
  return _tree_map(lambda *xs: mx.stack(xs, axis=0), *results)


def _efc_equality_tendon(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for tendon equality constraints."""
  if False:  # MLX port: single backend, no impl check
    pass  # MLX port: backend check removed

  eq_id = np.nonzero(m.eq_type == EqType.TENDON)[0]

  if (m.opt.disableflags & DisableBit.EQUALITY) or eq_id.size == 0:
    return None

  obj1id = m.eq_obj1id[eq_id]
  obj2id = m.eq_obj2id[eq_id]
  data_s = m.eq_data[eq_id]
  solref_s = m.eq_solref[eq_id]
  solimp_s = m.eq_solimp[eq_id]
  active_s = d.eq_active[eq_id]

  def _single_row(obj2id_i, data_i, solref_i, solimp_i, invweight_i, jac1_i, jac2_i, pos1_i, pos2_i, active_i):
    dif = pos2_i * (obj2id_i > -1)
    dif_power = mx.power(dif, mx.arange(0, 5, dtype=mx.float32))
    pos = pos1_i - mx.sum(data_i[:5] * dif_power)
    deriv = mx.sum(data_i[1:5] * dif_power[:4] * mx.arange(1, 5, dtype=mx.float32)) * (obj2id_i > -1)
    j = jac1_i + jac2_i * -deriv
    zero = mx.zeros_like(pos)

    efc = _row(j, pos, pos, invweight_i, solref_i, solimp_i, zero, zero)
    return _tree_map(lambda x: x * active_i, efc)

  inv1, inv2 = m.tendon_invweight0[obj1id], m.tendon_invweight0[obj2id]
  jac1, jac2 = (d._impl or d).ten_J[obj1id], (d._impl or d).ten_J[obj2id]
  pos1 = d.ten_length[obj1id] - m.tendon_length0[obj1id]
  pos2 = d.ten_length[obj2id] - m.tendon_length0[obj2id]
  invweight = inv1 + inv2 * (obj2id > -1)

  results = [
      _single_row(
          int(obj2id[i]), data_s[i], solref_s[i], solimp_s[i],
          invweight[i], jac1[i], jac2[i], pos1[i], pos2[i], active_s[i],
      )
      for i in range(eq_id.size)
  ]
  if not results:
    return None
  return _tree_map(lambda *xs: mx.stack(xs, axis=0), *results)


def _efc_friction(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for dof frictionloss."""
  if False:  # MLX port: single backend, no impl check
    pass  # MLX port: backend check removed

  dof_id = np.nonzero((np.array(m.dof_frictionloss) > 0))[0]
  tendon_id = np.nonzero((np.array(m.tendon_frictionloss) > 0))[0]

  size = dof_id.size + tendon_id.size
  if (m.opt.disableflags & DisableBit.FRICTIONLOSS) or (size == 0):
    return None

  # Build dof args
  eye_nv = mx.eye(m.nv)
  j_dof = eye_nv[dof_id]
  fl_dof = m.dof_frictionloss[dof_id]
  iw_dof = m.dof_invweight0[dof_id]
  sr_dof = m.dof_solref[dof_id]
  si_dof = m.dof_solimp[dof_id]

  # Build tendon args
  j_ten = (d._impl or d).ten_J[tendon_id]
  fl_ten = m.tendon_frictionloss[tendon_id]
  iw_ten = m.tendon_invweight0[tendon_id]
  sr_ten = m.tendon_solref_fri[tendon_id]
  si_ten = m.tendon_solimp_fri[tendon_id]

  # Concatenate dof and tendon args
  if dof_id.size > 0 and tendon_id.size > 0:
    j_all = mx.concatenate([j_dof, j_ten], axis=0)
    fl_all = mx.concatenate([fl_dof, fl_ten], axis=0)
    iw_all = mx.concatenate([iw_dof, iw_ten], axis=0)
    sr_all = mx.concatenate([sr_dof, sr_ten], axis=0)
    si_all = mx.concatenate([si_dof, si_ten], axis=0)
  elif dof_id.size > 0:
    j_all, fl_all, iw_all, sr_all, si_all = j_dof, fl_dof, iw_dof, sr_dof, si_dof
  else:
    j_all, fl_all, iw_all, sr_all, si_all = j_ten, fl_ten, iw_ten, sr_ten, si_ten

  def _single_row(j, frictionloss, invweight, solref, solimp):
    z = mx.zeros_like(frictionloss)
    return _row(j, z, z, invweight, solref, solimp, z, frictionloss)

  results = [
      _single_row(j_all[i], fl_all[i], iw_all[i], sr_all[i], si_all[i])
      for i in range(size)
  ]
  if not results:
    return None
  return _tree_map(lambda *xs: mx.stack(xs, axis=0), *results)


def _efc_limit_ball(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for ball joint limits."""

  jnt_id = np.nonzero((m.jnt_type == JointType.BALL) & m.jnt_limited)[0]

  if (m.opt.disableflags & DisableBit.LIMIT) or jnt_id.size == 0:
    return None

  def _single_row(qposadr, dofadr, jnt_range, jnt_margin, solref, solimp):
    axis, angle = math.quat_to_axis_angle(d.qpos[mx.arange(4) + qposadr])
    # ball rotation angle is always positive
    axis, angle = math.normalize_with_norm(axis * angle)
    pos = mx.max(jnt_range) - angle - jnt_margin
    active = pos < 0
    j = mx.zeros((m.nv,))
    j = _set_at_slice(j, dofadr, dofadr + 3, -axis)
    invweight = m.dof_invweight0[dofadr]
    z = mx.zeros_like(pos)

    return _row(
        j * active, pos * active, pos, invweight, solref, solimp, jnt_margin, z
    )

  qposadr_s = int(m.jnt_qposadr[jnt_id])
  dofadr_s = int(m.jnt_dofadr[jnt_id])
  range_s = m.jnt_range[jnt_id]
  margin_s = m.jnt_margin[jnt_id]
  solref_s = m.jnt_solref[jnt_id]
  solimp_s = m.jnt_solimp[jnt_id]

  results = [
      _single_row(
          int(qposadr_s[i]), int(dofadr_s[i]),
          range_s[i], margin_s[i], solref_s[i], solimp_s[i],
      )
      for i in range(jnt_id.size)
  ]
  if not results:
    return None
  return _tree_map(lambda *xs: mx.stack(xs, axis=0), *results)


def _efc_limit_slide_hinge(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for slide and hinge joint limits."""

  slide_hinge = np.isin(m.jnt_type, (JointType.SLIDE, JointType.HINGE))
  jnt_id = np.nonzero(slide_hinge & m.jnt_limited)[0]

  if (m.opt.disableflags & DisableBit.LIMIT) or jnt_id.size == 0:
    return None

  def _single_row(qposadr, dofadr, jnt_range, jnt_margin, solref, solimp):
    qpos = d.qpos[qposadr]
    dist_min = qpos - jnt_range[0]
    dist_max = jnt_range[1] - qpos
    pos = mx.minimum(dist_min, dist_max) - jnt_margin
    active = pos < 0
    j = mx.zeros((m.nv,))
    sign_val = mx.where(dist_min < dist_max, mx.array(1.0), mx.array(-1.0))
    j = _set_at(j, dofadr, sign_val)
    invweight = m.dof_invweight0[dofadr]
    z = mx.zeros_like(pos)

    return _row(
        j * active, pos * active, pos, invweight, solref, solimp, jnt_margin, z
    )

  qposadr_s = int(m.jnt_qposadr[jnt_id])
  dofadr_s = int(m.jnt_dofadr[jnt_id])
  range_s = m.jnt_range[jnt_id]
  margin_s = m.jnt_margin[jnt_id]
  solref_s = m.jnt_solref[jnt_id]
  solimp_s = m.jnt_solimp[jnt_id]

  results = [
      _single_row(
          int(qposadr_s[i]), int(dofadr_s[i]),
          range_s[i], margin_s[i], solref_s[i], solimp_s[i],
      )
      for i in range(jnt_id.size)
  ]
  if not results:
    return None
  return _tree_map(lambda *xs: mx.stack(xs, axis=0), *results)


def _efc_limit_tendon(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for tendon limits."""
  if False:  # MLX port: single backend, no impl check
    pass  # MLX port: backend check removed

  tendon_id = np.nonzero(m.tendon_limited)[0]

  if (m.opt.disableflags & DisableBit.LIMIT) or tendon_id.size == 0:
    return None

  length = d.ten_length[tendon_id]
  j = (d._impl or d).ten_J[tendon_id]
  range_ = m.tendon_range[tendon_id]
  margin = m.tendon_margin[tendon_id]
  invweight = m.tendon_invweight0[tendon_id]
  solref = m.tendon_solref_lim[tendon_id]
  solimp = m.tendon_solimp_lim[tendon_id]

  dist_min = length - range_[:, 0]
  dist_max = range_[:, 1] - length
  pos = mx.minimum(dist_min, dist_max) - margin
  active = pos < 0
  sign_val = (dist_min < dist_max).astype(mx.float32) * 2 - 1
  # Multiply j rows by sign * active
  j = j * (sign_val * active)[:, None]
  zero = mx.zeros_like(pos)

  # vmap _row -> explicit loop
  results = [
      _row(j[i], pos[i] * active[i], pos[i], invweight[i], solref[i], solimp[i], margin[i], zero[i])
      for i in range(tendon_id.size)
  ]
  if not results:
    return None
  return _tree_map(lambda *xs: mx.stack(xs, axis=0), *results)


def _efc_contact_frictionless(m: Model, d: Data) -> Optional[_Efc]:
  """Calculates constraint rows for frictionless contacts."""
  if False:  # MLX port: single backend, no impl check
    pass  # MLX port: backend check removed

  con_id = np.nonzero((d._impl or d).contact.dim == 1)[0]

  if con_id.size == 0:
    return None

  def _single_row(c_dist, c_includemargin, c_geom, c_pos, c_frame, c_solref, c_solimp):
    # Ensure all contact fields are mx.array (they may arrive as numpy)
    c_dist = mx.array(c_dist) if not isinstance(c_dist, mx.array) else c_dist
    c_includemargin = mx.array(c_includemargin) if not isinstance(c_includemargin, mx.array) else c_includemargin
    c_geom = mx.array(c_geom) if not isinstance(c_geom, mx.array) else c_geom
    c_pos = mx.array(c_pos) if not isinstance(c_pos, mx.array) else c_pos
    c_frame = mx.array(c_frame) if not isinstance(c_frame, mx.array) else c_frame
    c_solref = mx.array(c_solref) if not isinstance(c_solref, mx.array) else c_solref
    c_solimp = mx.array(c_solimp) if not isinstance(c_solimp, mx.array) else c_solimp

    pos = c_dist - c_includemargin
    active = pos < 0
    body1 = mx.array(m.geom_bodyid)[int(c_geom[0].item())]
    body2 = mx.array(m.geom_bodyid)[int(c_geom[1].item())]
    jac1p, _ = support.jac(m, d, c_pos, int(body1.item()))
    jac2p, _ = support.jac(m, d, c_pos, int(body2.item()))
    frame_3x3 = mx.reshape(c_frame, (3, 3))
    j = (frame_3x3 @ (jac2p - jac1p).T)[0]
    invweight = m.body_invweight0[int(body1.item()), 0] + m.body_invweight0[int(body2.item()), 0]

    return _row(
        j * active,
        pos * active,
        pos,
        invweight,
        c_solref,
        c_solimp,
        c_includemargin,
        mx.zeros_like(pos),
    )

  contact = (d._impl or d).contact
  results = [
      _single_row(
          contact.dist[int(i)], contact.includemargin[int(i)], contact.geom[int(i)],
          contact.pos[int(i)], contact.frame[int(i)], contact.solref[int(i)], contact.solimp[int(i)],
      )
      for i in con_id
  ]
  if not results:
    return None
  return _tree_map(lambda *xs: mx.stack(xs, axis=0), *results)


def _efc_contact_pyramidal(m: Model, d: Data, condim: int) -> Optional[_Efc]:
  """Calculates constraint rows for frictional pyramidal contacts."""
  if (
      False  # MLX port: single backend
      or not isinstance(m.opt._impl, OptionMLX)
  ):
    pass  # MLX port: backend check removed

  con_id = np.nonzero((d._impl or d).contact.dim == condim)[0]

  if con_id.size == 0:
    return None

  def _single_row(c_dist, c_includemargin, c_geom, c_pos, c_frame, c_friction, c_solref, c_solimp):
    # Ensure all contact fields are mx.array (they may arrive as numpy)
    c_dist = mx.array(c_dist) if not isinstance(c_dist, mx.array) else c_dist
    c_includemargin = mx.array(c_includemargin) if not isinstance(c_includemargin, mx.array) else c_includemargin
    c_geom = mx.array(c_geom) if not isinstance(c_geom, mx.array) else c_geom
    c_pos = mx.array(c_pos) if not isinstance(c_pos, mx.array) else c_pos
    c_frame = mx.array(c_frame) if not isinstance(c_frame, mx.array) else c_frame
    c_friction = mx.array(c_friction) if not isinstance(c_friction, mx.array) else c_friction
    c_solref = mx.array(c_solref) if not isinstance(c_solref, mx.array) else c_solref
    c_solimp = mx.array(c_solimp) if not isinstance(c_solimp, mx.array) else c_solimp

    pos = c_dist - c_includemargin
    active = pos < 0
    body1 = int(np.array(m.geom_bodyid)[int(c_geom[0].item())])
    body2 = int(np.array(m.geom_bodyid)[int(c_geom[1].item())])
    jac1p, jac1r = support.jac(m, d, c_pos, body1)
    jac2p, jac2r = support.jac(m, d, c_pos, body2)
    frame_3x3 = mx.reshape(c_frame, (3, 3))
    diff = frame_3x3 @ (jac2p - jac1p).T
    if condim > 3:
      diff = mx.concatenate([diff, (frame_3x3 @ (jac2r - jac1r).T)], axis=0)
    # a pair of opposing pyramid edges per friction dimension
    fri = mx.repeat(c_friction[: condim - 1], 2, axis=0)
    # Negate odd indices
    signs = mx.ones(fri.shape)
    for idx in range(1, fri.shape[0], 2):
      signs = _set_at(signs, idx, mx.array(-1.0))
    fri = fri * signs
    # repeat condims of jacdiff to match +/- friction directions
    j = diff[0] + mx.repeat(diff[1:condim], 2, axis=0) * fri[:, None]

    # pyramidal has common invweight across all edges
    invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
    invweight = invweight + fri[0] * fri[0] * invweight
    invweight = invweight * 2 * fri[0] * fri[0] / m.opt.impratio

    return _row(
        j * active,
        pos * active,
        pos,
        invweight,
        c_solref,
        c_solimp,
        c_includemargin,
        mx.zeros_like(pos),
    )

  contact = (d._impl or d).contact
  results = [
      _single_row(
          contact.dist[int(i)], contact.includemargin[int(i)], contact.geom[int(i)],
          contact.pos[int(i)], contact.frame[int(i)], contact.friction[int(i)],
          contact.solref[int(i)], contact.solimp[int(i)],
      )
      for i in con_id
  ]
  if not results:
    return None
  # concatenate to drop row grouping
  return _tree_map(lambda *xs: mx.concatenate(xs, axis=0), *results)


def _efc_contact_elliptic(m: Model, d: Data, condim: int) -> Optional[_Efc]:
  """Calculates constraint rows for frictional elliptic contacts."""
  if (
      False  # MLX port: single backend
      or not isinstance(m.opt._impl, OptionMLX)
  ):
    pass  # MLX port: backend check removed

  con_id = np.nonzero((d._impl or d).contact.dim == condim)[0]

  if con_id.size == 0:
    return None

  def _single_row(c_dist, c_includemargin, c_geom, c_pos, c_frame, c_friction,
                   c_solref, c_solreffriction, c_solimp):
    # Ensure all contact fields are mx.array (they may arrive as numpy)
    c_dist = mx.array(c_dist) if not isinstance(c_dist, mx.array) else c_dist
    c_includemargin = mx.array(c_includemargin) if not isinstance(c_includemargin, mx.array) else c_includemargin
    c_geom = mx.array(c_geom) if not isinstance(c_geom, mx.array) else c_geom
    c_pos = mx.array(c_pos) if not isinstance(c_pos, mx.array) else c_pos
    c_frame = mx.array(c_frame) if not isinstance(c_frame, mx.array) else c_frame
    c_friction = mx.array(c_friction) if not isinstance(c_friction, mx.array) else c_friction
    c_solref = mx.array(c_solref) if not isinstance(c_solref, mx.array) else c_solref
    c_solreffriction = mx.array(c_solreffriction) if not isinstance(c_solreffriction, mx.array) else c_solreffriction
    c_solimp = mx.array(c_solimp) if not isinstance(c_solimp, mx.array) else c_solimp

    pos = c_dist - c_includemargin
    active = pos < 0
    obj1id = mx.array(m.geom_bodyid)[int(c_geom[0].item())]
    obj2id = mx.array(m.geom_bodyid)[int(c_geom[1].item())]
    jac1p, jac1r = support.jac(m, d, c_pos, int(obj1id.item()))
    jac2p, jac2r = support.jac(m, d, c_pos, int(obj2id.item()))
    frame_3x3 = mx.reshape(c_frame, (3, 3))
    j = frame_3x3 @ (jac2p - jac1p).T
    if condim > 3:
      j = mx.concatenate([j, (frame_3x3 @ (jac2r - jac1r).T)[: condim - 3]])
    invweight = m.body_invweight0[int(obj1id.item()), 0] + m.body_invweight0[int(obj2id.item()), 0]

    # normal row comes from solref, remaining rows from solreffriction
    solreffriction = c_solreffriction + c_solref * (~mx.any(c_solreffriction != 0))
    solreffriction = mx.tile(solreffriction[None, :], (condim - 1, 1))
    solref_full = mx.concatenate([c_solref[None, :], solreffriction])
    fri = mx.square(c_friction[0]) / mx.square(c_friction[1 : condim - 1])
    invweight_arr = mx.array([invweight, invweight / m.opt.impratio])
    invweight_arr = mx.concatenate([invweight_arr, invweight_arr[1] * fri])
    pos_aref = mx.zeros(condim)
    pos_aref = _set_at(pos_aref, 0, pos)

    return _row(
        j * active,
        pos_aref * active,
        pos,
        invweight_arr,
        solref_full,
        c_solimp,
        c_includemargin,
        mx.zeros_like(pos),
    )

  contact = (d._impl or d).contact
  results = [
      _single_row(
          contact.dist[int(i)], contact.includemargin[int(i)], contact.geom[int(i)],
          contact.pos[int(i)], contact.frame[int(i)], contact.friction[int(i)],
          contact.solref[int(i)], contact.solreffriction[int(i)], contact.solimp[int(i)],
      )
      for i in con_id
  ]
  if not results:
    return None
  # concatenate to drop row grouping
  return _tree_map(lambda *xs: mx.concatenate(xs, axis=0), *results)


def counts(efc_type: np.ndarray) -> Tuple[int, int, int, int]:
  """Returns equality, friction, limit, and contact constraint counts."""
  ne = int((efc_type == ConstraintType.EQUALITY).sum())
  nf = int((efc_type == ConstraintType.FRICTION_DOF).sum())
  nf += int((efc_type == ConstraintType.FRICTION_TENDON).sum())
  nl = int((efc_type == ConstraintType.LIMIT_JOINT).sum())
  nl += int((efc_type == ConstraintType.LIMIT_TENDON).sum())
  nc_f = int((efc_type == ConstraintType.CONTACT_FRICTIONLESS).sum())
  nc_p = int((efc_type == ConstraintType.CONTACT_PYRAMIDAL).sum())
  nc_e = int((efc_type == ConstraintType.CONTACT_ELLIPTIC).sum())
  nc = nc_f + nc_p + nc_e

  return ne, nf, nl, nc


def make_efc_type(
    m: Union[Model, mujoco.MjModel], dim: Optional[np.ndarray] = None
) -> np.ndarray:
  """Returns efc_type that outlines the type of each constraint row."""
  if m.opt.disableflags & DisableBit.CONSTRAINT:
    return np.empty(0, dtype=int)

  dim = collision_driver.make_condim(m) if dim is None else dim
  efc_types = []

  if not m.opt.disableflags & DisableBit.EQUALITY:
    num_rows = int((m.eq_type == EqType.CONNECT).sum()) * 3
    num_rows += int((m.eq_type == EqType.WELD).sum()) * 6
    num_rows += int((m.eq_type == EqType.JOINT).sum())
    num_rows += int((m.eq_type == EqType.TENDON).sum())
    efc_types += [ConstraintType.EQUALITY] * num_rows

  if not m.opt.disableflags & DisableBit.FRICTIONLOSS:
    nf_dof = (
        (np.array(m.dof_frictionloss) > 0).sum()
        if isinstance(m, Model) and isinstance(m._impl, ModelMLX)
        else (m.dof_frictionloss > 0).sum()
    )
    efc_types += [ConstraintType.FRICTION_DOF] * int(nf_dof)
    nf_tendon = (
        (np.array(m.tendon_frictionloss) > 0).sum()
        if isinstance(m, Model) and isinstance(m._impl, ModelMLX)
        else (m.tendon_frictionloss > 0).sum()
    )
    efc_types += [ConstraintType.FRICTION_TENDON] * int(nf_tendon)

  if not m.opt.disableflags & DisableBit.LIMIT:
    efc_types += [ConstraintType.LIMIT_JOINT] * int(m.jnt_limited.sum())
    efc_types += [ConstraintType.LIMIT_TENDON] * int(m.tendon_limited.sum())

  if not m.opt.disableflags & DisableBit.CONTACT:
    for condim in (1, 3, 4, 6):
      n = int((dim == condim).sum())
      if condim == 1:
        efc_types += [ConstraintType.CONTACT_FRICTIONLESS] * n
      elif m.opt.cone == ConeType.PYRAMIDAL:
        efc_types += [ConstraintType.CONTACT_PYRAMIDAL] * (condim - 1) * 2 * n
      elif m.opt.cone == ConeType.ELLIPTIC:
        efc_types += [ConstraintType.CONTACT_ELLIPTIC] * condim * n
      else:
        raise ValueError(f'Unknown cone: {m.opt.cone}')

  return np.array(efc_types)


def make_efc_address(
    m: Union[Model, mujoco.MjModel], dim: np.ndarray, efc_type: np.ndarray
) -> np.ndarray:
  """Returns efc_address that maps contacts to constraint row address."""
  offsets = np.array([0], dtype=int)
  for condim in (1, 3, 4, 6):
    n = (dim == condim).sum()
    if n == 0:
      continue
    if condim == 1:
      offsets = np.concatenate((offsets, [1] * n))
    elif m.opt.cone == ConeType.PYRAMIDAL:
      offsets = np.concatenate((offsets, [(condim - 1) * 2] * n))
    elif m.opt.cone == ConeType.ELLIPTIC:
      offsets = np.concatenate((offsets, [condim] * n))
    else:
      raise ValueError(f'Unknown cone: {m.opt.cone}')

  _, _, _, nc = counts(efc_type)
  address = efc_type.size - nc + np.cumsum(offsets)[:-1]

  return address


def make_constraint(m: Model, d: Data) -> Data:
  """Creates constraint jacobians and other supporting data."""

  if m.opt.disableflags & DisableBit.CONSTRAINT:
    efcs = ()
  else:
    efcs = (
        _efc_equality_connect(m, d),
        _efc_equality_weld(m, d),
        _efc_equality_joint(m, d),
        _efc_equality_tendon(m, d),
        _efc_friction(m, d),
        _efc_limit_ball(m, d),
        _efc_limit_slide_hinge(m, d),
        _efc_limit_tendon(m, d),
        _efc_contact_frictionless(m, d),
    )
    if m.opt.cone == ConeType.ELLIPTIC:
      con_fn = _efc_contact_elliptic
    else:
      con_fn = _efc_contact_pyramidal
    efcs += tuple(con_fn(m, d, dim) for dim in (3, 4, 6))
    efcs = tuple(efc for efc in efcs if efc is not None)

  if not efcs:
    z = mx.array([])
    d = d.tree_replace({'_impl.efc_J': mx.zeros((0, m.nv))})
    d = d.tree_replace({
        '_impl.efc_D': z,
        '_impl.efc_aref': z,
        '_impl.efc_frictionloss': z,
        '_impl.efc_pos': z,
        '_impl.efc_margin': z,
    })
    return d

  efc = _tree_map(lambda *x: mx.concatenate(x, axis=0), *efcs)

  # vmap fn -> explicit loop
  def _fn_single(efc_J, efc_pos_aref, efc_pos_imp, efc_invweight, efc_solref,
                  efc_solimp, efc_margin, efc_frictionloss):
    k, b, imp = _kbi(m, efc_solref, efc_solimp, efc_pos_imp)
    r = mx.maximum(efc_invweight * (1 - imp) / imp, mx.array(mujoco.mjMINVAL))
    aref = -b * (efc_J @ d.qvel) - k * imp * efc_pos_aref
    return aref, r, efc_pos_aref + efc_margin, efc_margin, efc_frictionloss

  n_efc = efc.J.shape[0]
  arefs, rs, poss, margins, flosses = [], [], [], [], []
  for i in range(n_efc):
    a, r, p, mg, fl = _fn_single(
        efc.J[i], efc.pos_aref[i], efc.pos_imp[i], efc.invweight[i],
        efc.solref[i], efc.solimp[i], efc.margin[i], efc.frictionloss[i],
    )
    arefs.append(a)
    rs.append(r)
    poss.append(p)
    margins.append(mg)
    flosses.append(fl)

  aref = mx.stack(arefs)
  r = mx.stack(rs)
  pos = mx.stack(poss)
  margin = mx.stack(margins)
  frictionloss = mx.stack(flosses)

  d = d.tree_replace({
      '_impl.efc_J': efc.J,
      '_impl.efc_D': 1 / r,
      '_impl.efc_aref': aref,
      '_impl.efc_pos': pos,
      '_impl.efc_margin': margin,
  })
  d = d.tree_replace({'_impl.efc_frictionloss': frictionloss})

  return d


# ---------------------------------------------------------------------------
# Utility: set a single index or slice in an MLX array (replaces .at[].set)
# ---------------------------------------------------------------------------

def _set_at(arr: mx.array, idx: int, val: mx.array) -> mx.array:
  """Set arr[idx] = val, returning a new array (MLX arrays are immutable)."""
  # Build a one-hot mask and blend
  n = arr.shape[0]
  mask = mx.arange(n) == idx
  if val.ndim == 0:
    return mx.where(mask, val, arr)
  return mx.where(mask, val, arr)


def _set_at_slice(arr: mx.array, start: int, stop: int, vals: mx.array) -> mx.array:
  """Set arr[start:stop] = vals, returning a new array."""
  n = arr.shape[0]
  indices = mx.arange(n)
  mask = (indices >= start) & (indices < stop)
  # Expand vals into full array
  full_vals = mx.zeros_like(arr)
  # Place vals at the right offset
  for i in range(stop - start):
    idx = start + i
    full_vals = mx.where(mx.arange(n) == idx, vals[i], full_vals)
  return mx.where(mask, full_vals, arr)
