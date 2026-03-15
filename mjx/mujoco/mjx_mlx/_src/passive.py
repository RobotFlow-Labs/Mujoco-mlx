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
"""Passive forces (MLX port)."""

from typing import Tuple

import numpy as np
import mlx.core as mx
from mujoco.mjx_mlx._src import math
from mujoco.mjx_mlx._src import scan
from mujoco.mjx_mlx._src import support
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DataMLX
from mujoco.mjx_mlx._src.types import DisableBit
from mujoco.mjx_mlx._src.types import JointType
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import ModelMLX
from mujoco.mjx_mlx._src.types import OptionMLX


def _spring_damper(m: Model, d: Data) -> mx.array:
  """Applies joint level spring and damping forces."""
  # Early return if no springs or damping
  stiff = np.array(m.jnt_stiffness) if hasattr(m, 'jnt_stiffness') else np.zeros(0)
  damp = np.array(m.dof_damping) if hasattr(m, 'dof_damping') else np.zeros(0)
  if not np.any(stiff != 0) and not np.any(damp != 0):
    return mx.zeros((m.nv,))

  def fn(jnt_typs, stiffness, qpos_spring, qpos):
    qpos_i = 0
    qfrcs = []
    for i in range(len(jnt_typs)):
      jnt_typ = JointType(jnt_typs[i])
      q = qpos[qpos_i : qpos_i + jnt_typ.qpos_width()]
      qs = qpos_spring[qpos_i : qpos_i + jnt_typ.qpos_width()]
      qfrc = mx.zeros((jnt_typ.dof_width(),))
      if jnt_typ == JointType.FREE:
        # linear spring on position
        lin_frc = -stiffness[i] * (q[:3] - qs[:3])
        # rotational spring on orientation
        rot_frc = -stiffness[i] * math.quat_sub(q[3:7], qs[3:7])
        qfrc = mx.concatenate([lin_frc, rot_frc])
      elif jnt_typ == JointType.BALL:
        qfrc = -stiffness[i] * math.quat_sub(q, qs)
      elif jnt_typ in (JointType.SLIDE, JointType.HINGE):
        qfrc = -stiffness[i] * (q - qs)
      else:
        raise RuntimeError(f'unrecognized joint type: {jnt_typ}')
      qfrcs.append(qfrc)
      qpos_i += jnt_typ.qpos_width()
    return mx.concatenate(qfrcs) if qfrcs else mx.zeros((0,))

  # dof-level springs
  qfrc = mx.zeros((m.nv,))
  has_springs = np.any(np.array(m.jnt_stiffness) != 0) if hasattr(m, 'jnt_stiffness') else False
  if not m.opt.disableflags & DisableBit.SPRING and has_springs:
    qfrc = scan.flat(
        m,
        fn,
        'jjqq',
        'v',
        m.jnt_type,
        m.jnt_stiffness,
        m.qpos_spring,
        d.qpos,
    )

  # dof-level dampers
  if not m.opt.disableflags & DisableBit.DAMPER:
    qfrc = qfrc - m.dof_damping * d.qvel

  # tendon-level springs
  if not m.opt.disableflags & DisableBit.SPRING:
    spring_T = mx.transpose(m.tendon_lengthspring)  # (2, ntendon)
    below = spring_T[0] - d.ten_length
    above = spring_T[1] - d.ten_length
    frc_spring = mx.where(below > 0, m.tendon_stiffness * below, 0)
    frc_spring = mx.where(above < 0, m.tendon_stiffness * above, frc_spring)
  else:
    frc_spring = mx.zeros((m.ntendon,))

  # tendon-level dampers
  if not m.opt.disableflags & DisableBit.DAMPER:
    frc_damper = -m.tendon_damping * (d._impl or d).ten_velocity
  else:
    frc_damper = mx.zeros((m.ntendon,))

  qfrc = qfrc + (d._impl or d).ten_J.T @ (frc_spring + frc_damper)

  return qfrc


def _gravcomp(m: Model, d: Data) -> mx.array:
  """Applies body-level gravity compensation."""
  force = -m.opt.gravity * (m.body_mass * m.body_gravcomp)[:, None]

  # Loop over bodies instead of vmap
  qfrc = mx.zeros((m.nv,))
  for body_id in range(m.nbody):
    jacp, _ = support.jac(m, d, d.xipos[body_id], body_id)
    qfrc = qfrc + jacp @ force[body_id]

  return qfrc


def _inertia_box_fluid_model(
    m: Model,
    inertia: mx.array,
    mass: mx.array,
    root_com: mx.array,
    xipos: mx.array,
    ximat: mx.array,
    cvel: mx.array,
) -> Tuple[mx.array, mx.array]:
  """Fluid forces based on inertia-box approximation."""
  box = mx.broadcast_to(inertia[None, :], (3, 3)).copy()
  box = box * (mx.ones((3, 3)) - 2 * mx.eye(3))
  box = 6.0 * mx.clip(mx.sum(box, axis=-1), a_min=1e-12)
  box = mx.sqrt(box / mx.maximum(mass, mx.array(1e-12))) * (mass > 0.0)

  # transform to local coordinate frame
  offset = xipos - root_com
  lvel = math.transform_motion(cvel, offset, ximat)
  lwind = ximat.T @ m.opt.wind
  lvel = mx.concatenate([lvel[:3], lvel[3:] - lwind])

  # set viscous force and torque
  diam = mx.mean(box)
  lfrc_ang = lvel[:3] * -mx.array(3.141592653589793) * diam**3 * m.opt.viscosity
  lfrc_vel = lvel[3:] * -3.0 * mx.array(3.141592653589793) * diam * m.opt.viscosity

  # add lift and drag force and torque
  scale_vel = mx.array([box[1] * box[2], box[0] * box[2], box[0] * box[1]])
  scale_ang = mx.array([
      box[0] * (box[1] ** 4 + box[2] ** 4),
      box[1] * (box[0] ** 4 + box[2] ** 4),
      box[2] * (box[0] ** 4 + box[1] ** 4),
  ])
  lfrc_vel = lfrc_vel - 0.5 * m.opt.density * scale_vel * mx.abs(lvel[3:]) * lvel[3:]
  lfrc_ang = lfrc_ang - (
      1.0 * m.opt.density * scale_ang * mx.abs(lvel[:3]) * lvel[:3] / 64.0
  )

  # rotate to global orientation: lfrc -> bfrc
  force = ximat @ lfrc_vel
  torque = ximat @ lfrc_ang

  return force, torque


def _fluid(m: Model, d: Data) -> mx.array:
  """Applies body-level viscosity, lift and drag."""
  import numpy as np  # for root_com indexing

  qfrc = mx.zeros((m.nv,))
  for i in range(m.nbody):
    root_com_i = d.subtree_com[int(m.body_rootid[i])]
    force, torque = _inertia_box_fluid_model(
        m,
        m.body_inertia[i],
        m.body_mass[i],
        root_com_i,
        d.xipos[i],
        d.ximat[i],
        d.cvel[i],
    )
    qfrc_i = support.apply_ft(m, d, force, torque, d.xipos[i], i)
    qfrc = qfrc + qfrc_i

  return qfrc


def passive(m: Model, d: Data) -> Data:
  """Adds all passive forces."""

  if m.opt.disableflags & (DisableBit.SPRING | DisableBit.DAMPER):
    return d.replace(
        qfrc_passive=mx.zeros((m.nv,)),
        qfrc_gravcomp=mx.zeros((m.nv,)),
    )

  qfrc_passive = _spring_damper(m, d)
  qfrc_gravcomp = mx.zeros((m.nv,))

  if m.ngravcomp and not m.opt.disableflags & DisableBit.GRAVITY:
    qfrc_gravcomp = _gravcomp(m, d)
    # add gravcomp unless added via actuators
    qfrc_passive = qfrc_passive + qfrc_gravcomp * (
        1 - m.jnt_actgravcomp[m.dof_jntid]
    )

  has_fluid = getattr(m.opt._impl, 'has_fluid_params', False) if m.opt._impl else (m.opt.density > 0 or m.opt.viscosity > 0)
  if has_fluid:
    qfrc_passive = qfrc_passive + _fluid(m, d)

  d = d.replace(qfrc_passive=qfrc_passive, qfrc_gravcomp=qfrc_gravcomp)
  return d
