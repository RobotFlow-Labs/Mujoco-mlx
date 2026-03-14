# Copyright 2025 DeepMind Technologies Limited
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
"""Derivative functions."""

from typing import Optional

import mlx.core as mx
import numpy as np
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.types import BiasType
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DataMLX
from mujoco.mjx_mlx._src.types import DisableBit
from mujoco.mjx_mlx._src.types import DynType
from mujoco.mjx_mlx._src.types import GainType
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import ModelMLX
from mujoco.mjx_mlx._src.types import OptionMLX
# pylint: enable=g-importing-member

def deriv_smooth_vel(m: Model, d: Data) -> Optional[mx.array]:
  """Analytical derivative of smooth forces w.r.t. velocities."""
  if (
      not isinstance(m._impl, ModelMLX)
      or not isinstance(d._impl, DataMLX)
      or not isinstance(m.opt._impl, OptionMLX)
  ):
    raise ValueError('deriv_smooth_vel requires MLX MJX implementation.')

  qderiv = None

  # qDeriv += d qfrc_actuator / d qvel
  if not m.opt.disableflags & DisableBit.ACTUATION:
    affine_bias = m.actuator_biastype == BiasType.AFFINE
    bias_vel = m.actuator_biasprm[:, 2] * affine_bias
    affine_gain = m.actuator_gaintype == GainType.AFFINE
    gain_vel = m.actuator_gainprm[:, 2] * affine_gain
    ctrl_np = np.array(d.ctrl)
    ctrl_np[m.actuator_dyntype != DynType.NONE] = np.array(d.act)
    ctrl = mx.array(ctrl_np)
    vel = bias_vel + gain_vel * ctrl
    qderiv = d._impl.actuator_moment.T @ (
        d._impl.actuator_moment * vel[:, None]
    )

  # qDeriv += d qfrc_passive / d qvel
  if not m.opt.disableflags & DisableBit.DAMPER:
    if qderiv is None:
      qderiv = -mx.diag(m.dof_damping)
    else:
      qderiv -= mx.diag(m.dof_damping)
    if m.ntendon:
      qderiv -= d._impl.ten_J.T @ mx.diag(m.tendon_damping) @ d._impl.ten_J

  if not m.opt.disableflags & (DisableBit.DAMPER | DisableBit.SPRING):
    # TODO(robotics-simulation): fluid drag model
    if m.opt._impl.has_fluid_params:  # pytype: disable=attribute-error
      raise NotImplementedError('fluid drag not supported for implicitfast')

  # TODO(team): rne derivative

  return qderiv
