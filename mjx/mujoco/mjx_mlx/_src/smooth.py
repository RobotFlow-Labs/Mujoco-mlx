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
"""Smooth dynamics functions (MLX port stub).

Full port coming in a later phase. These stubs provide the correct
signatures so that forward.py, passive.py, and inverse.py can be imported.
"""

import mlx.core as mx
from mujoco.mjx_mlx._src.types import Data, Model


def kinematics(m: Model, d: Data) -> Data:
  """Forward kinematics."""
  raise NotImplementedError('smooth.kinematics not yet ported to MLX')


def com_pos(m: Model, d: Data) -> Data:
  """Center of mass position."""
  raise NotImplementedError('smooth.com_pos not yet ported to MLX')


def camlight(m: Model, d: Data) -> Data:
  """Camera and light computations."""
  raise NotImplementedError('smooth.camlight not yet ported to MLX')


def tendon(m: Model, d: Data) -> Data:
  """Tendon kinematics."""
  raise NotImplementedError('smooth.tendon not yet ported to MLX')


def crb(m: Model, d: Data) -> Data:
  """Composite rigid body algorithm."""
  raise NotImplementedError('smooth.crb not yet ported to MLX')


def tendon_armature(m: Model, d: Data) -> Data:
  """Add tendon armature to mass matrix."""
  raise NotImplementedError('smooth.tendon_armature not yet ported to MLX')


def factor_m(m: Model, d: Data) -> Data:
  """Factorize the mass matrix."""
  raise NotImplementedError('smooth.factor_m not yet ported to MLX')


def transmission(m: Model, d: Data) -> Data:
  """Compute actuator transmission."""
  raise NotImplementedError('smooth.transmission not yet ported to MLX')


def com_vel(m: Model, d: Data) -> Data:
  """Center of mass velocity."""
  raise NotImplementedError('smooth.com_vel not yet ported to MLX')


def rne(m: Model, d: Data) -> Data:
  """Recursive Newton-Euler algorithm."""
  raise NotImplementedError('smooth.rne not yet ported to MLX')


def tendon_bias(m: Model, d: Data) -> Data:
  """Tendon bias forces."""
  raise NotImplementedError('smooth.tendon_bias not yet ported to MLX')


def solve_m(m: Model, d: Data, rhs: mx.array) -> mx.array:
  """Solve M * x = rhs for x."""
  raise NotImplementedError('smooth.solve_m not yet ported to MLX')
